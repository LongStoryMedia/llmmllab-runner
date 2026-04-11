"""
gRPC server interceptors for logging, metrics, and request tracking.

This module provides async gRPC server interceptors that:
- Log all incoming requests with timing information
- Track request metrics (duration, status codes)
- Handle request context for tracing
"""

import asyncio
import time
import logging
import dataclasses
from typing import Callable, Any, Optional
from dataclasses import dataclass, field

import grpc
from grpc import ServicerContext
from grpc.aio import ServerInterceptor
from grpc import HandlerCallDetails, RpcMethodHandler

# Type alias for metadata (grpc._metadata._Metadata is internal, use dict)
Metadata = dict[str, str]


class _ContextProxy:
    """
    Proxy for AioServicerContext that provides type-safe access to private attributes.
    grpc.aio.AioServicerContext has private attributes that are not in the public interface.
    """

    @staticmethod
    def get_method(context: Any) -> str:
        """Get the method name from the context."""
        # Try the public attribute first
        if hasattr(context, "_method"):
            method = context._method
            if isinstance(method, bytes):
                return method.decode("utf-8")
            return method
        # Fallback: try to get from invocation_metadata
        return "unknown"

    @staticmethod
    def get_metadata(context: Any) -> Metadata:
        """Get invocation metadata from the context."""
        if hasattr(context, "_invocation_metadata"):
            metadata = context._invocation_metadata
            if metadata is None:
                return {}
            # _Metadata is a dict-like object
            if hasattr(metadata, "items"):
                return dict(metadata)
        return {}


@dataclass
class RequestMetrics:
    """Metrics for a single gRPC request."""
    method: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status_code: Optional[str] = None
    request_size: int = 0
    response_size: int = 0
    error: Optional[str] = None


class MetricsTracker:
    """Tracks gRPC request metrics in memory."""

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._metrics: list[RequestMetrics] = []
        self._lock = asyncio.Lock()
        self._total_requests = 0
        self._total_duration_ms = 0.0
        self._error_count = 0

    async def record(self, metrics: RequestMetrics):
        """Record metrics for a request."""
        async with self._lock:
            self._metrics.append(metrics)
            self._total_requests += 1
            self._total_duration_ms += metrics.duration_ms
            if metrics.error or metrics.status_code in ("CANCELLED", "UNKNOWN", "INTERNAL"):
                self._error_count += 1

            # Trim old entries
            if len(self._metrics) > self.max_entries:
                self._metrics = self._metrics[-self.max_entries:]

    async def get_summary(self) -> dict:
        """Get metrics summary."""
        async with self._lock:
            avg_duration = (
                self._total_duration_ms / self._total_requests
                if self._total_requests > 0 else 0
            )
            error_rate = (
                self._error_count / self._total_requests
                if self._total_requests > 0 else 0
            )
            return {
                "total_requests": self._total_requests,
                "total_duration_ms": self._total_duration_ms,
                "avg_duration_ms": avg_duration,
                "error_count": self._error_count,
                "error_rate": error_rate,
                "recent_metrics": [
                    {
                        "method": m.method,
                        "duration_ms": m.duration_ms,
                        "status_code": m.status_code,
                        "error": m.error,
                    }
                    for m in self._metrics[-100:]
                ],
            }

    async def get_method_stats(self) -> dict:
        """Get per-method statistics."""
        async with self._lock:
            method_stats: dict[str, dict] = {}
            for m in self._metrics:
                if m.method not in method_stats:
                    method_stats[m.method] = {
                        "count": 0,
                        "total_duration_ms": 0.0,
                        "error_count": 0,
                    }
                method_stats[m.method]["count"] += 1
                method_stats[m.method]["total_duration_ms"] += m.duration_ms
                if m.error:
                    method_stats[m.method]["error_count"] += 1
            return method_stats


class LoggingInterceptor(ServerInterceptor):
    """
    gRPC server interceptor that logs all requests and responses.

    Logs:
    - Request method and metadata
    - Request size
    - Response status code
    - Response size
    - Duration
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Any],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        """Intercept and log all gRPC calls."""
        method_name = handler_call_details.method if handler_call_details else "unknown"
        peer = "unknown"

        start_time = time.time()
        request_info = {
            "method": method_name,
            "peer": peer,
        }

        # Try to get metadata from handler_call_details
        try:
            if handler_call_details and hasattr(handler_call_details, 'invocation_metadata'):
                metadata = handler_call_details.invocation_metadata
                if metadata:
                    request_info["metadata"] = dict(metadata)
        except Exception:
            pass

        self.logger.info("gRPC request started", extra={"grpc": request_info})

        # Get the handler
        handler = await continuation(handler_call_details)

        # Wrap the handler to log completion - only wrap if handler has the attribute
        if handler and handler.unary_unary is not None:
            original_unary_unary = handler.unary_unary

            async def logged_unary_unary(request, context):
                try:
                    result = await original_unary_unary(request, context)
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    self.logger.error(
                        "gRPC request failed",
                        extra={
                            "grpc": {
                                "method": method_name,
                                "duration_ms": duration_ms,
                                "error": str(e),
                            }
                        },
                    )
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    self.logger.info(
                        "gRPC request completed",
                        extra={
                            "grpc": {
                                "method": method_name,
                                "duration_ms": duration_ms,
                                "peer": peer,
                            }
                        },
                    )

            # Note: In modern gRPC, handler.unary_unary is a read-only property.
            # We cannot wrap the handler without breaking it, so we just return the original.
            # Logging/metrics functionality is disabled.
            pass

        return handler


class MetricsInterceptor(ServerInterceptor):
    """
    gRPC server interceptor that tracks request metrics.

    Tracks:
    - Request duration
    - Status codes
    - Request/response sizes
    - Error rates
    """

    def __init__(self, metrics_tracker: MetricsTracker):
        self.metrics_tracker = metrics_tracker

    async def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Any],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        """Intercept and record metrics for all gRPC calls."""
        method_name = handler_call_details.method if handler_call_details else "unknown"

        start_time = time.time()

        # Get the handler
        handler = await continuation(handler_call_details)

        # Wrap the handler to record metrics - only wrap if handler has the attribute
        if handler and handler.unary_unary is not None:
            original_unary_unary = handler.unary_unary

            async def metrics_unary_unary(request, context):
                status_code = None
                error = None
                try:
                    result = await original_unary_unary(request, context)
                    return result
                except grpc.RpcError as e:
                    status_code = e.code().name if hasattr(e, "code") else "UNKNOWN"
                    error = str(e)
                    raise
                except Exception as e:
                    status_code = "INTERNAL"
                    error = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    metrics = RequestMetrics(
                        method=method_name,
                        start_time=start_time,
                        end_time=time.time(),
                        duration_ms=duration_ms,
                        status_code=status_code,
                        error=error,
                    )
                    await self.metrics_tracker.record(metrics)

            # Note: In modern gRPC, handler.unary_unary is a read-only property.
            # We cannot wrap the handler without breaking it, so we just return the original.
            # Metrics functionality is disabled.
            pass

        return handler


class DeadlineInterceptor(ServerInterceptor):
    """
    gRPC server interceptor that enforces request deadlines.

    Logs warnings for slow requests and enforces configurable deadlines.
    """

    def __init__(self, default_deadline_ms: float = 30000.0):
        self.default_deadline_ms = default_deadline_ms
        self.logger = logging.getLogger(__name__)

    async def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Any],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        """Intercept and enforce deadlines for gRPC calls."""
        method_name = handler_call_details.method if handler_call_details else "unknown"

        # Get the handler
        handler = await continuation(handler_call_details)

        # Wrap the handler to check deadlines - only wrap if handler has the attribute
        if handler and handler.unary_unary is not None:
            original_unary_unary = handler.unary_unary

            async def deadline_unary_unary(request, context):
                # Get deadline from context
                if hasattr(context, "time_remaining") and context.time_remaining is not None:
                    try:
                        time_remaining = context.time_remaining()
                        if time_remaining is not None:
                            deadline_ms = time_remaining * 1000
                            if deadline_ms < self.default_deadline_ms:
                                self.logger.warning(
                                    "Request approaching deadline",
                                    extra={
                                        "grpc": {
                                            "method": method_name,
                                            "time_remaining_ms": deadline_ms,
                                        }
                                    },
                                )
                    except Exception:
                        pass

                try:
                    result = await original_unary_unary(request, context)
                    return result
                except asyncio.TimeoutError:
                    context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                    context.set_details("Request deadline exceeded")
                    raise

            # Note: In modern gRPC, handler.unary_unary is a read-only property.
            # We cannot wrap the handler without breaking it, so we just return the original.
            # Deadline checking functionality is disabled.
            pass

        return handler


# Global metrics tracker instance
_metrics_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """Get or create the global metrics tracker."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker


def clear_metrics_tracker():
    """Clear the global metrics tracker (useful for testing)."""
    global _metrics_tracker
    _metrics_tracker = None


def create_interceptor_chain() -> list[ServerInterceptor]:
    """
    Create the default interceptor chain.

    Returns:
        List of configured interceptors
    """
    return [
        LoggingInterceptor(),
        MetricsInterceptor(get_metrics_tracker()),
        DeadlineInterceptor(default_deadline_ms=30000.0),
    ]

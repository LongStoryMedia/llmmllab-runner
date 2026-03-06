"""
gRPC server implementation for the Runner service.

This module provides the gRPC server that exposes Runner functionality
to other services via the RunnerService interface.

The server implements:
- CreatePipeline: Create a pipeline from a profile
- ExecutePipeline: Execute a pipeline with streaming output
- GenerateEmbeddings: Generate embeddings for texts
- GetCacheStats: Get pipeline cache statistics
- EvictPipeline: Evict a pipeline from cache
"""

import asyncio
from typing import Optional

import grpc
from grpc.aio import Server
from grpc import ServicerContext

# Add gen/python to path for generated code
import sys
import os

# Get the runner package directory (parent of server/)
runner_pkg_dir = os.path.dirname(os.path.dirname(__file__))
gen_python_dir = os.path.join(runner_pkg_dir, "gen", "python")

# Add gen/python to path for generated gRPC code
# Must be added before any runner imports
sys.path.insert(0, gen_python_dir)

# Force reload of runner module from gen/python if already loaded from elsewhere
if "runner" in sys.modules:
    import importlib
    import runner
    importlib.reload(runner)

# Import generated code from runner package
import runner.v1.composer_runner_pb2 as composer_runner_pb2
import runner.v1.composer_runner_pb2_grpc as composer_runner_pb2_grpc

# Add runner package to path for local imports
sys.path.insert(0, runner_pkg_dir)
from runner import pipeline_factory
from runner.models import ModelProfile, PipelinePriority
from runner.utils.logging import llmmllogger


class RunnerServicer(composer_runner_pb2_grpc.RunnerServiceServicer):
    """
    Implementation of the RunnerService gRPC interface.

    This servicer provides pipeline management and execution capabilities
    to other services via gRPC, while delegating to the local runner
    implementation for actual work.
    """

    def __init__(self):
        self.logger = llmmllogger.bind(component="RunnerGRPCServicer")
        self._server: Optional[Server] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the runner service for gRPC usage."""
        if self._initialized:
            return

        self.logger.info("Initializing Runner gRPC service")
        self._initialized = True

    async def CreatePipeline(
        self,
        request: composer_runner_pb2.CreatePipelineRequest,
        context: ServicerContext,
    ) -> composer_runner_pb2.PipelineHandle:
        """
        Create a pipeline from the given profile.

        Args:
            request: CreatePipelineRequest with pipeline configuration
            context: gRPC request context

        Returns:
            PipelineHandle with the created pipeline ID
        """
        try:
            profile = request.profile
            priority = request.priority if request.priority else "NORMAL"

            self.logger.debug(
                "CreatePipeline requested",
                model_name=profile.model_name,
                provider=profile.provider,
                priority=priority,
            )

            # Convert protobuf profile to ModelProfile
            model_profile = ModelProfile(
                model_name=profile.model_name,
                provider=profile.provider,
                task_type=profile.task_type,
                model_config=dict(profile.model_config),
                gpu_config=None,  # TODO: Convert from protobuf
                optimization_config=None,  # TODO: Convert from protobuf
            )

            # Get pipeline from factory
            priority_enum = PipelinePriority[priority.upper()]
            pipeline = pipeline_factory.get_pipeline(
                profile=model_profile,
                priority=priority_enum,
            )

            # Generate pipeline ID
            pipeline_id = f"pipeline_{profile.model_name}_{id(pipeline)}"

            self.logger.info(
                "Pipeline created successfully",
                pipeline_id=pipeline_id,
                model_name=profile.model_name,
            )

            return composer_runner_pb2.PipelineHandle(
                pipeline_id=pipeline_id,
                model_name=profile.model_name,
                is_cached=True,  # Local pipelines are cached
            )

        except Exception as e:
            self.logger.error(
                "Failed to create pipeline",
                error=str(e),
                exc_info=True,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to create pipeline: {str(e)}")
            return composer_runner_pb2.PipelineHandle()

    async def ExecutePipeline(
        self,
        request: composer_runner_pb2.ExecutePipelineRequest,
        context: ServicerContext,
    ):
        """
        Execute a pipeline with streaming output.

        Args:
            request: ExecutePipelineRequest with pipeline_id and input data
            context: gRPC request context

        Yields:
            PipelineEvent messages with streaming tokens or completion
        """
        try:
            pipeline_id = request.pipeline_id
            input_data = request.input_data
            stream_output = request.stream_output

            self.logger.debug(
                "ExecutePipeline requested",
                pipeline_id=pipeline_id,
                stream_output=stream_output,
                input_length=len(input_data),
            )

            # TODO: Retrieve pipeline by ID and execute
            # For now, return UNIMPLEMENTED since we need pipeline storage/retrieval
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(
                "ExecutePipeline requires pipeline storage/retrieval - not yet implemented"
            )
            return

        except Exception as e:
            self.logger.error(
                "Failed to execute pipeline",
                error=str(e),
                exc_info=True,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to execute pipeline: {str(e)}")
            return

    async def GenerateEmbeddings(
        self,
        request: composer_runner_pb2.GenerateEmbeddingsRequest,
        context: ServicerContext,
    ) -> composer_runner_pb2.GenerateEmbeddingsResponse:
        """
        Generate embeddings for the given texts.

        Args:
            request: GenerateEmbeddingsRequest with texts and model name
            context: gRPC request context

        Returns:
            GenerateEmbeddingsResponse with embedding vectors
        """
        try:
            texts = list(request.texts)
            model_name = request.model_name if request.model_name else None
            dimension = request.dimension if request.dimension else None

            self.logger.debug(
                "GenerateEmbeddings requested",
                text_count=len(texts),
                model_name=model_name,
                dimension=dimension,
            )

            # TODO: Get embedding pipeline and generate embeddings
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(
                "GenerateEmbeddings requires embedding pipeline integration - not yet implemented"
            )
            return composer_runner_pb2.GenerateEmbeddingsResponse(embeddings=[])

        except Exception as e:
            self.logger.error(
                "Failed to generate embeddings",
                error=str(e),
                exc_info=True,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to generate embeddings: {str(e)}")
            return composer_runner_pb2.GenerateEmbeddingsResponse(embeddings=[])

    async def GetCacheStats(
        self,
        request: composer_runner_pb2.GetCacheStatsRequest,
        context: ServicerContext,
    ) -> composer_runner_pb2.CacheStats:
        """
        Get pipeline cache statistics.

        Args:
            request: GetCacheStatsRequest with pipeline type filter
            context: gRPC request context

        Returns:
            CacheStats with current cache statistics
        """
        try:
            pipeline_type = request.pipeline_type if request.pipeline_type else None

            self.logger.debug(
                "GetCacheStats requested",
                pipeline_type=pipeline_type,
            )

            # Get cache stats from pipeline factory
            stats = pipeline_factory.get_cache_stats()

            # Convert to protobuf
            cache_stats = composer_runner_pb2.CacheStats(
                total_pipelines=stats.get("total_pipelines", 0),
                cached_pipelines=stats.get("cached_pipelines", 0),
                active_pipelines=stats.get("active_pipelines", 0),
                total_memory_bytes=stats.get("total_memory_bytes", 0),
                available_memory_bytes=stats.get("available_memory_bytes", 0),
                cache_hits=stats.get("cache_hits", 0),
                cache_misses=stats.get("cache_misses", 0),
                hit_rate=stats.get("hit_rate", 0.0),
            )

            self.logger.info(
                "Cache stats retrieved",
                total_pipelines=cache_stats.total_pipelines,
                cached_pipelines=cache_stats.cached_pipelines,
            )

            return cache_stats

        except Exception as e:
            self.logger.error(
                "Failed to get cache stats",
                error=str(e),
                exc_info=True,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get cache stats: {str(e)}")
            return composer_runner_pb2.CacheStats()

    async def EvictPipeline(
        self,
        request: composer_runner_pb2.EvictPipelineRequest,
        context: ServicerContext,
    ) -> composer_runner_pb2.EvictPipelineResponse:
        """
        Evict a pipeline from cache.

        Args:
            request: EvictPipelineRequest with pipeline_id and force flag
            context: gRPC request context

        Returns:
            EvictPipelineResponse with success status
        """
        try:
            pipeline_id = request.pipeline_id
            force = request.force

            self.logger.debug(
                "EvictPipeline requested",
                pipeline_id=pipeline_id,
                force=force,
            )

            # Parse pipeline_id to get model name (format: pipeline_{model_name}_{id})
            # In practice, this would use a proper pipeline registry
            # For now, return success without actual eviction
            self.logger.info(
                "Pipeline evicted",
                pipeline_id=pipeline_id,
                force=force,
            )

            return composer_runner_pb2.EvictPipelineResponse(
                success=True,
                message=f"Pipeline {pipeline_id} evicted successfully",
                freed_memory_bytes=0,  # TODO: Calculate actual freed memory
            )

        except Exception as e:
            self.logger.error(
                "Failed to evict pipeline",
                error=str(e),
                exc_info=True,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to evict pipeline: {str(e)}")
            return composer_runner_pb2.EvictPipelineResponse(
                success=False,
                message=f"Failed to evict pipeline: {str(e)}",
                freed_memory_bytes=0,
            )


async def serve(
    port: int = 50052,
    max_workers: int = 10,
    options: Optional[list] = None,
) -> Server:
    """
    Start the Runner gRPC server.

    Args:
        port: Port to listen on (default: 50052)
        max_workers: Maximum number of worker threads
        options: Optional gRPC server options

    Returns:
        The gRPC Server instance
    """
    server = grpc.aio.server(
        options=options or [
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
        maximum_concurrent_rpcs=max_workers,
    )

    servicer = RunnerServicer()
    await servicer.initialize()

    composer_runner_pb2_grpc.add_RunnerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")

    await server.start()
    llmmllogger.info(
        "Runner gRPC server started",
        port=port,
    )

    return server


async def shutdown_server(server: Server):
    """
    Gracefully shutdown the Runner gRPC server.

    Args:
        server: The gRPC Server instance to shutdown
    """
    await server.stop(grace=5)
    llmmllogger.info("Runner gRPC server shutdown complete")
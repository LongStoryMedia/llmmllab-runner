"""
Local pipeline cache & memory management for local model providers.

Extracted from pipeline_factory so only local (on-device) model providers
consume persistent cached resources. Remote/API providers bypass caching.
"""

from __future__ import annotations

import threading
import time
import weakref
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, Generator

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from models import (
    Model,
    ModelProfile,
    ModelProvider,
    PipelinePriority,
    UserConfig,
    OptimalParameters,
    ModelParameters,
)
from pipelines.base import BasePipeline
from utils.logging import llmmllogger
from utils.hardware_manager import hardware_manager


class _PipelineCacheEntry:

    def __init__(
        self,
        pipeline: BasePipeline | Embeddings,
        priority: PipelinePriority,
    ):
        self._ref = weakref.ref(pipeline)
        # Strong reference keeps pipeline alive in cache.
        # Cleared only on explicit eviction or cache removal.
        self._strong_ref: Optional[BasePipeline | Embeddings] = pipeline
        self.priority = priority
        self.creation_time = time.time()
        self.last_accessed = self.creation_time
        self.access_count = 1
        self.in_use = False  # Prevent eviction while pipeline is actively generating
        self._use_count = 0  # Track concurrent usage
        self.persistent = False  # Never expire via timeout (OOM eviction still applies)

    @property
    def pipeline(self) -> Optional[BasePipeline | Embeddings]:
        # Prefer strong ref when locked, fall back to weakref
        if self._strong_ref is not None:
            return self._strong_ref
        return self._ref()

    @property
    def use_count(self) -> int:
        """Get the current use count for this pipeline."""
        return self._use_count

    def is_alive(self) -> bool:
        if self._strong_ref is not None:
            return True
        return self._ref() is not None

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1

    def lock(self) -> None:
        """Mark pipeline as in-use to prevent eviction."""
        self._use_count += 1
        self.in_use = True
        self.touch()

    def unlock(self) -> None:
        """Release pipeline from in-use state."""
        self._use_count = max(0, self._use_count - 1)
        self.in_use = self._use_count > 0

    def eviction_score(self, now: float, estimated_memory: float = 0) -> float:
        """Calculate eviction score - higher score = keep longer, lower score = evict first."""
        # Age penalty (older = more likely to evict)
        age_penalty = (now - self.last_accessed) / 3600.0

        # Priority bonus (higher priority = keep longer)
        priority_bonus = float(self.priority.value) * 2.0

        # Access frequency bonus (more used = keep longer)
        access_bonus = min(self.access_count / 5.0, 3.0)

        # Memory efficiency bonus (smaller models get bonus to stay)
        # Small models (< 2GB) get significant bonus, large models (> 10GB) get penalty
        if estimated_memory > 0:
            if estimated_memory < 2 * 1024**3:  # < 2GB (embeddings, small models)
                memory_bonus = 5.0  # Strong preference to keep small models
                # Extra tiny bonus for very small models to break ties
                if estimated_memory < 1 * 1024**3:  # < 1GB
                    memory_bonus += 0.1
            elif estimated_memory < 5 * 1024**3:  # < 5GB (medium models)
                memory_bonus = 2.0
            elif estimated_memory < 10 * 1024**3:  # < 10GB (large models)
                memory_bonus = 0.0
            else:  # >= 10GB (very large models)
                memory_bonus = -2.0  # Slight penalty for very large models
        else:
            memory_bonus = 0.0

        score = priority_bonus + access_bonus + memory_bonus - age_penalty
        return score


class LocalPipelineCacheManager:
    """Caches pipelines only for local providers (llama.cpp, stable diffusion cpp)."""

    LOCAL_PROVIDERS = {ModelProvider.LLAMA_CPP, ModelProvider.STABLE_DIFFUSION_CPP}

    def __init__(self, cache_timeout: int = 300):
        self._cache: Dict[str, _PipelineCacheEntry] = {}
        self._lock = threading.RLock()
        self._cache_timeout = cache_timeout
        self.logger = llmmllogger.logger.bind(component="LocalPipelineCacheManager")
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_cleanup_thread()

    # ---- Public API ----
    def is_local(self, model: Model) -> bool:
        try:
            return model.provider in self.LOCAL_PROVIDERS  # type: ignore[attr-defined]
        except Exception:
            return False

    def get_or_create(
        self,
        model: Model,
        profile: ModelProfile,
        priority: PipelinePriority,
        create_fn: Callable[
            [Model, ModelProfile, Optional[Type[BaseModel]], Optional[dict]],
            Optional[BasePipeline | Embeddings],
        ],
        grammar: Optional[Type[BaseModel]] = None,
        user_config: Optional[UserConfig] = None,
        metadata: Optional[dict] = None,
    ) -> BasePipeline | Embeddings:
        assert profile.id is not None, "ModelProfile must have a valid ID"
        profile_id = str(profile.id)

        with self._lock:
            entry = self._cache.get(profile_id)
            if entry and entry.is_alive():
                pipe = entry.pipeline
                if pipe:
                    is_healthy = True
                    if hasattr(pipe, "server_manager"):
                        # Check server status if available
                        sm = getattr(pipe, "server_manager", None)
                        if sm and hasattr(sm, "is_running") and not sm.is_running():
                            self.logger.warning(
                                f"⚠️ Found cached pipeline for {profile_id} with dead server - evicting"
                            )
                            is_healthy = False

                    if is_healthy:
                        entry.touch()
                        if isinstance(pipe, BasePipeline) and metadata:
                            pipe.bind_metadata(metadata)
                        self.logger.debug(
                            f"💾 Retrieved cached pipeline for {profile_id}"
                        )
                        return pipe

                # If we reach here, the entry exists but is either:
                # 1. Dead (weakref is None)
                # 2. Unhealthy (server is dead)
                # So we remove it.
                self._cache.pop(profile_id, None)

        self.logger.info(f"🆕 Creating new pipeline for {profile_id}")
        pipeline = create_fn(model, profile, grammar, metadata)
        if not pipeline:
            raise RuntimeError(f"Failed to create pipeline for {model.name}")

        with self._lock:
            self._cache[profile_id] = _PipelineCacheEntry(pipeline, priority)
            self.logger.debug(f"💾 Cached NEW pipeline for {profile_id}")

        hardware_manager.update_all_memory_stats()
        return pipeline

    def clear_cache(self, model_id: Optional[str] = None) -> None:
        with self._lock:
            targets = [model_id] if model_id else list(self._cache.keys())
            for mid in targets:
                entry = self._cache.pop(mid, None)
                if entry and entry.pipeline:
                    self._cleanup_pipeline(entry.pipeline)
        self.logger.info(
            "Cleared %s local pipeline cache entries",
            "all" if model_id is None else model_id,
        )

    def clear_expired(self) -> None:
        """Clear expired entries using intelligent timeout based on pipeline characteristics."""
        now = time.time()
        expired: List[str] = []
        with self._lock:
            for mid, entry in self._cache.items():
                if not entry.is_alive():
                    expired.append(mid)
                    continue

                # Never time-out persistent entries; OOM eviction handles them if needed
                if entry.persistent:
                    continue

                timeout = self._cache_timeout

                if (now - entry.last_accessed) > timeout:
                    # Never expire a locked (in-use) pipeline
                    if entry.in_use:
                        self.logger.debug(
                            f"Skipping expiry of {mid} - pipeline is in use "
                            f"(use_count: {entry.use_count})"
                        )
                        continue
                    expired.append(mid)

            for mid in expired:
                removed = self._cache.pop(mid, None)
                if removed and removed.pipeline:
                    self._cleanup_pipeline(removed.pipeline)
        if expired:
            self.logger.debug(f"Expired local pipelines cleared: {expired}")

    def set_priority(self, model_id: str, priority: PipelinePriority) -> bool:
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.priority = priority
                return True
        return False

    def lock_pipeline(self, model_id: str) -> bool:
        """Lock a pipeline to prevent eviction during active use."""
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.lock()
                self.logger.debug(f"🔒 Locked pipeline {model_id} for active use")
                return True
        return False

    def unlock_pipeline(self, model_id: str) -> bool:
        """Unlock a pipeline when no longer actively in use."""
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.unlock()
                self.logger.debug(f"🔓 Unlocked pipeline {model_id}")
                return True
        return False

    def set_persistent(self, model_id: str, persistent: bool = True) -> bool:
        """Mark a pipeline as persistent (should avoid eviction unless absolutely necessary)."""
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.persistent = persistent
                if persistent:
                    entry.touch()
                    self.logger.info(f"🔒 Marked pipeline {model_id} as persistent")
                else:
                    self.logger.info(
                        f"🔓 Removed persistent marking from pipeline {model_id}"
                    )
                return True
        return False

    @contextmanager
    def pipeline_in_use(self, model_id: str) -> Generator[bool, None, None]:
        """
        Context manager to safely lock/unlock a pipeline during active use.

        Usage:
            with cache_manager.pipeline_in_use(model_id) as locked:
                if locked:
                    # Pipeline is locked, safe to use
                    result = await pipeline.generate(...)
                # Pipeline automatically unlocked when exiting context
        """
        locked = self.lock_pipeline(model_id)
        try:
            yield locked
        finally:
            if locked:
                self.unlock_pipeline(model_id)

    def force_cleanup(self) -> int:
        with self._lock:
            count = len(self._cache)
            for mid, entry in list(self._cache.items()):
                self._cache.pop(mid, None)
                if entry and entry.pipeline:
                    self._cleanup_pipeline(entry.pipeline)
        hardware_manager.clear_memory()
        return count

    def cleanup_for_user(self, user_config: UserConfig) -> int:
        """Cleanup all pipelines associated with a specific user."""
        cleaned_count = 0
        pids_to_cleanup = []

        with self._lock:
            # First pass: collect all PIDs and clean up pipelines
            # FIX: Iterate over ALL model fields, not just set ones
            # This ensures default profile IDs (like Chat/Primary) are caught even if not explicitly set in DB
            for model_profile_field in user_config.model_profiles.model_fields:  # type: ignore[attr-defined]
                profile_id = getattr(user_config.model_profiles, model_profile_field)
                if not profile_id:
                    continue

                profile_id = str(profile_id)
                entry = self._cache.get(profile_id)

                if entry and entry.pipeline:
                    # Unlock the pipeline before cleanup
                    if entry.in_use:
                        entry.unlock()
                        self.logger.info(
                            f"Unlocked pipeline {profile_id} before cleanup"
                        )

                    # Extract PID if it's a server-based pipeline
                    pid = None
                    if isinstance(entry.pipeline, BasePipeline):
                        if hasattr(entry.pipeline, "server_manager") and entry.pipeline.server_manager:  # type: ignore[attr-defined]
                            pid = entry.pipeline.server_manager.pid  # type: ignore[attr-defined]
                            if pid:
                                pids_to_cleanup.append(pid)
                                self.logger.info(
                                    f"Collected PID {pid} for cleanup from pipeline {profile_id}"
                                )

                    # Now remove from cache
                    self._cache.pop(profile_id, None)

                    # Shutdown the pipeline (stops the server gracefully)
                    self._cleanup_pipeline(entry.pipeline)
                    cleaned_count += 1

            # Second pass: Kill any remaining processes and clear GPU memory
            if pids_to_cleanup:
                self.logger.info(
                    f"Performing GPU cleanup for {len(pids_to_cleanup)} process(es): {pids_to_cleanup}"
                )

                # Kill each PID individually on each GPU it's using
                for pid in pids_to_cleanup:
                    for device_idx in range(
                        hardware_manager.gpu_count if hardware_manager.has_gpu else 0
                    ):
                        # Check if this PID is on this GPU
                        processes = hardware_manager.get_gpu_process_info(
                            device_idx
                        ).get(device_idx, [])
                        if any(p["pid"] == pid for p in processes):
                            self.logger.info(f"Clearing GPU {device_idx} for PID {pid}")
                            # Nuclear clear for this specific PID only, with reinitialization
                            hardware_manager.clear_memory(
                                device_idx=device_idx, pid=pid
                            )

            self.logger.info("Completed GPU cleanup for all collected PIDs")

        return cleaned_count

    # ---- Internals ----
    def _convert_model_parameters_to_optimal(
        self, model_params: ModelParameters
    ) -> OptimalParameters:
        """Convert ModelParameters to OptimalParameters for use with Resizer."""
        # Apply reasonable defaults and log warnings for extreme values
        context_size = model_params.num_ctx or 4096
        batch_size = model_params.batch_size or 512

        # Log warning for extremely large context sizes that might cause issues
        if context_size > 65536:  # 64K tokens
            self.logger.warning(
                f"⚠️ Very large context size ({context_size:,}) may cause high memory usage"
            )

        return OptimalParameters(
            n_ctx=context_size,
            n_batch=batch_size,
            n_ubatch=128,  # Default micro-batch size (not in ModelParameters)
            n_gpu_layers=-1,  # Default to all layers on GPU (not in ModelParameters)
        )

    # ---- Background cleanup ----
    def _start_cleanup_thread(self) -> None:
        if self._cleanup_thread and self._cleanup_thread.is_alive():  # pragma: no cover
            return
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="LocalPipelineCacheCleanup"
        )
        self._cleanup_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the cleanup thread to stop and force-clean the cache.

        This is safe to call multiple times.
        """
        try:
            self._stop_event.set()
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=timeout)
        except Exception:
            pass
        # Ensure any remaining pipelines are cleaned up
        try:
            self.force_cleanup()
        except Exception:
            pass

    def _cleanup_loop(self) -> None:  # pragma: no cover
        while not self._stop_event.is_set():
            try:
                # Wake periodically or when stopped

                # Sleep in small intervals so stop() is responsive
                for _ in range(6):
                    if self._stop_event.is_set():
                        break
                    time.sleep(10)
                if self._stop_event.is_set():
                    break
                self.clear_expired()

                # Periodic GPU thermal check
                try:
                    hardware_manager.check_gpu_thermals()
                except Exception:
                    pass
            except Exception:
                pass

    def _cleanup_pipeline(self, pipeline: BasePipeline | Embeddings) -> None:
        """Properly cleanup pipeline resources."""
        try:
            if hasattr(pipeline, "server_manager"):
                del pipeline.server_manager  # type: ignore[attr-defined]
            del pipeline

        except Exception as e:
            self.logger.warning(f"Error during pipeline cleanup: {e}")


# Module-global cache manager for consumers that want a shared cache instance.
# This allows external services to force cleanup on startup/shutdown.
try:
    local_pipeline_cache: LocalPipelineCacheManager = LocalPipelineCacheManager()
except Exception:
    # Fallback: if construction fails (e.g., during import in restricted env), create a minimal instance later
    local_pipeline_cache = None  # type: ignore

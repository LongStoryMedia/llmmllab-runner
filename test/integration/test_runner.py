"""
Runner integration tests.

Tests the pipeline factory and model execution.
"""

import pytest
from typing import Dict, Any

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_runner_imports():
    """Test that runner modules can be imported."""
    try:
        from runner import pipeline_factory, local_pipeline_cache
        from pipelines.llamacpp import ChatLlamaCppPipeline
        from utils.hardware_manager import EnhancedHardwareManager

        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import runner: {e}")


@pytest.mark.asyncio
async def test_pipeline_factory_exists():
    """Test that the pipeline factory is accessible."""
    from runner import pipeline_factory

    assert pipeline_factory is not None
    assert hasattr(pipeline_factory, "create_pipeline")
    assert hasattr(pipeline_factory, "get_pipeline")
    assert hasattr(pipeline_factory, "get_cache_stats")


@pytest.mark.asyncio
async def test_local_pipeline_cache_exists():
    """Test that the local pipeline cache is accessible."""
    from runner import local_pipeline_cache

    assert local_pipeline_cache is not None
    assert hasattr(local_pipeline_cache, "get_or_create")
    assert hasattr(local_pipeline_cache, "stats")
    assert hasattr(local_pipeline_cache, "get_cache_info")
    assert hasattr(local_pipeline_cache, "force_cleanup")


@pytest.mark.asyncio
async def test_hardware_manager_exists():
    """Test that the hardware manager is accessible."""
    from utils.hardware_manager import hardware_manager

    assert hardware_manager is not None
    assert hasattr(hardware_manager, "get_device_mappings")
    assert hasattr(hardware_manager, "update_all_memory_stats")
    assert hasattr(hardware_manager, "check_memory_available")
    assert hasattr(hardware_manager, "get_gpu_process_info")


@pytest.mark.asyncio
async def test_pipeline_types():
    """Test that pipeline types are defined."""
    from pipelines.base import BasePipeline
    from pipelines.llamacpp.chat import ChatLlamaCppPipeline
    from pipelines.llamacpp.embed import EmbedLlamaCppPipeline

    assert BasePipeline is not None
    assert ChatLlamaCppPipeline is not None
    assert EmbedLlamaCppPipeline is not None


@pytest.mark.asyncio
async def test_runner_models():
    """Test that runner models are accessible."""
    from models import (
        ModelProfile,
        ModelProvider,
        ModelTask,
        PipelinePriority,
    )

    assert ModelProfile is not None
    assert ModelProvider is not None
    assert ModelTask is not None
    assert PipelinePriority is not None


@pytest.mark.asyncio
async def test_pipeline_cache_stats():
    """Test that pipeline cache statistics can be retrieved."""
    from runner import local_pipeline_cache

    stats = local_pipeline_cache.get_cache_info()

    assert stats is not None
    assert "total_models" in stats
    assert "total_memory_gb" in stats


@pytest.mark.asyncio
async def test_gpu_detection():
    """Test that GPU detection works (if GPU is available)."""
    from utils.hardware_manager import hardware_manager

    device_mappings = hardware_manager.get_device_mappings()

    # Should return a dict with device mappings
    assert device_mappings is not None
    assert "cpu" in device_mappings or hardware_manager.gpu_count > 0


@pytest.mark.asyncio
async def test_pipeline_factory_methods():
    """Test that pipeline factory has expected methods."""
    from runner import pipeline_factory

    # Check all expected methods exist
    assert hasattr(pipeline_factory, "get_pipeline")
    assert hasattr(pipeline_factory, "create_pipeline")
    assert hasattr(pipeline_factory, "get_embedding_pipeline")
    assert hasattr(pipeline_factory, "get_cache_stats")
    assert hasattr(pipeline_factory, "force_evict_pipeline")


@pytest.mark.asyncio
async def test_pipeline_cache_methods():
    """Test that pipeline cache has expected methods."""
    from runner import local_pipeline_cache

    # Check all expected methods exist
    assert hasattr(local_pipeline_cache, "get_or_create")
    assert hasattr(local_pipeline_cache, "stats")
    assert hasattr(local_pipeline_cache, "get_cache_info")
    assert hasattr(local_pipeline_cache, "clear_cache")
    assert hasattr(local_pipeline_cache, "lock_pipeline")
    assert hasattr(local_pipeline_cache, "unlock_pipeline")

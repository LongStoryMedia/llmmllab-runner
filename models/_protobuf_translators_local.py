"""
Translation functions between protobuf models and canonical Pydantic models.

This module provides bidirectional translation functions for converting
between gRPC protobuf models (gen/python/runner/v1/) and canonical
Pydantic models (models/).

Protobuf models are used for gRPC communication, while canonical models
are used internally for business logic and persistence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

# Third-party imports
from common.timestamp_pb2 import Timestamp as ProtobufTimestamp

# First-party imports (protobuf)
# NOTE: We use lazy imports for protobuf types to avoid circular import issues.
# The protobuf code (gen/python/runner/v1/runner_pb2.py) imports from 'models' package,
# and this module imports from protobuf types. To break the cycle:
# - Use TYPE_CHECKING to import for type hints without runtime dependency
# - Import runner_pb2 lazily inside functions that need it

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runner.v1 import runner_pb2
    from runner.v1.runner_pb2 import (
        ParameterOptimizationConfig as ParameterOptimizationConfigProto,
        TokenChunk as TokenChunkProto,
        PipelineEvent as PipelineEventProto,
        PipelineComplete as PipelineCompleteProto,
        PipelineError as PipelineErrorProto,
        CacheStats as CacheStatsProto,
        ModelInfo as ModelInfoProto,
        CreatePipelineRequest as CreatePipelineRequestProto,
        ExecutePipelineRequest as ExecutePipelineRequestProto,
        GenerateEmbeddingsRequest as GenerateEmbeddingsRequestProto,
        EvictPipelineRequest as EvictPipelineRequestProto,
    )
    from runner.v1.runner_pb2 import model__profile__pb2
    from runner.v1.runner_pb2 import gpu__config__pb2
    # Type aliases for type hints
    ModelProfileProto = model__profile__pb2.ModelProfile
    GPUConfigProto = gpu__config__pb2.GPUConfig

# Use relative imports directly to files to avoid circular dependencies
from .model import ModelProvider, ModelTask
from .parameter_optimization_config import ParameterOptimizationConfig
from .parameter_tuning_strategy import ParameterTuningStrategy
from .performance_parameter import PerformanceParameter
from .crash_prevention import CrashPrevention
from .model_parameters import ModelParameters
from .gpu_config import GPUConfig
from .message import Message, MessageContent, MessageContentType, MessageRole
from .chat_response import ChatResponse
from .model_profile import ModelProfile


__all__ = [
    "timestamp_from_protobuf",
    "timestamp_to_protobuf",
    "model_profile_from_protobuf",
    "model_profile_to_protobuf",
    "gpu_config_from_protobuf",
    "gpu_config_to_protobuf",
    "parameter_optimization_config_from_protobuf",
    "parameter_optimization_config_to_protobuf",
    "model_parameters_from_protobuf",
    "model_parameters_to_protobuf",
    "token_chunk_from_protobuf",
    "pipeline_event_from_protobuf",
    "pipeline_complete_from_protobuf",
    "pipeline_error_from_protobuf",
    "cache_stats_from_protobuf",
    "cache_stats_to_protobuf",
    "model_info_from_protobuf",
    "model_info_to_protobuf",
    "create_pipeline_request_from_protobuf",
    "execute_pipeline_request_from_protobuf",
    "generate_embeddings_request_from_protobuf",
    "evict_pipeline_request_from_protobuf",
]


def _get_runner_pb2_module():
    """Lazy load the runner_pb2 module to avoid circular imports."""
    from runner.v1 import runner_pb2
    return runner_pb2


def _get_model_profile_proto():
    """Lazy load the ModelProfile protobuf type."""
    return _get_runner_pb2_module().model__profile__pb2.ModelProfile


def _get_gpu_config_proto():
    """Lazy load the GPUConfig protobuf type."""
    return _get_runner_pb2_module().model__profile__pb2.gpu__config__pb2.GPUConfig


def timestamp_from_protobuf(timestamp: ProtobufTimestamp) -> datetime:
    """Convert protobuf Timestamp to Python datetime."""
    return datetime.fromtimestamp(timestamp.seconds)


def timestamp_to_protobuf(dt: datetime) -> ProtobufTimestamp:
    """Convert Python datetime to protobuf Timestamp."""
    return ProtobufTimestamp(seconds=int(dt.timestamp()))


def _provider_from_protobuf(provider_str: str) -> Optional[ModelProvider]:
    """Convert protobuf provider string to ModelProvider enum."""
    provider_map = {
        "llama_cpp": ModelProvider.LLAMA_CPP,
        "hf": ModelProvider.HF,
        "hugging_face": ModelProvider.HUGGING_FACE,
        "openai": ModelProvider.OPENAI,
        "stable_diffusion_cpp": ModelProvider.STABLE_DIFFUSION_CPP,
        "anthropic": ModelProvider.ANTHROPIC,
        "other": ModelProvider.OTHER,
    }
    return provider_map.get(provider_str, ModelProvider.OTHER)


def _provider_to_protobuf(provider: ModelProvider) -> str:
    """Convert ModelProvider enum to protobuf string."""
    return provider.value


def _task_from_protobuf(task_str: str) -> Optional[ModelTask]:
    """Convert protobuf task string to ModelTask enum."""
    task_map = {
        "TextToText": ModelTask.TEXTTOTEXT,
        "TextToImage": ModelTask.TEXTTOIMAGE,
        "ImageToText": ModelTask.IMAGETOTEXT,
        "ImageToImage": ModelTask.IMAGETOIMAGE,
        "TextToAudio": ModelTask.TEXTTOAUDIO,
        "AudioToText": ModelTask.AUDIOTOTEXT,
        "TextToVideo": ModelTask.TEXTTOVIDEO,
        "VideoToText": ModelTask.VIDEOTOTEXT,
        "TextToSpeech": ModelTask.TEXTTOSPEECH,
        "SpeechToText": ModelTask.SPEECHTOTEXT,
        "TextToEmbeddings": ModelTask.TEXTTOEMBEDDINGS,
        "VisionTextToText": ModelTask.VISIONTEXTTOTEXT,
        "ImageTextToImage": ModelTask.IMAGETEXTTOIMAGE,
        "TextToRanking": ModelTask.TEXTTORANKING,
    }
    return task_map.get(task_str, ModelTask.TEXTTOTEXT)


def _task_to_protobuf(task: ModelTask) -> str:
    """Convert ModelTask enum to protobuf string."""
    return task.value


def _tuning_strategy_from_protobuf(
    strategy_str: str,
) -> Optional[ParameterTuningStrategy]:
    """Convert protobuf strategy string to ParameterTuningStrategy enum."""
    strategy_map = {
        "binary_search": ParameterTuningStrategy.BINARY_SEARCH,
        "exponential_backoff": ParameterTuningStrategy.EXPONENTIAL_BACKOFF,
        "conservative_increment": ParameterTuningStrategy.CONSERVATIVE_INCREMENT,
    }
    return strategy_map.get(strategy_str, ParameterTuningStrategy.BINARY_SEARCH)


def _tuning_strategy_to_protobuf(strategy: ParameterTuningStrategy) -> str:
    """Convert ParameterTuningStrategy enum to protobuf string."""
    return strategy.value


def _reasoning_effort_from_protobuf(value: int) -> Optional[str]:
    """Convert protobuf ReasoningEffortEnum value to ModelParameters string.

    Protobuf enum mapping:
        0 = REASONINGEFFORTENUM_UNSPECIFIED -> None (use default)
        1 = REASONINGEFFORTENUM_LOW -> "low"
        2 = REASONINGEFFORTENUM_MEDIUM -> "medium"
        3 = REASONINGEFFORTENUM_HIGH -> "high"
    """
    if value == 0:
        return None
    mapping = {
        1: "low",
        2: "medium",
        3: "high",
    }
    return mapping.get(value)


def model_parameters_from_protobuf(parameters_proto) -> ModelParameters:
    """Convert protobuf ModelParameters to ModelParameters model.

    Args:
        parameters_proto: ModelParameters protobuf message or dict.

    Returns:
        ModelParameters model.
    """
    # If dict, use existing conversion
    if isinstance(parameters_proto, dict):
        # Map common protobuf config keys to ModelParameters fields
        param_map = {
            "num_ctx": "num_ctx",
            "n_ctx": "num_ctx",
            "context_size": "num_ctx",
            "context_length": "num_ctx",
            "repeat_last_n": "repeat_last_n",
            "repeat_penalty": "repeat_penalty",
            "repetition_penalty": "repeat_penalty",
            "temperature": "temperature",
            "seed": "seed",
            "stop_sequences": "stop",
            "stop": "stop",
            "max_tokens": "num_predict",
            "max_completion_tokens": "num_predict",
            "num_predict": "num_predict",
            "top_k": "top_k",
            "top_p": "top_p",
            "min_p": "min_p",
            "n_predict": "num_predict",
            "batch_size": "batch_size",
            "micro_batch_size": "micro_batch_size",
            "n_gpu_layers": "n_gpu_layers",
            "gpu_layers": "gpu_layers",
            "n_cpu_moe": "n_cpu_moe",
            "kv_on_cpu": "kv_on_cpu",
            "kv_offload": "kv_on_cpu",
            "offload_kqv": "offload_kqv",
            "flash_attention": "flash_attention",
            "flash_attn": "flash_attention",
            "reasoning_effort": "reasoning_effort",
        }

        params = {}
        for key, value in parameters_proto.items():
            normalized_key = param_map.get(key, key)
            # Handle reasoning_effort enum conversion from int to string
            if normalized_key == "reasoning_effort" and isinstance(value, int):
                converted = _reasoning_effort_from_protobuf(value)
                if converted is not None:
                    params[normalized_key] = converted
            else:
                params[normalized_key] = value

        return ModelParameters(**params)

    # If protobuf message, extract fields
    params = {}
    if hasattr(parameters_proto, "num_ctx"):
        params["num_ctx"] = parameters_proto.num_ctx
    if hasattr(parameters_proto, "repeat_last_n"):
        params["repeat_last_n"] = parameters_proto.repeat_last_n
    if hasattr(parameters_proto, "repeat_penalty"):
        params["repeat_penalty"] = parameters_proto.repeat_penalty
    if hasattr(parameters_proto, "temperature"):
        params["temperature"] = parameters_proto.temperature
    if hasattr(parameters_proto, "seed"):
        params["seed"] = parameters_proto.seed
    if hasattr(parameters_proto, "stop"):
        params["stop"] = parameters_proto.stop
    if hasattr(parameters_proto, "num_predict"):
        params["num_predict"] = parameters_proto.num_predict
    if hasattr(parameters_proto, "top_k"):
        params["top_k"] = parameters_proto.top_k
    if hasattr(parameters_proto, "top_p"):
        params["top_p"] = parameters_proto.top_p
    if hasattr(parameters_proto, "min_p"):
        params["min_p"] = parameters_proto.min_p
    if hasattr(parameters_proto, "batch_size"):
        params["batch_size"] = parameters_proto.batch_size
    if hasattr(parameters_proto, "micro_batch_size"):
        params["micro_batch_size"] = parameters_proto.micro_batch_size
    if hasattr(parameters_proto, "n_gpu_layers"):
        params["n_gpu_layers"] = parameters_proto.n_gpu_layers
    if hasattr(parameters_proto, "n_cpu_moe"):
        params["n_cpu_moe"] = parameters_proto.n_cpu_moe
    if hasattr(parameters_proto, "kv_on_cpu"):
        params["kv_on_cpu"] = parameters_proto.kv_on_cpu
    if hasattr(parameters_proto, "flash_attention"):
        params["flash_attention"] = parameters_proto.flash_attention
    if hasattr(parameters_proto, "reasoning_effort"):
        value = parameters_proto.reasoning_effort
        converted = _reasoning_effort_from_protobuf(value)
        if converted is not None:
            params["reasoning_effort"] = converted

    return ModelParameters(**params)


def model_parameters_to_protobuf(params: ModelParameters) -> dict:
    """Convert ModelParameters to protobuf-compatible dict."""
    config = {}

    # Map ModelParameters fields to common protobuf config keys
    field_map = {
        "num_ctx": ["n_ctx", "num_ctx", "context_size", "context_length"],
        "repeat_last_n": ["repeat_last_n"],
        "repeat_penalty": ["repeat_penalty", "repetition_penalty"],
        "temperature": ["temperature"],
        "seed": ["seed"],
        "stop": ["stop", "stop_sequences"],
        "num_predict": [
            "max_tokens",
            "max_completion_tokens",
            "n_predict",
            "num_predict",
        ],
        "top_k": ["top_k"],
        "top_p": ["top_p"],
        "min_p": ["min_p"],
        "batch_size": ["batch_size"],
        "micro_batch_size": ["micro_batch_size"],
        "n_gpu_layers": ["n_gpu_layers", "gpu_layers"],
        "n_cpu_moe": ["n_cpu_moe"],
        "kv_on_cpu": ["kv_on_cpu", "kv_offload"],
        "offload_kqv": ["offload_kqv"],
        "flash_attention": ["flash_attention", "flash_attn"],
        "reasoning_effort": ["reasoning_effort"],
    }

    for field_name, proto_keys in field_map.items():
        if hasattr(params, field_name):
            value = getattr(params, field_name)
            # Use first key as primary, add all others
            if value is not None:
                config[proto_keys[0]] = value

    return config


def gpu_config_from_protobuf(gpu_config_proto) -> Optional[GPUConfig]:
    """Convert protobuf GPUConfig to GPUConfig model.

    Args:
        gpu_config_proto: GPUConfig protobuf message.

    Returns:
        GPUConfig model or None if input is None.
    """
    if gpu_config_proto is None:
        return None

    # Lazy import to avoid circular dependency
    GPUConfigProto = _get_gpu_config_proto()

    # Map protobuf fields to GPUConfig
    gpu_layers = gpu_config_proto.gpu_layers
    main_gpu = gpu_config_proto.main_gpu
    main_gpu_device_id = getattr(gpu_config_proto, "main_gpu_device_id", None)
    gpu_memory_fraction = getattr(gpu_config_proto, "gpu_memory_fraction", None)
    flash_attn = getattr(gpu_config_proto, "flash_attn", None)

    # Convert main_gpu from string (protobuf) to int if needed
    if isinstance(main_gpu, str):
        try:
            main_gpu = int(main_gpu)
        except ValueError:
            main_gpu = -1

    # Map flash_attn to offload_kqv (inverted logic)
    offload_kqv = True
    if flash_attn is False:
        offload_kqv = False

    return GPUConfig(
        gpu_layers=gpu_layers,
        main_gpu=main_gpu,
        main_gpu_device_id=main_gpu_device_id,
        offload_kqv=offload_kqv,
    )


def gpu_config_to_protobuf(gpu_config: GPUConfig) -> dict:
    """Convert GPUConfig to protobuf-compatible dict.

    Args:
        gpu_config: GPUConfig model.

    Returns:
        Dict compatible with protobuf GPUConfig.
    """
    result = {}

    if gpu_config.gpu_layers is not None:
        result["gpu_layers"] = gpu_config.gpu_layers

    if gpu_config.main_gpu is not None:
        result["main_gpu"] = (
            str(gpu_config.main_gpu)
            if isinstance(gpu_config.main_gpu, int)
            else gpu_config.main_gpu
        )

    if gpu_config.main_gpu_device_id is not None:
        result["main_gpu_device_id"] = gpu_config.main_gpu_device_id

    # Map offload_kqv to flash_attn (inverted logic)
    if hasattr(gpu_config, "offload_kqv") and gpu_config.offload_kqv is not None:
        result["flash_attn"] = gpu_config.offload_kqv

    # Map tensor_split fields if present
    if hasattr(gpu_config, "tensor_split") and gpu_config.tensor_split is not None:
        result["tensor_split"] = gpu_config.tensor_split

    if (
        hasattr(gpu_config, "tensor_split_devices")
        and gpu_config.tensor_split_devices is not None
    ):
        result["tensor_split_devices"] = gpu_config.tensor_split_devices

    if hasattr(gpu_config, "split_mode") and gpu_config.split_mode is not None:
        result["split_mode"] = gpu_config.split_mode

    return result


def crash_prevention_from_protobuf(config: Optional[dict] = None) -> CrashPrevention:
    """Create CrashPrevention from optional protobuf config dict."""
    defaults = {
        "enable_preallocation_test": True,
        "memory_buffer_mb": 1024,
        "timeout_seconds": 120,
        "enable_graceful_degradation": True,
    }

    if config:
        defaults.update(config)

    return CrashPrevention(**defaults)


def crash_prevention_to_protobuf(crash_prevention: CrashPrevention) -> dict:
    """Convert CrashPrevention to protobuf-compatible dict."""
    return {
        "enable_preallocation_test": crash_prevention.enable_preallocation_test,
        "memory_buffer_mb": crash_prevention.memory_buffer_mb,
        "timeout_seconds": crash_prevention.timeout_seconds,
        "enable_graceful_degradation": crash_prevention.enable_graceful_degradation,
    }


def performance_parameter_from_protobuf(proto_params: list) -> list:
    """Convert protobuf performance parameters to PerformanceParameter models."""
    params = []
    for proto_param in proto_params:
        if isinstance(proto_param, dict):
            proto_param = type("PerfParamProto", (), proto_param)()

        parameter_name = getattr(proto_param, "parameter_name", "n_ctx")
        priority = getattr(proto_param, "priority", 10)
        strategy_str = getattr(proto_param, "tuning_strategy", "binary_search")
        max_search_attempts = getattr(proto_param, "max_search_attempts", 10)
        floor = getattr(proto_param, "floor", 1)
        operator = getattr(proto_param, "operator", "+")
        modifier = getattr(proto_param, "modifier", 1)
        max_value = getattr(proto_param, "max_value", 100)
        tuning_strategy = _tuning_strategy_from_protobuf(strategy_str)
        assert tuning_strategy is not None, f"Invalid tuning strategy: {strategy_str}"

        params.append(
            PerformanceParameter(
                parameter_name=parameter_name,  # type: ignore
                priority=priority,
                tuning_strategy=tuning_strategy,
                max_search_attempts=max_search_attempts,
                floor=floor,
                operator=operator,  # type: ignore
                modifier=modifier,
                max_value=max_value,
            )
        )
    return params


def performance_parameter_to_protobuf(params: list) -> list:
    """Convert PerformanceParameter models to protobuf-compatible list."""
    result = []
    for param in params:
        result.append(
            {
                "parameter_name": param.parameter_name,
                "priority": param.priority,
                "tuning_strategy": _tuning_strategy_to_protobuf(param.tuning_strategy),
                "max_search_attempts": param.max_search_attempts,
                "floor": param.floor,
                "operator": param.operator,
                "modifier": param.modifier,
                "max_value": param.max_value,
            }
        )
    return result


def parameter_optimization_config_from_protobuf(
    proto_config,
) -> Optional[ParameterOptimizationConfig]:
    """Convert protobuf ParameterOptimizationConfig to model.

    Args:
        proto_config: ParameterOptimizationConfig protobuf message or None.

    Returns:
        ParameterOptimizationConfig model or None if input is None.
    """
    if proto_config is None:
        return None

    # Lazy import to avoid circular dependency
    ParameterOptimizationConfigProto = _get_runner_pb2_module().ParameterOptimizationConfig

    # ParameterOptimizationConfig has batch_size, context_size, threads, mlock, mmap
    # enabled is inferred from batch_size > 0
    enabled = getattr(proto_config, "batch_size", 0) > 0
    proto_params = []
    crash_config = {
        "enable_preallocation_test": getattr(proto_config, "mlock", False),
        "memory_buffer_mb": 1024,
        "timeout_seconds": getattr(proto_config, "context_size", 120),
        "enable_graceful_degradation": getattr(proto_config, "mmap", False),
    }

    if not enabled and not proto_params:
        return None

    return ParameterOptimizationConfig(
        enabled=enabled,
        parameters=performance_parameter_from_protobuf(proto_params),
        crash_prevention=crash_prevention_from_protobuf(crash_config),
    )


def parameter_optimization_config_to_protobuf(
    config: ParameterOptimizationConfig,
) -> dict:
    """Convert ParameterOptimizationConfig to protobuf-compatible dict."""
    return {
        "enabled": config.enabled,
        "parameters": performance_parameter_to_protobuf(config.parameters),
        "crash_prevention": crash_prevention_to_protobuf(config.crash_prevention),
    }


def model_profile_from_protobuf(
    profile_proto,
    priority: Optional[str] = None,
    grammar_type: Optional[str] = None,
) -> ModelProfile:
    """Convert protobuf ModelProfile to ModelProfile model.

    Args:
        profile_proto: ModelProfile protobuf message.
        priority: Optional pipeline priority string.
        grammar_type: Optional grammar type string.

    Returns:
        ModelProfile model.
    """
    # Lazy import to avoid circular dependency
    ModelProfileProto = _get_model_profile_proto()

    model_name = profile_proto.model_name
    # ModelProfile has 'parameters' field, not 'model_config'
    parameters_proto = profile_proto.parameters
    gpu_config_proto = profile_proto.gpu_config
    # Use parameter_optimization (not optimization_config)
    opt_config_proto = profile_proto.parameter_optimization

    # Convert parameters from protobuf ModelParameters to Pydantic ModelParameters
    parameters = model_parameters_from_protobuf(parameters_proto)

    return ModelProfile(
        user_id="default_user",  # Not in protobuf - set default
        name=model_name,
        model_name=model_name,
        parameters=parameters,
        system_prompt="",  # Not in protobuf - set default
        type=0,  # Not in protobuf - set default
        gpu_config=gpu_config_from_protobuf(gpu_config_proto),
        parameter_optimization=parameter_optimization_config_from_protobuf(
            opt_config_proto
        ),
    )


def model_profile_to_protobuf(model_profile: ModelProfile) -> dict:
    """Convert ModelProfile to protobuf-compatible dict.

    Args:
        model_profile: ModelProfile model.

    Returns:
        Dict compatible with protobuf ModelProfile.
    """
    result = {
        "model_name": model_profile.model_name,
        "model_config": model_parameters_to_protobuf(model_profile.parameters),
    }

    if model_profile.gpu_config:
        result["gpu_config"] = gpu_config_to_protobuf(model_profile.gpu_config)

    if model_profile.parameter_optimization:
        result["optimization_config"] = parameter_optimization_config_to_protobuf(
            model_profile.parameter_optimization
        )

    return result


def token_chunk_from_protobuf(chunk_proto) -> MessageContent:
    """Convert protobuf TokenChunk to MessageContent.

    Args:
        chunk_proto: TokenChunk protobuf message.

    Returns:
        MessageContent model with text content.
    """
    # Lazy import to avoid circular dependency
    TokenChunkProto = _get_runner_pb2_module().TokenChunk

    token = chunk_proto.token
    token_id = chunk_proto.token_id
    probability = chunk_proto.probability
    metadata = dict(chunk_proto.metadata)

    # Build text content from token
    text = token

    return MessageContent(
        type=MessageContentType.TEXT,
        text=text,
    )


def pipeline_complete_from_protobuf(
    complete_proto,
) -> dict:
    """Process protobuf PipelineComplete and extract output data.

    Args:
        complete_proto: PipelineComplete protobuf message.

    Returns:
        Dict with output_data and duration_ms.
    """
    # Lazy import to avoid circular dependency
    PipelineCompleteProto = _get_runner_pb2_module().PipelineComplete

    output_data = complete_proto.output_data
    duration_ms = complete_proto.duration_ms

    # Decode bytes to string if possible
    try:
        output_str = output_data.decode("utf-8") if output_data else ""
    except UnicodeDecodeError:
        output_str = output_data.decode("latin-1") if output_data else ""

    return {
        "output": output_str,
        "duration_ms": duration_ms,
    }


def pipeline_error_from_protobuf(error_proto) -> dict:
    """Convert protobuf PipelineError to error dict.

    Args:
        error_proto: PipelineError protobuf message.

    Returns:
        Dict with error message and code.
    """
    # Lazy import to avoid circular dependency
    PipelineErrorProto = _get_runner_pb2_module().PipelineError

    message = error_proto.message
    error_code = error_proto.error_code

    return {
        "message": message,
        "error_code": error_code,
    }


def pipeline_event_from_protobuf(event_proto) -> ChatResponse:
    """Convert protobuf PipelineEvent to ChatResponse.

    Args:
        event_proto: PipelineEvent protobuf message.

    Returns:
        ChatResponse model with token chunk or completion info.
    """
    # Lazy imports to avoid circular dependency
    PipelineEventProto = _get_runner_pb2_module().PipelineEvent
    PipelineCompleteProto = _get_runner_pb2_module().PipelineComplete
    PipelineErrorProto = _get_runner_pb2_module().PipelineError
    TokenChunkProto = _get_runner_pb2_module().TokenChunk

    token_chunk_proto = event_proto.token_chunk
    complete_proto = event_proto.complete
    error_proto = event_proto.error

    # Build message content from token chunk
    contents = []
    if token_chunk_proto:
        content = token_chunk_from_protobuf(token_chunk_proto)
        contents.append(content)

    # Check for completion
    finish_reason = None
    if complete_proto:
        complete_data = pipeline_complete_from_protobuf(complete_proto)
        if "output" in complete_data:
            contents.append(
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=complete_data["output"],
                )
            )
        finish_reason = "stop"

    # Check for error
    if error_proto:
        error_data = pipeline_error_from_protobuf(error_proto)
        finish_reason = "error"

    return ChatResponse(
        done=finish_reason is not None and finish_reason != "error",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=contents,
        ),
        finish_reason=finish_reason,
    )


def cache_stats_from_protobuf(stats_proto) -> dict:
    """Convert protobuf CacheStats to a dict (CacheStats model doesn't exist).

    Args:
        stats_proto: CacheStats protobuf message.

    Returns:
        Dict with cache statistics.
    """
    # Lazy import to avoid circular dependency
    CacheStatsProto = _get_runner_pb2_module().CacheStats

    return {
        "total_pipelines": stats_proto.total_pipelines,
        "cached_pipelines": stats_proto.cached_pipelines,
        "active_pipelines": stats_proto.active_pipelines,
        "total_memory_bytes": stats_proto.total_memory_bytes,
        "available_memory_bytes": stats_proto.available_memory_bytes,
        "cache_hits": stats_proto.cache_hits,
        "cache_misses": stats_proto.cache_misses,
        "hit_rate": stats_proto.hit_rate,
    }


def cache_stats_to_protobuf(stats: dict) -> dict:
    """Convert cache stats dict to protobuf-compatible dict.

    Args:
        stats: Dict with cache statistics.

    Returns:
        Dict compatible with protobuf CacheStats.
    """
    return {
        "total_pipelines": stats.get("total_pipelines", 0),
        "cached_pipelines": stats.get("cached_pipelines", 0),
        "active_pipelines": stats.get("active_pipelines", 0),
        "total_memory_bytes": stats.get("total_memory_bytes", 0),
        "available_memory_bytes": stats.get("available_memory_bytes", 0),
        "cache_hits": stats.get("cache_hits", 0),
        "cache_misses": stats.get("cache_misses", 0),
        "hit_rate": stats.get("hit_rate", 0.0),
    }


def model_info_from_protobuf(info_proto) -> dict:
    """Convert protobuf ModelInfo to a dict.

    Args:
        info_proto: ModelInfo protobuf message.

    Returns:
        Dict with model info.
    """
    # Lazy import to avoid circular dependency
    ModelInfoProto = _get_runner_pb2_module().ModelInfo
    ProtobufTimestamp = _get_runner_pb2_module().common_dot_timestamp__pb2.Timestamp

    model_name = info_proto.model_name
    provider_str = info_proto.provider
    task_type_str = info_proto.task_type
    is_loaded = info_proto.is_loaded
    memory_bytes = info_proto.memory_bytes
    loaded_at_proto = info_proto.loaded_at

    loaded_at = None
    if loaded_at_proto:
        loaded_at = timestamp_from_protobuf(loaded_at_proto)

    return {
        "model_name": model_name,
        "provider": _provider_from_protobuf(provider_str),
        "task_type": _task_from_protobuf(task_type_str),
        "is_loaded": is_loaded,
        "memory_bytes": memory_bytes,
        "loaded_at": loaded_at,
    }


def model_info_to_protobuf(info: dict) -> dict:
    """Convert model info dict to protobuf-compatible dict.

    Args:
        info: Dict with model info.

    Returns:
        Dict compatible with protobuf ModelInfo.
    """
    result = {
        "model_name": info.get("model_name", ""),
        "provider": _provider_to_protobuf(info.get("provider", "")),
        "task_type": _task_to_protobuf(info.get("task_type", "")),
        "is_loaded": info.get("is_loaded", False),
        "memory_bytes": info.get("memory_bytes", 0),
    }

    if info.get("loaded_at"):
        result["loaded_at"] = timestamp_to_protobuf(info["loaded_at"])

    return result


# Convenience functions for common conversion patterns


def create_pipeline_request_from_protobuf(
    request_proto,
) -> dict:
    """Convert CreatePipelineRequest to a dict for pipeline factory.

    Args:
        request_proto: CreatePipelineRequest protobuf message.

    Returns:
        Dict with profile, priority, and grammar_type.
    """
    # Lazy imports to avoid circular dependency
    CreatePipelineRequestProto = _get_runner_pb2_module().CreatePipelineRequest

    profile_proto = request_proto.profile
    priority = request_proto.priority
    grammar_type = request_proto.grammar_type

    return {
        "profile": (
            model_profile_from_protobuf(profile_proto) if profile_proto else None
        ),
        "priority": priority,
        "grammar_type": grammar_type,
    }


def execute_pipeline_request_from_protobuf(
    request_proto,
) -> dict:
    """Convert ExecutePipelineRequest to a dict for pipeline execution.

    Args:
        request_proto: ExecutePipelineRequest protobuf message.

    Returns:
        Dict with pipeline_id, input_data, and stream_output.
    """
    # Lazy import to avoid circular dependency
    ExecutePipelineRequestProto = _get_runner_pb2_module().ExecutePipelineRequest

    return {
        "pipeline_id": request_proto.pipeline_id,
        "input_data": request_proto.input_data,
        "stream_output": request_proto.stream_output,
    }


def generate_embeddings_request_from_protobuf(
    request_proto,
) -> dict:
    """Convert GenerateEmbeddingsRequest to a dict for embedding generation.

    Args:
        request_proto: GenerateEmbeddingsRequest protobuf message.

    Returns:
        Dict with texts, model_name, and dimension.
    """
    # Lazy import to avoid circular dependency
    GenerateEmbeddingsRequestProto = _get_runner_pb2_module().GenerateEmbeddingsRequest

    texts = list(request_proto.texts)
    model_name = request_proto.model_name
    dimension = request_proto.dimension

    return {
        "texts": texts,
        "model_name": model_name,
        "dimension": dimension,
    }


def evict_pipeline_request_from_protobuf(
    request_proto,
) -> dict:
    """Convert EvictPipelineRequest to a dict for pipeline eviction.

    Args:
        request_proto: EvictPipelineRequest protobuf message.

    Returns:
        Dict with pipeline_id and force flag.
    """
    # Lazy import to avoid circular dependency
    EvictPipelineRequestProto = _get_runner_pb2_module().EvictPipelineRequest

    return {
        "pipeline_id": request_proto.pipeline_id,
        "force": request_proto.force,
    }
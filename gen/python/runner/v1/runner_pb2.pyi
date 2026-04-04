from common import timestamp_pb2 as _timestamp_pb2
from common import version_pb2 as _version_pb2
import model_profile_pb2 as _model_profile_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PipelineHandle(_message.Message):
    __slots__ = ("pipeline_id", "model_name", "is_cached")
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    IS_CACHED_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    model_name: str
    is_cached: bool
    def __init__(self, pipeline_id: _Optional[str] = ..., model_name: _Optional[str] = ..., is_cached: bool = ...) -> None: ...

class ParameterOptimizationConfig(_message.Message):
    __slots__ = ("batch_size", "context_size", "threads", "mlock", "mmap")
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_SIZE_FIELD_NUMBER: _ClassVar[int]
    THREADS_FIELD_NUMBER: _ClassVar[int]
    MLOCK_FIELD_NUMBER: _ClassVar[int]
    MMAP_FIELD_NUMBER: _ClassVar[int]
    batch_size: int
    context_size: int
    threads: int
    mlock: bool
    mmap: bool
    def __init__(self, batch_size: _Optional[int] = ..., context_size: _Optional[int] = ..., threads: _Optional[int] = ..., mlock: bool = ..., mmap: bool = ...) -> None: ...

class TokenChunk(_message.Message):
    __slots__ = ("token", "token_id", "probability", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    token: str
    token_id: int
    probability: float
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, token: _Optional[str] = ..., token_id: _Optional[int] = ..., probability: _Optional[float] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class PipelineEvent(_message.Message):
    __slots__ = ("token_chunk", "complete", "error")
    TOKEN_CHUNK_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    token_chunk: TokenChunk
    complete: PipelineComplete
    error: PipelineError
    def __init__(self, token_chunk: _Optional[_Union[TokenChunk, _Mapping]] = ..., complete: _Optional[_Union[PipelineComplete, _Mapping]] = ..., error: _Optional[_Union[PipelineError, _Mapping]] = ...) -> None: ...

class PipelineComplete(_message.Message):
    __slots__ = ("output_data", "duration_ms")
    OUTPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    output_data: bytes
    duration_ms: int
    def __init__(self, output_data: _Optional[bytes] = ..., duration_ms: _Optional[int] = ...) -> None: ...

class PipelineError(_message.Message):
    __slots__ = ("message", "error_code")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    error_code: str
    def __init__(self, message: _Optional[str] = ..., error_code: _Optional[str] = ...) -> None: ...

class ErrorMetadata(_message.Message):
    __slots__ = ("error_type", "resource_type", "resource_id", "details")
    class DetailsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    error_type: str
    resource_type: str
    resource_id: str
    details: _containers.ScalarMap[str, str]
    def __init__(self, error_type: _Optional[str] = ..., resource_type: _Optional[str] = ..., resource_id: _Optional[str] = ..., details: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ErrorDetails(_message.Message):
    __slots__ = ("metadata", "suggested_actions", "stack_trace", "timestamp_ms")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    SUGGESTED_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    STACK_TRACE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    metadata: ErrorMetadata
    suggested_actions: _containers.RepeatedScalarFieldContainer[str]
    stack_trace: str
    timestamp_ms: int
    def __init__(self, metadata: _Optional[_Union[ErrorMetadata, _Mapping]] = ..., suggested_actions: _Optional[_Iterable[str]] = ..., stack_trace: _Optional[str] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("success", "message", "error_code", "error_details")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_DETAILS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    error_code: str
    error_details: ErrorDetails
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., error_code: _Optional[str] = ..., error_details: _Optional[_Union[ErrorDetails, _Mapping]] = ...) -> None: ...

class GenerateEmbeddingsRequest(_message.Message):
    __slots__ = ("texts", "model_name", "dimension")
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    texts: _containers.RepeatedScalarFieldContainer[str]
    model_name: str
    dimension: int
    def __init__(self, texts: _Optional[_Iterable[str]] = ..., model_name: _Optional[str] = ..., dimension: _Optional[int] = ...) -> None: ...

class Embedding(_message.Message):
    __slots__ = ("values", "index")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    index: int
    def __init__(self, values: _Optional[_Iterable[float]] = ..., index: _Optional[int] = ...) -> None: ...

class GenerateEmbeddingsResponse(_message.Message):
    __slots__ = ("embeddings", "model_dimension")
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    MODEL_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    embeddings: _containers.RepeatedCompositeFieldContainer[Embedding]
    model_dimension: int
    def __init__(self, embeddings: _Optional[_Iterable[_Union[Embedding, _Mapping]]] = ..., model_dimension: _Optional[int] = ...) -> None: ...

class CacheStats(_message.Message):
    __slots__ = ("total_pipelines", "cached_pipelines", "active_pipelines", "total_memory_bytes", "available_memory_bytes", "cache_hits", "cache_misses", "hit_rate")
    TOTAL_PIPELINES_FIELD_NUMBER: _ClassVar[int]
    CACHED_PIPELINES_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_PIPELINES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    CACHE_HITS_FIELD_NUMBER: _ClassVar[int]
    CACHE_MISSES_FIELD_NUMBER: _ClassVar[int]
    HIT_RATE_FIELD_NUMBER: _ClassVar[int]
    total_pipelines: int
    cached_pipelines: int
    active_pipelines: int
    total_memory_bytes: int
    available_memory_bytes: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    def __init__(self, total_pipelines: _Optional[int] = ..., cached_pipelines: _Optional[int] = ..., active_pipelines: _Optional[int] = ..., total_memory_bytes: _Optional[int] = ..., available_memory_bytes: _Optional[int] = ..., cache_hits: _Optional[int] = ..., cache_misses: _Optional[int] = ..., hit_rate: _Optional[float] = ...) -> None: ...

class CreatePipelineRequest(_message.Message):
    __slots__ = ("profile", "priority", "grammar_type", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    GRAMMAR_TYPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    profile: _model_profile_pb2.ModelProfile
    priority: str
    grammar_type: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, profile: _Optional[_Union[_model_profile_pb2.ModelProfile, _Mapping]] = ..., priority: _Optional[str] = ..., grammar_type: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ExecutePipelineRequest(_message.Message):
    __slots__ = ("pipeline_id", "input_data", "stream_output")
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    STREAM_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    input_data: bytes
    stream_output: bool
    def __init__(self, pipeline_id: _Optional[str] = ..., input_data: _Optional[bytes] = ..., stream_output: bool = ...) -> None: ...

class GetCacheStatsRequest(_message.Message):
    __slots__ = ("pipeline_type",)
    PIPELINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    pipeline_type: str
    def __init__(self, pipeline_type: _Optional[str] = ...) -> None: ...

class EvictPipelineRequest(_message.Message):
    __slots__ = ("pipeline_id", "force")
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    force: bool
    def __init__(self, pipeline_id: _Optional[str] = ..., force: bool = ...) -> None: ...

class EvictPipelineResponse(_message.Message):
    __slots__ = ("success", "message", "freed_memory_bytes")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    FREED_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    freed_memory_bytes: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., freed_memory_bytes: _Optional[int] = ...) -> None: ...

class PipelineInfo(_message.Message):
    __slots__ = ("pipeline_id", "model_name", "pipeline_type", "is_cached", "created_at", "last_accessed", "memory_bytes")
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    IS_CACHED_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_ACCESSED_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    model_name: str
    pipeline_type: str
    is_cached: bool
    created_at: int
    last_accessed: int
    memory_bytes: int
    def __init__(self, pipeline_id: _Optional[str] = ..., model_name: _Optional[str] = ..., pipeline_type: _Optional[str] = ..., is_cached: bool = ..., created_at: _Optional[int] = ..., last_accessed: _Optional[int] = ..., memory_bytes: _Optional[int] = ...) -> None: ...

class ListPipelinesRequest(_message.Message):
    __slots__ = ("pipeline_type", "limit")
    PIPELINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    pipeline_type: str
    limit: int
    def __init__(self, pipeline_type: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...

class ListPipelinesResponse(_message.Message):
    __slots__ = ("pipelines",)
    PIPELINES_FIELD_NUMBER: _ClassVar[int]
    pipelines: _containers.RepeatedCompositeFieldContainer[PipelineInfo]
    def __init__(self, pipelines: _Optional[_Iterable[_Union[PipelineInfo, _Mapping]]] = ...) -> None: ...

class GetPipelineInfoRequest(_message.Message):
    __slots__ = ("pipeline_id",)
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    def __init__(self, pipeline_id: _Optional[str] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

class LoadModelRequest(_message.Message):
    __slots__ = ("model_name", "profile")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    profile: _model_profile_pb2.ModelProfile
    def __init__(self, model_name: _Optional[str] = ..., profile: _Optional[_Union[_model_profile_pb2.ModelProfile, _Mapping]] = ...) -> None: ...

class UnloadModelRequest(_message.Message):
    __slots__ = ("model_name",)
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    def __init__(self, model_name: _Optional[str] = ...) -> None: ...

class GetModelInfoRequest(_message.Message):
    __slots__ = ("model_name",)
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    def __init__(self, model_name: _Optional[str] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("model_name", "provider", "task_type", "is_loaded", "memory_bytes", "loaded_at")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    TASK_TYPE_FIELD_NUMBER: _ClassVar[int]
    IS_LOADED_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    LOADED_AT_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    provider: str
    task_type: str
    is_loaded: bool
    memory_bytes: int
    loaded_at: _timestamp_pb2.Timestamp
    def __init__(self, model_name: _Optional[str] = ..., provider: _Optional[str] = ..., task_type: _Optional[str] = ..., is_loaded: bool = ..., memory_bytes: _Optional[int] = ..., loaded_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetModelListRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ModelList(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

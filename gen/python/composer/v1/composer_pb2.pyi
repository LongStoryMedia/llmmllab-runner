import datetime

from google.api import annotations_pb2 as _annotations_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from common import timestamp_pb2 as _timestamp_pb2_1
from common import version_pb2 as _version_pb2
from messages import message_pb2 as _message_pb2
from messages import user_config_pb2 as _user_config_pb2
from messages import dynamic_tool_pb2 as _dynamic_tool_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExecuteWorkflowRequest(_message.Message):
    __slots__ = ("user_id", "workflow_type", "model_name", "user_config", "tools", "messages")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    USER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    TOOLS_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    workflow_type: str
    model_name: str
    user_config: _user_config_pb2.UserConfig
    tools: _dynamic_tool_pb2.DynamicTool
    messages: _containers.RepeatedCompositeFieldContainer[_message_pb2.Message]
    def __init__(self, user_id: _Optional[str] = ..., workflow_type: _Optional[str] = ..., model_name: _Optional[str] = ..., user_config: _Optional[_Union[_user_config_pb2.UserConfig, _Mapping]] = ..., tools: _Optional[_Union[_dynamic_tool_pb2.DynamicTool, _Mapping]] = ..., messages: _Optional[_Iterable[_Union[_message_pb2.Message, _Mapping]]] = ...) -> None: ...

class WorkflowEvent(_message.Message):
    __slots__ = ("event_type", "timestamp", "step_start", "step_complete", "error", "output")
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STEP_START_FIELD_NUMBER: _ClassVar[int]
    STEP_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    event_type: str
    timestamp: _timestamp_pb2.Timestamp
    step_start: WorkflowStepStart
    step_complete: WorkflowStepComplete
    error: WorkflowError
    output: WorkflowOutput
    def __init__(self, event_type: _Optional[str] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., step_start: _Optional[_Union[WorkflowStepStart, _Mapping]] = ..., step_complete: _Optional[_Union[WorkflowStepComplete, _Mapping]] = ..., error: _Optional[_Union[WorkflowError, _Mapping]] = ..., output: _Optional[_Union[WorkflowOutput, _Mapping]] = ...) -> None: ...

class WorkflowStepStart(_message.Message):
    __slots__ = ("step_name", "inputs")
    class InputsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STEP_NAME_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    step_name: str
    inputs: _containers.ScalarMap[str, str]
    def __init__(self, step_name: _Optional[str] = ..., inputs: _Optional[_Mapping[str, str]] = ...) -> None: ...

class WorkflowStepComplete(_message.Message):
    __slots__ = ("step_name", "outputs", "duration_ms")
    class OutputsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STEP_NAME_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    step_name: str
    outputs: _containers.ScalarMap[str, str]
    duration_ms: int
    def __init__(self, step_name: _Optional[str] = ..., outputs: _Optional[_Mapping[str, str]] = ..., duration_ms: _Optional[int] = ...) -> None: ...

class WorkflowError(_message.Message):
    __slots__ = ("message", "error_code", "details")
    class DetailsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    message: str
    error_code: str
    details: _containers.ScalarMap[str, str]
    def __init__(self, message: _Optional[str] = ..., error_code: _Optional[str] = ..., details: _Optional[_Mapping[str, str]] = ...) -> None: ...

class WorkflowOutput(_message.Message):
    __slots__ = ("data", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DATA_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, data: _Optional[bytes] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ChatDelta(_message.Message):
    __slots__ = ("message_id", "role", "content", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    message_id: str
    role: str
    content: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, message_id: _Optional[str] = ..., role: _Optional[str] = ..., content: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ChatResponseDelta(_message.Message):
    __slots__ = ("delta", "created_at", "context", "finish_reason", "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "processing", "state", "prev_state")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DURATION_FIELD_NUMBER: _ClassVar[int]
    LOAD_DURATION_FIELD_NUMBER: _ClassVar[int]
    PROMPT_EVAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    PROMPT_EVAL_DURATION_FIELD_NUMBER: _ClassVar[int]
    EVAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    EVAL_DURATION_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    PREV_STATE_FIELD_NUMBER: _ClassVar[int]
    delta: ChatDelta
    created_at: _timestamp_pb2.Timestamp
    context: _containers.RepeatedScalarFieldContainer[float]
    finish_reason: str
    total_duration: float
    load_duration: float
    prompt_eval_count: float
    prompt_eval_duration: float
    eval_count: float
    eval_duration: float
    processing: str
    state: str
    prev_state: str
    def __init__(self, delta: _Optional[_Union[ChatDelta, _Mapping]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., context: _Optional[_Iterable[float]] = ..., finish_reason: _Optional[str] = ..., total_duration: _Optional[float] = ..., load_duration: _Optional[float] = ..., prompt_eval_count: _Optional[float] = ..., prompt_eval_duration: _Optional[float] = ..., eval_count: _Optional[float] = ..., eval_duration: _Optional[float] = ..., processing: _Optional[str] = ..., state: _Optional[str] = ..., prev_state: _Optional[str] = ...) -> None: ...

class ChatResponseComplete(_message.Message):
    __slots__ = ("complete", "created_at", "context", "finish_reason", "total_duration", "load_duration")
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DURATION_FIELD_NUMBER: _ClassVar[int]
    LOAD_DURATION_FIELD_NUMBER: _ClassVar[int]
    complete: WorkflowComplete
    created_at: _timestamp_pb2.Timestamp
    context: _containers.RepeatedScalarFieldContainer[float]
    finish_reason: str
    total_duration: float
    load_duration: float
    def __init__(self, complete: _Optional[_Union[WorkflowComplete, _Mapping]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., context: _Optional[_Iterable[float]] = ..., finish_reason: _Optional[str] = ..., total_duration: _Optional[float] = ..., load_duration: _Optional[float] = ...) -> None: ...

class ChatResponse(_message.Message):
    __slots__ = ("delta", "complete")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    delta: ChatResponseDelta
    complete: ChatResponseComplete
    def __init__(self, delta: _Optional[_Union[ChatResponseDelta, _Mapping]] = ..., complete: _Optional[_Union[ChatResponseComplete, _Mapping]] = ...) -> None: ...

class WorkflowComplete(_message.Message):
    __slots__ = ("output_data", "duration_ms")
    OUTPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    output_data: bytes
    duration_ms: int
    def __init__(self, output_data: _Optional[bytes] = ..., duration_ms: _Optional[int] = ...) -> None: ...

class TodoItem(_message.Message):
    __slots__ = ("id", "title", "completed", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    title: str
    completed: bool
    created_at: int
    def __init__(self, id: _Optional[str] = ..., title: _Optional[str] = ..., completed: bool = ..., created_at: _Optional[int] = ...) -> None: ...

class GetWorkflowStatusRequest(_message.Message):
    __slots__ = ("user_id", "workflow_type")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_TYPE_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    workflow_type: str
    def __init__(self, user_id: _Optional[str] = ..., workflow_type: _Optional[str] = ...) -> None: ...

class GetWorkflowStatusResponse(_message.Message):
    __slots__ = ("user_id", "workflow_type", "status", "created_at", "completed_at")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKFLOW_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    workflow_type: str
    status: str
    created_at: _timestamp_pb2.Timestamp
    completed_at: _timestamp_pb2.Timestamp
    def __init__(self, user_id: _Optional[str] = ..., workflow_type: _Optional[str] = ..., status: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., completed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ListWorkflowTypesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListWorkflowTypesResponse(_message.Message):
    __slots__ = ("workflow_types",)
    WORKFLOW_TYPES_FIELD_NUMBER: _ClassVar[int]
    workflow_types: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, workflow_types: _Optional[_Iterable[str]] = ...) -> None: ...

class GetWorkflowSchemaRequest(_message.Message):
    __slots__ = ("workflow_type",)
    WORKFLOW_TYPE_FIELD_NUMBER: _ClassVar[int]
    workflow_type: str
    def __init__(self, workflow_type: _Optional[str] = ...) -> None: ...

class WorkflowSchema(_message.Message):
    __slots__ = ("workflow_type", "json_schema", "description")
    WORKFLOW_TYPE_FIELD_NUMBER: _ClassVar[int]
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    workflow_type: str
    json_schema: str
    description: str
    def __init__(self, workflow_type: _Optional[str] = ..., json_schema: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

"""
OpenAI SDK adapter for llama.cpp integration.

Creates an OpenAI client connected to the local llama.cpp server and
exposes it as a BaseChatModel (langchain_core) for use with composer agents.
Replaces the previous langchain_openai.ChatOpenAI dependency.
"""

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel

from models import Model, ModelProfile, ModelProfileType
from pipelines.base import BasePipeline
from server_manager import LlamaCppServerManager
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ChatLlamaCppPipeline")


class ReasoningAwareAIMessageChunk(AIMessageChunk):
    """Extended AIMessageChunk that captures reasoning content."""

    def __init__(self, reasoning_content: str = "", **kwargs):
        super().__init__(**kwargs)
        self.reasoning_content = reasoning_content


class ChatLlamaCppPipeline(BasePipeline):
    """
    OpenAI SDK adapter connected to a local llama.cpp server.

    Implements BaseChatModel._generate and _stream by talking directly to
    the llama.cpp OpenAI-compatible /v1/chat/completions endpoint via the
    official ``openai`` Python SDK, removing the langchain_openai dependency.
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(model, profile, grammar, metadata)
        self.user_config = kwargs.get("user_config", None)
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )

        # Create server manager
        self.server_manager = LlamaCppServerManager(
            model=model,
            profile=profile,
            user_config=self.user_config,
        )

        self._openai_client: Optional[OpenAI] = None
        self.started = False
        self.metadata = metadata or {}

        # Tool state for disable_streaming="tool_calling" equivalent
        self._bound_tools: Optional[list] = None
        self._tool_choice: Optional[str] = None

        # Extra body params forwarded to llama.cpp (e.g. chat_template_kwargs)
        self._extra_body: dict = {
            "chat_template_kwargs": {"enable_thinking": False},
        }

        # Initialize server and client
        self._initialize_persistent_server()

    def _initialize_persistent_server(self):
        """Initialize llama.cpp server and create OpenAI client."""
        try:
            self._logger.info(f"Starting server for model {self.model.name}")
            assert self.server_manager is not None
            self.started = self.server_manager.start()
            if not self.started:
                raise RuntimeError(
                    f"Failed to start server for model {self.model.name}"
                )

            base_url = self.server_manager.get_api_endpoint("")

            self._openai_client = OpenAI(
                base_url=base_url,
                api_key="not-needed",
                timeout=float(self.server_manager.startup_timeout),
                max_retries=1,
            )

            self._logger.info(
                f"OpenAI SDK client initialized with base_url: {base_url}"
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize server and OpenAI client: {e}")
            raise

    # ------------------------------------------------------------------
    # Message conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: List[BaseMessage]) -> List[dict]:
        """Convert langchain BaseMessage list to OpenAI message dicts."""
        result: List[dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                d: Dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    d["content"] = msg.content
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("args", {})),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(d)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "content": (
                            msg.content
                            if isinstance(msg.content, str)
                            else json.dumps(msg.content)
                        ),
                        "tool_call_id": msg.tool_call_id,
                    }
                )
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _build_request_kwargs(self, **kwargs) -> dict:
        """Build common kwargs for the OpenAI completions call."""
        profile_max = self.profile.parameters.max_tokens
        max_tokens = profile_max if (profile_max and profile_max > 0) else None

        req: Dict[str, Any] = {
            "model": "local-model",
            "temperature": self.profile.parameters.temperature or 0.7,
            "top_p": self.profile.parameters.top_p or 0.9,
        }
        if max_tokens:
            req["max_tokens"] = max_tokens

        # Tools from bind_tools() via RunnableBinding kwargs, or stored directly
        tools = kwargs.pop("tools", None) or self._bound_tools
        if tools:
            req["tools"] = tools
            if self._tool_choice:
                req["tool_choice"] = self._tool_choice

        if self._extra_body:
            req["extra_body"] = self._extra_body

        stop = kwargs.pop("stop", None)
        if stop:
            req["stop"] = stop

        return req

    @staticmethod
    def _parse_tool_calls(message) -> list:
        """Extract tool calls from an OpenAI response message."""
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []
        return [
            {
                "name": tc.function.name,
                "args": (
                    json.loads(tc.function.arguments)
                    if tc.function.arguments
                    else {}
                ),
                "id": tc.id,
                "type": "tool_call",
            }
            for tc in message.tool_calls
        ]

    # ------------------------------------------------------------------
    # BaseChatModel interface
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        """Generate chat completions (non-streaming)."""
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized")

        self._logger.debug(
            f"Generating with {len(messages)} messages"
        )

        oai_messages = self._convert_messages(messages)
        req = self._build_request_kwargs(stop=stop, **kwargs)

        response = self._openai_client.chat.completions.create(
            messages=oai_messages,
            stream=False,
            **req,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = self._parse_tool_calls(choice.message)

        ai_message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            response_metadata={
                "finish_reason": choice.finish_reason,
                "model": response.model,
            },
        )

        generation = ChatGeneration(message=ai_message)
        return ChatResult(
            generations=[generation],
            llm_output={
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else {},
            },
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions with reasoning_content support.

        When tools are bound, falls back to _generate to avoid streaming
        tool-call diff errors from llama.cpp (equivalent to the old
        ``disable_streaming="tool_calling"`` on ChatOpenAI).
        """
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized")

        # Fallback to non-streaming when tools are present to avoid
        # "Invalid diff: now finding less tool calls!" errors.
        tools = kwargs.get("tools") or self._bound_tools
        if tools:
            result = self._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            for gen in result.generations:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=gen.message.content or "",
                        tool_calls=getattr(gen.message, "tool_calls", []),
                        response_metadata=gen.message.response_metadata,
                    ),
                    generation_info=gen.generation_info,
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        gen.message.content or "", chunk=chunk
                    )
                yield chunk
            return

        self._logger.debug(f"Streaming with {len(messages)} messages")

        oai_messages = self._convert_messages(messages)
        req = self._build_request_kwargs(stop=stop, **kwargs)

        stream = self._openai_client.chat.completions.create(
            messages=oai_messages,
            stream=True,
            **req,
        )

        for event in stream:
            if not event.choices:
                continue

            choice = event.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            content = delta.content or ""
            reasoning_content = getattr(delta, "reasoning_content", "") or ""

            if finish_reason:
                self._logger.debug(
                    "Stream finished",
                    extra={"finish_reason": finish_reason},
                )

            # Build the message chunk
            if reasoning_content:
                msg_chunk = ReasoningAwareAIMessageChunk(
                    content=content,
                    reasoning_content=reasoning_content,
                )
            else:
                msg_chunk = AIMessageChunk(content=content)

            # Handle streaming tool call deltas
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                msg_chunk.tool_call_chunks = [
                    {
                        "name": (
                            tc.function.name
                            if tc.function and tc.function.name
                            else None
                        ),
                        "args": (
                            tc.function.arguments
                            if tc.function and tc.function.arguments
                            else ""
                        ),
                        "id": tc.id,
                        "index": tc.index,
                    }
                    for tc in delta.tool_calls
                ]

            gen_chunk = ChatGenerationChunk(
                message=msg_chunk,
                generation_info=(
                    {"finish_reason": finish_reason} if finish_reason else {}
                ),
            )

            if run_manager:
                run_manager.on_llm_new_token(content, chunk=gen_chunk)

            yield gen_chunk

    # ------------------------------------------------------------------
    # Tool binding & metadata
    # ------------------------------------------------------------------

    def bind_tools(self, tools: list, **kwargs):
        """Bind tools to the model for tool-calling.

        Accepts LangChain BaseTool instances, Pydantic models, or
        OpenAI-format tool dicts.  Returns a RunnableBinding so
        ``model.bind_tools(tools).invoke(messages)`` works.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted: list = []
        for t in tools:
            if isinstance(t, dict):
                formatted.append(t)
            else:
                formatted.append(convert_to_openai_tool(t))

        self._bound_tools = formatted

        if "tool_choice" in kwargs:
            self._tool_choice = kwargs.pop("tool_choice")

        return self.bind(tools=formatted, **kwargs)

    def bind_metadata(self, metadata: dict):
        """Bind additional metadata to the pipeline."""
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(metadata)
        return self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Shutdown the llama.cpp server."""
        if self.started and hasattr(self, "server_manager"):
            self._logger.info(f"Shutting down server for {self.model.name}")
            self.server_manager.stop()
            self.started = False

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        self.shutdown()

    @property
    def _llm_type(self) -> str:
        return "llamacpp_openai_sdk"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        assert self.server_manager is not None
        return {
            "model_name": self.model.name,
            "server_port": self.server_manager.port,
            "pipeline_type": "openai_sdk",
        }

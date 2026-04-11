"""Minimal BaseChatModel wrapper around the OpenAI SDK for remote OpenAI models."""

import json
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from pipelines.llamacpp.chat import ChatLlamaCppPipeline

from config import env_config


class OpenAIChatModel(ChatLlamaCppPipeline.__bases__[0]):  # type: ignore[misc]
    """Thin BaseChatModel that talks to the OpenAI API via the ``openai`` SDK."""

    model_name: str = "gpt-4"

    def __init__(self, model_name: str = "gpt-4", **kwargs):
        from openai import OpenAI

        super().__init__(name=model_name, **kwargs)
        self._model_name = model_name
        self._client = OpenAI(api_key=env_config.OPENAI_API_KEY)

    @property
    def _llm_type(self) -> str:
        return "openai_remote"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        from pipelines.llamacpp.chat import ChatLlamaCppPipeline

        oai_messages = ChatLlamaCppPipeline._convert_messages(messages)
        req: Dict[str, Any] = {"model": self._model_name}
        if stop:
            req["stop"] = stop
        tools = kwargs.get("tools")
        if tools:
            req["tools"] = tools

        response = self._client.chat.completions.create(
            messages=oai_messages,  # type: ignore[arg-type]
            stream=False,
            **req,
        )

        choice = response.choices[0]
        tool_calls = ChatLlamaCppPipeline._parse_tool_calls(choice.message)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=choice.message.content or "",
                        tool_calls=tool_calls,
                        response_metadata={
                            "finish_reason": choice.finish_reason,
                            "model": response.model,
                        },
                    )
                )
            ],
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        from pipelines.llamacpp.chat import ChatLlamaCppPipeline

        oai_messages = ChatLlamaCppPipeline._convert_messages(messages)
        req: Dict[str, Any] = {"model": self._model_name}
        if stop:
            req["stop"] = stop
        tools = kwargs.get("tools")
        if tools:
            req["tools"] = tools

        stream = self._client.chat.completions.create(
            messages=oai_messages,  # type: ignore[arg-type]
            stream=True,
            **req,
        )

        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            content = delta.content or ""
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=chunk)
            yield chunk

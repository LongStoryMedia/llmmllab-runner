"""Minimal BaseChatModel wrapper around the Anthropic SDK for remote Anthropic models."""

from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.language_models import BaseChatModel

from config import env_config


class AnthropicChatModel(BaseChatModel):
    """Thin BaseChatModel that talks to the Anthropic API via the ``anthropic`` SDK."""

    model_name: str = "claude-sonnet-4-20250514"

    def __init__(self, model_name: str = "claude-sonnet-4-20250514", **kwargs):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic SDK is required for the Anthropic provider. "
                "Install with: pip install anthropic"
            ) from e

        super().__init__(name=model_name, **kwargs)
        self._model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "anthropic_remote"

    def _get_client(self):
        import anthropic

        return anthropic.Anthropic(api_key=env_config.ANTHROPIC_API_KEY)

    @staticmethod
    def _convert_messages(
        messages: List[BaseMessage],
    ) -> tuple[Optional[str], List[dict]]:
        """Convert langchain messages to Anthropic format, extracting system prompt."""
        system_prompt: Optional[str] = None
        result: List[dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = str(msg.content)
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content or ""})
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return system_prompt, result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        client = self._get_client()
        system_prompt, anthropic_messages = self._convert_messages(messages)

        req: Dict[str, Any] = {
            "model": self._model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_prompt:
            req["system"] = system_prompt
        if stop:
            req["stop_sequences"] = stop

        response = client.messages.create(**req)
        content = response.content[0].text if response.content else ""

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=content,
                        response_metadata={
                            "stop_reason": response.stop_reason,
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
        client = self._get_client()
        system_prompt, anthropic_messages = self._convert_messages(messages)

        req: Dict[str, Any] = {
            "model": self._model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_prompt:
            req["system"] = system_prompt
        if stop:
            req["stop_sequences"] = stop

        with client.messages.stream(**req) as stream:
            for text in stream.text_stream:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=text)
                )
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

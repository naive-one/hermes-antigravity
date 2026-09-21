from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from .openai_compat import openai_completion_object
from .runtime import generate_chat_completion


def _completion_to_stream(response: Any):
    choice = response.choices[0]
    message = choice.message
    tool_calls = []
    for index, tool_call in enumerate(getattr(message, "tool_calls", None) or []):
        tool_calls.append(
            SimpleNamespace(
                index=index,
                id=getattr(tool_call, "id", None),
                type=getattr(tool_call, "type", "function"),
                function=getattr(tool_call, "function", None),
            )
        )
    delta = SimpleNamespace(
        role="assistant",
        content=getattr(message, "content", None),
        tool_calls=tool_calls or None,
        reasoning=getattr(message, "reasoning", None),
        reasoning_content=getattr(message, "reasoning_content", None),
        reasoning_details=getattr(message, "reasoning_details", None),
    )
    chunk = SimpleNamespace(
        id=getattr(response, "id", ""),
        object="chat.completion.chunk",
        model=getattr(response, "model", ""),
        choices=[
            SimpleNamespace(
                index=0,
                delta=delta,
                finish_reason=getattr(choice, "finish_reason", "stop"),
            )
        ],
        usage=getattr(response, "usage", None),
    )
    return iter((chunk,))


class _AsyncChunkStream:
    def __init__(self, chunks: Any) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


class AntigravityHermesClient:
    """OpenAI-compatible client facade backed by Antigravity's native transport."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or "hermes-antigravity"
        self.base_url = base_url or "antigravity://native"
        self.is_closed = False
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_chat_completion)
        )

    def close(self) -> None:
        self.is_closed = True

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        if self.is_closed:
            raise RuntimeError("Antigravity client is closed")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._complete(kwargs)
        return self._complete_async(kwargs)

    @staticmethod
    def _complete(kwargs: dict[str, Any]) -> Any:
        response = openai_completion_object(generate_chat_completion(kwargs))
        return _completion_to_stream(response) if kwargs.get("stream") else response

    async def _complete_async(self, kwargs: dict[str, Any]) -> Any:
        result = await asyncio.to_thread(self._complete, kwargs)
        return _AsyncChunkStream(result) if kwargs.get("stream") else result

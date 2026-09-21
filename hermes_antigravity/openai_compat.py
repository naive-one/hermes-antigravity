from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from .models import normalize_model_id


@dataclass
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


def parse_chat_request(payload: dict[str, Any]) -> ChatRequest:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    reasoning = payload.get("reasoning_effort")
    if reasoning is None and isinstance(payload.get("reasoning"), dict):
        reasoning = payload["reasoning"].get("effort")
    if reasoning is None and isinstance(payload.get("extra_body"), dict):
        extra_reasoning = payload["extra_body"].get("reasoning")
        if isinstance(extra_reasoning, dict):
            reasoning = extra_reasoning.get("effort")

    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = payload.get("max_completion_tokens")

    return ChatRequest(
        model=normalize_model_id(str(payload.get("model") or "")),
        messages=messages,
        tools=payload.get("tools") if isinstance(payload.get("tools"), list) else [],
        tool_choice=payload.get("tool_choice"),
        reasoning_effort=str(reasoning) if reasoning is not None else None,
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        temperature=payload.get("temperature") if isinstance(payload.get("temperature"), (int, float)) else None,
        top_p=payload.get("top_p") if isinstance(payload.get("top_p"), (int, float)) else None,
    )


def _response(upstream: dict[str, Any]) -> dict[str, Any]:
    return upstream.get("response") if isinstance(upstream.get("response"), dict) else upstream


def _finish(reason: str | None) -> str:
    if reason == "MAX_TOKENS":
        return "length"
    if reason in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"}:
        return "content_filter"
    return "stop"


def _tool_call_id(call: dict[str, Any]) -> str:
    raw = json.dumps(call, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "call_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _usage(resp: dict[str, Any]) -> dict[str, int]:
    usage = resp.get("usageMetadata") or {}
    prompt = int(usage.get("promptTokenCount") or 0)
    thoughts = int(usage.get("thoughtsTokenCount") or 0)
    completion = int(usage.get("candidatesTokenCount") or 0) + thoughts
    total = int(usage.get("totalTokenCount") or prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": thoughts,
    }


def to_openai_completion(model: str, upstream: dict[str, Any]) -> dict[str, Any]:
    resp = _response(upstream)
    candidates = resp.get("candidates") or []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    parts = ((candidate.get("content") or {}).get("parts") or []) if isinstance(candidate, dict) else []

    text: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    text_signatures: list[str] = []
    reasoning_signatures: list[str] = []

    for part in parts:
        if not isinstance(part, dict):
            continue
        if "functionCall" in part:
            call = part.get("functionCall") or {}
            name = call.get("name") or "tool"
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            tool_call: dict[str, Any] = {
                "id": call.get("id") or _tool_call_id(call),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, separators=(",", ":"), ensure_ascii=False),
                },
            }
            signature = part.get("thoughtSignature")
            if isinstance(signature, str) and signature:
                # Hermes/LiteLLM may ignore this extension, but preserving it in
                # the raw completion lets compatible transports replay it.
                tool_call["thought_signature"] = signature
                tool_call["extra_content"] = {"google": {"thought_signature": signature}}
            tool_calls.append(tool_call)
            continue

        value = part.get("text")
        if not isinstance(value, str):
            continue
        signature = part.get("thoughtSignature")
        if part.get("thought"):
            reasoning.append(value)
            if isinstance(signature, str) and signature:
                reasoning_signatures.append(signature)
        else:
            text.append(value)
            if isinstance(signature, str) and signature:
                text_signatures.append(signature)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text) if text else None,
    }
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if reasoning_signatures:
        message["reasoning_signature"] = reasoning_signatures[-1]
    if text_signatures:
        message["text_signature"] = text_signatures[-1]
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": normalize_model_id(model),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else _finish(candidate.get("finishReason")),
            }
        ],
        "usage": _usage(resp),
    }


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        # Hermes' ChatCompletionsTransport expects tool_call.extra_content to
        # remain a plain dict so Gemini thought_signature can be persisted in
        # ToolCall.provider_data and replayed on the next turn.
        return SimpleNamespace(
            **{
                key: item if key == "extra_content" and isinstance(item, dict) else _namespace(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def openai_completion_object(completion: dict[str, Any]) -> SimpleNamespace:
    completion = dict(completion)
    choices = []
    for raw_choice in completion.get("choices") or []:
        choice = dict(raw_choice)
        message = dict(choice.get("message") or {})
        message.setdefault("content", None)
        message.setdefault("tool_calls", None)
        choice["message"] = message
        choices.append(choice)
    completion["choices"] = choices
    completion.setdefault(
        "usage",
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    return _namespace(completion)

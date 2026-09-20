from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from copy import deepcopy
from typing import Any

from .models import (
    get_max_output_tokens,
    get_model_enum,
    normalize_model_id,
    normalize_effort,
    resolve_wire_model_id,
    strip_provider_prefix,
)

SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def thinking_config(runtime_model: str, effort: str | None) -> dict[str, Any] | None:
    level = normalize_effort(effort)
    if runtime_model.startswith("claude-"):
        return {
            "includeThoughts": level != "off",
            "thinkingBudget": 1024 if level != "off" else 0,
        }
    if runtime_model.startswith("gpt-oss-") or runtime_model.startswith("openai/gpt-oss-"):
        return {
            "includeThoughts": level != "off",
            "thinkingBudget": 8192 if level != "off" else 0,
        }
    if runtime_model.startswith("gemini-3.5-flash") or runtime_model == "gemini-3-flash-agent":
        budget = 0 if level == "off" else (10000 if level == "high" else 4000 if level == "medium" else 1000)
        return {"includeThoughts": level != "off", "thinkingBudget": budget}
    if runtime_model.startswith("gemini-3.1-pro") or runtime_model == "gemini-pro-agent":
        budget = 0 if level == "off" else (10001 if level == "high" else 1001)
        return {"includeThoughts": level != "off", "thinkingBudget": budget}
    if runtime_model.startswith("gemini-"):
        budget = 0 if level == "off" else (-1 if level == "high" else 4000 if level == "medium" else 1000)
        return {"includeThoughts": level != "off", "thinkingBudget": budget}
    return None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
        return "\n".join(texts)
    return str(content)


def _parts_from_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str) and item["text"].strip():
                part: dict[str, Any] = {"text": item["text"]}
                signature = item.get("thoughtSignature") or item.get("thought_signature") or item.get("textSignature")
                if isinstance(signature, str) and signature:
                    part["thoughtSignature"] = signature
                parts.append(part)
            elif item.get("type") == "image_url":
                image = item.get("image_url")
                url = image.get("url") if isinstance(image, dict) else None
                if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                    meta, data = url.split(",", 1)
                    mime = meta[5:].split(";", 1)[0] or "application/octet-stream"
                    parts.append({"inlineData": {"mimeType": mime, "data": data}})
                else:
                    parts.append({"text": "[image omitted]"})
        return parts
    text = _content_text(content)
    return [{"text": text}] if text.strip() else []


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": raw}
    return {}


def _clean_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    banned = {
        "$schema",
        "$defs",
        "definitions",
        "additionalProperties",
        "patternProperties",
        "unevaluatedProperties",
    }

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in banned}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    out = clean(deepcopy(schema))
    if not isinstance(out, dict):
        return {"type": "object", "properties": {}}
    out.setdefault("type", "object")
    out.setdefault("properties", {})
    return out


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    declarations: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        declarations.append(
            {
                "name": fn["name"],
                "description": fn.get("description") or "",
                "parameters": _clean_schema(fn.get("parameters")),
            }
        )
    return [{"functionDeclarations": declarations}] if declarations else None


def _tool_config(tools: list[dict[str, Any]], tool_choice: Any, runtime_model: str) -> dict[str, Any] | None:
    if isinstance(tool_choice, str):
        choice = tool_choice.lower()
        if choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if choice in {"required", "any"}:
            return {"functionCallingConfig": {"mode": "ANY"}}
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if tool_choice.get("type") == "function" else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    if tools or runtime_model.startswith("claude-"):
        return {"functionCallingConfig": {"mode": "VALIDATED"}}
    return None


def _session_id(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            text = _content_text(message.get("content"))
            if text.strip():
                digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
                return "-" + str(int.from_bytes(digest, "big") & ((1 << 63) - 1))
    return "-" + str(secrets.randbelow(9_000_000_000_000_000_000))


def _tool_call_signature(tool_call: dict[str, Any]) -> str | None:
    for key in ("thoughtSignature", "thought_signature", "signature"):
        value = tool_call.get(key)
        if isinstance(value, str) and value:
            return value
    extra = tool_call.get("extra_content")
    if isinstance(extra, dict):
        google = extra.get("google")
        if isinstance(google, dict):
            value = google.get("thought_signature") or google.get("thoughtSignature")
            if isinstance(value, str) and value:
                return value
    return None


def _requires_thought_signature(runtime_model: str) -> bool:
    return runtime_model.startswith("gemini-")


def _envelope_labels(runtime_model: str, *, model_enum: str | None = None, step: int = 2) -> dict[str, str]:
    labels = {
        "last_step_index": str(step - 1),
        "trajectory_id": str(uuid.uuid4()),
        "used_claude": str(runtime_model.startswith("claude-")).lower(),
        "used_claude_conservative": str(runtime_model.startswith("claude-")).lower(),
    }
    resolved = model_enum or get_model_enum(runtime_model)
    if resolved:
        labels["model_enum"] = resolved
    return labels


def build_generate_content_request(
    *,
    model: str,
    project_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    tool_choice: Any = None,
    runtime_model_override: str | None = None,
    model_enum_override: str | None = None,
) -> dict[str, Any]:
    logical_model = normalize_model_id(model)
    runtime_model = runtime_model_override or resolve_wire_model_id(logical_model, reasoning_effort)
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in {"system", "developer"}:
            text = _content_text(message.get("content"))
            if text.strip():
                system_parts.append({"text": text})
            continue

        if role == "user":
            parts = _parts_from_content(message.get("content"))
            if parts:
                contents.append({"role": "user", "parts": parts})
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                thought_part: dict[str, Any] = {"thought": True, "text": reasoning}
                sig = message.get("reasoning_signature") or message.get("thoughtSignature")
                if isinstance(sig, str) and sig:
                    thought_part["thoughtSignature"] = sig
                parts.append(thought_part)
            parts.extend(_parts_from_content(message.get("content")))

            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                fn = tool_call.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else None
                if not name:
                    continue
                call_id = tool_call.get("id")
                if call_id:
                    call_names[str(call_id)] = str(name)
                call: dict[str, Any] = {
                    "functionCall": {
                        "name": name,
                        "args": _parse_args(fn.get("arguments") if isinstance(fn, dict) else None),
                    }
                }
                if runtime_model.startswith("claude-") or runtime_model.startswith("gpt-oss-"):
                    if call_id:
                        call["functionCall"]["id"] = str(call_id)
                signature = _tool_call_signature(tool_call)
                if signature:
                    call["thoughtSignature"] = signature
                elif _requires_thought_signature(runtime_model):
                    call["thoughtSignature"] = SKIP_THOUGHT_SIGNATURE
                parts.append(call)

            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if role in {"tool", "function"}:
            name = message.get("name") or call_names.get(str(message.get("tool_call_id") or "")) or "tool"
            response: dict[str, Any] = {
                "name": str(name),
                "response": {"output": _content_text(message.get("content"))},
            }
            call_id = message.get("tool_call_id")
            if call_id and (runtime_model.startswith("claude-") or runtime_model.startswith("gpt-oss-")):
                response["id"] = str(call_id)
            part = {"functionResponse": response}
            if (
                contents
                and contents[-1].get("role") == "user"
                and any("functionResponse" in p for p in contents[-1].get("parts", []))
            ):
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Continue the active task."}]})
    elif contents[-1].get("role") == "model":
        contents.append({"role": "user", "parts": [{"text": "Continue the active task using the available context."}]})

    cap = get_max_output_tokens(logical_model, runtime_model)
    generation_config: dict[str, Any] = {
        "maxOutputTokens": min(max_tokens, cap) if isinstance(max_tokens, int) and max_tokens > 0 else cap,
    }
    thinking = thinking_config(runtime_model, reasoning_effort)
    if thinking is not None:
        generation_config["thinkingConfig"] = thinking
    if temperature is not None:
        generation_config["temperature"] = temperature
    if top_p is not None:
        generation_config["topP"] = top_p

    request: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
        "sessionId": _session_id(messages),
        "labels": _envelope_labels(runtime_model, model_enum=model_enum_override),
    }
    if system_parts:
        request["systemInstruction"] = {"role": "system", "parts": system_parts}
    converted_tools = _tools(tools or [])
    if converted_tools:
        request["tools"] = converted_tools
    tool_config = _tool_config(tools or [], tool_choice, runtime_model)
    if tool_config:
        request["toolConfig"] = tool_config

    return {
        "project": project_id,
        "model": runtime_model,
        "request": request,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": f"agent/{uuid.uuid4()}/{int(time.time() * 1000)}/{uuid.uuid4()}/2",
    }

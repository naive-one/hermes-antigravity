from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from copy import deepcopy
from typing import Any

from .models import (
    get_max_output_tokens,
    get_model_enum,
    normalize_effort,
    normalize_model_id,
    resolve_wire_model_id,
)

ANTIGRAVITY_SYSTEM_INSTRUCTION = (
    "You are Antigravity, a powerful agentic AI coding assistant designed by Google DeepMind. "
    "You are pair programming with a user to solve coding tasks. Be concise, practical, and tool-aware."
)
ANTIGRAVITY_NO_PREAMBLE_INSTRUCTION = (
    'CRITICAL: NEVER output rule checks, formatting guidelines, constraint checklists '
    '(e.g. "No emdashes"), or your thinking/personality preambles in the final response. '
    "Output only the final response."
)
CONTINUATION_TEXT = "Continue the active task using the available instructions and context."
_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


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


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    return "".join("\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in text)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _sanitize_text(content)
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(_sanitize_text(item["text"]))
        return "\n".join(texts)
    return _sanitize_text(content)


def _parts_from_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str) and item["text"].strip():
                part: dict[str, Any] = {"text": _sanitize_text(item["text"])}
                signature = item.get("thoughtSignature") or item.get("thought_signature") or item.get("textSignature")
                if _valid_thought_signature(signature):
                    part["thoughtSignature"] = signature
                parts.append(part)
            elif item.get("type") in {"image_url", "image"}:
                if item.get("type") == "image_url":
                    image = item.get("image_url")
                    raw = image.get("url") if isinstance(image, dict) else None
                    explicit_mime = None
                else:
                    source = item.get("source") if isinstance(item.get("source"), dict) else {}
                    raw = item.get("data") or source.get("data")
                    explicit_mime = item.get("mimeType") or item.get("mediaType") or source.get("mediaType")
                if isinstance(raw, str) and raw.startswith("data:") and ";base64," in raw:
                    meta, data = raw.split(",", 1)
                    mime = explicit_mime or meta[5:].split(";", 1)[0] or "image/png"
                    parts.append({"inlineData": {"mimeType": mime, "data": data.strip()}})
                elif isinstance(raw, str) and item.get("type") == "image":
                    parts.append({"inlineData": {"mimeType": explicit_mime or "image/png", "data": raw.strip()}})
                elif raw:
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


def _resolve_local_pointer(ref: str, root: Any) -> Any:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported non-local schema reference: {ref}")
    current = root
    for token in ref[2:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not key.isdigit() or int(key) >= len(current):
                raise ValueError(f"unresolved schema reference: {ref}")
            current = current[int(key)]
        elif isinstance(current, dict) and key in current:
            current = current[key]
        else:
            raise ValueError(f"unresolved schema reference: {ref}")
    return current


def _dereference_schema(schema: Any) -> Any:
    root = deepcopy(schema)
    nodes = 0

    def walk(value: Any, refs: tuple[str, ...] = (), depth: int = 0) -> Any:
        nonlocal nodes
        nodes += 1
        if nodes > 10000 or depth > 64:
            raise ValueError("tool schema is too deep or too large")
        if isinstance(value, list):
            return [walk(v, refs, depth + 1) for v in value]
        if not isinstance(value, dict):
            return value

        ref = value.get("$ref")
        if isinstance(ref, str):
            if ref in refs:
                raise ValueError(f"recursive schema reference: {ref}")
            target = _resolve_local_pointer(ref, root)
            resolved = walk(deepcopy(target), refs + (ref,), depth + 1)
            siblings = {k: v for k, v in value.items() if k != "$ref"}
            if siblings:
                siblings = walk(siblings, refs, depth + 1)
                if isinstance(resolved, dict) and isinstance(siblings, dict):
                    return {**resolved, **siblings}
            return resolved

        return {k: walk(v, refs, depth + 1) for k, v in value.items()}

    return walk(root)


def _clean_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    dereferenced = _dereference_schema(schema)
    banned = {
        "$schema",
        "$defs",
        "definitions",
        "additionalProperties",
        "patternProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
    }

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in banned}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    out = clean(dereferenced)
    if not isinstance(out, dict):
        return {"type": "object", "properties": {}}
    out.setdefault("type", "object")
    out.setdefault("properties", {})
    return out


def _tools(tools: list[dict[str, Any]], runtime_model: str) -> list[dict[str, Any]] | None:
    declarations: list[dict[str, Any]] = []
    use_legacy_parameters = runtime_model.startswith("claude-") or runtime_model.startswith("gpt-oss-")
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        try:
            schema = _clean_schema(fn.get("parameters"))
        except ValueError:
            # Current pi-antigravity skips a declaration whose local schema
            # references cannot be resolved rather than sending invalid JSON.
            continue
        declaration: dict[str, Any] = {
            "name": fn["name"],
            "description": fn.get("description") or "",
        }
        if use_legacy_parameters:
            declaration["parameters"] = schema
        else:
            declaration["parametersJsonSchema"] = schema
        declarations.append(declaration)
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


def _valid_thought_signature(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) % 4 == 0
        and bool(_BASE64_SIGNATURE_PATTERN.fullmatch(value))
    )


def _gemini_requires_thought_signature(runtime_model: str) -> bool:
    if not runtime_model.startswith("gemini-"):
        return False
    match = re.match(r"^gemini-(\d+)", runtime_model)
    return not match or int(match.group(1)) >= 3


def _tool_call_signature(tool_call: dict[str, Any]) -> str | None:
    for key in ("thoughtSignature", "thought_signature", "signature"):
        value = tool_call.get(key)
        if _valid_thought_signature(value):
            return value
    extra = tool_call.get("extra_content")
    if isinstance(extra, dict):
        google = extra.get("google")
        if isinstance(google, dict):
            value = google.get("thought_signature") or google.get("thoughtSignature")
            if _valid_thought_signature(value):
                return value
    return None


def _sanitize_tool_call_id(value: Any, fallback_name: str = "tool") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(value or ""))[:64]
    return cleaned or f"{fallback_name}_call"


def _append_turn(contents: list[dict[str, Any]], role: str, parts: list[dict[str, Any]]) -> None:
    if not parts:
        return
    if contents and contents[-1].get("role") == role:
        contents[-1].setdefault("parts", []).extend(parts)
    else:
        contents.append({"role": role, "parts": parts})


def _stable_uuid(seed: str) -> str:
    raw = bytearray(hashlib.sha1(seed.encode("utf-8")).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x50
    raw[8] = (raw[8] & 0x3F) | 0x80
    hexed = raw.hex()
    return f"{hexed[:8]}-{hexed[8:12]}-{hexed[12:16]}-{hexed[16:20]}-{hexed[20:]}"


def _envelope(
    messages: list[dict[str, Any]],
    contents: list[dict[str, Any]],
    runtime_model: str,
    model_enum: str | None,
) -> tuple[str, str, dict[str, str]]:
    first = messages[0] if messages and isinstance(messages[0], dict) else {}
    seed = (
        f"{first.get('role') or 'user'}:"
        f"{first.get('timestamp') or ''}:"
        f"{_content_text(first.get('content'))[:64]}"
    )
    conversation_id = _stable_uuid(f"antigravity:conv:{seed}")
    trajectory_id = _stable_uuid(f"antigravity:traj:{seed}")
    step = max(1, len(contents))
    request_index = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
    raw_session = secrets.randbits(64)
    if raw_session >= 1 << 63:
        raw_session -= 1 << 64
    session_id = str(raw_session)

    is_claude = runtime_model.startswith("claude-")
    is_non_gemini = is_claude or not runtime_model.startswith("gemini-")
    labels = {
        "last_step_index": str(max(0, len(contents) - 1)),
        "request_id": f"{trajectory_id}-{request_index}",
        "trajectory_id": trajectory_id,
        "used_claude": str(is_claude).lower(),
        "used_claude_conservative": str(is_claude).lower(),
        "used_non_gemini_model": str(is_non_gemini).lower(),
    }
    resolved_enum = model_enum or get_model_enum(runtime_model)
    if resolved_enum:
        labels["model_enum"] = resolved_enum

    request_id = (
        f"agent/{conversation_id}/{int(time.time() * 1000)}/{trajectory_id}/{step}"
    )
    return request_id, session_id, labels


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
    dropped_tool_calls: dict[str, tuple[str, str]] = {}
    requires_signature = _gemini_requires_thought_signature(runtime_model)

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
            _append_turn(contents, "user", _parts_from_content(message.get("content")))
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                signature = message.get("reasoning_signature") or message.get("thoughtSignature")
                thought_part: dict[str, Any] = {"thought": True, "text": _sanitize_text(reasoning)}
                if _valid_thought_signature(signature):
                    thought_part["thoughtSignature"] = signature
                parts.append(thought_part)
            parts.extend(_parts_from_content(message.get("content")))

            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                fn = tool_call.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else None
                if not name:
                    continue
                args = _parse_args(fn.get("arguments") if isinstance(fn, dict) else None)
                call_id = str(tool_call.get("id") or "")
                if call_id:
                    call_names[call_id] = str(name)

                signature = _tool_call_signature(tool_call)
                if requires_signature and not signature:
                    args_text = json.dumps(args, separators=(",", ":"), ensure_ascii=False)
                    key = call_id or f"empty:{name}"
                    dropped_tool_calls[key] = (str(name), args_text)
                    if call_id:
                        dropped_tool_calls[_sanitize_tool_call_id(call_id, str(name))] = (str(name), args_text)
                    continue

                call: dict[str, Any] = {
                    "functionCall": {
                        "name": name,
                        "args": args,
                    }
                }
                if runtime_model.startswith("claude-") or runtime_model.startswith("gpt-oss-"):
                    call["functionCall"]["id"] = _sanitize_tool_call_id(call_id, str(name))
                if signature:
                    call["thoughtSignature"] = signature
                parts.append(call)

            _append_turn(contents, "model", parts)
            continue

        if role in {"tool", "function"}:
            call_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or call_names.get(call_id) or "tool")
            response_text = _content_text(message.get("content"))
            dropped = dropped_tool_calls.get(call_id)
            if dropped is None and call_id:
                dropped = dropped_tool_calls.get(_sanitize_tool_call_id(call_id, name))
            if dropped is not None:
                dropped_name, args_text = dropped
                label = f"`{dropped_name}`" if args_text == "{}" else f"`{dropped_name}` ({args_text})"
                _append_turn(
                    contents,
                    "user",
                    [{"text": _sanitize_text(f"[Observation from {label}:\n{response_text}]")}],
                )
                continue

            response: dict[str, Any] = {
                "name": name,
                "response": {"output": response_text},
            }
            if call_id and (runtime_model.startswith("claude-") or runtime_model.startswith("gpt-oss-")):
                response["id"] = _sanitize_tool_call_id(call_id, name)
            _append_turn(contents, "user", [{"functionResponse": response}])

    # Compaction may leave a function-call model turn without the user boundary
    # required by Antigravity. Restore it before sending the request.
    index = 0
    while index < len(contents):
        turn = contents[index]
        has_call = (
            turn.get("role") == "model"
            and any("functionCall" in p for p in turn.get("parts", []))
        )
        if has_call and (index == 0 or contents[index - 1].get("role") != "user"):
            contents.insert(index, {"role": "user", "parts": [{"text": CONTINUATION_TEXT}]})
            index += 1
        index += 1

    has_user_text = any(
        turn.get("role") == "user"
        and any(isinstance(p.get("text"), str) and p["text"].strip() for p in turn.get("parts", []))
        for turn in contents
    )
    if not has_user_text and contents:
        user_turn = next((turn for turn in contents if turn.get("role") == "user"), None)
        if user_turn is None:
            contents.insert(0, {"role": "user", "parts": [{"text": CONTINUATION_TEXT}]})
        else:
            user_turn.setdefault("parts", []).append({"text": CONTINUATION_TEXT})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Apply the active system instructions."}]})
    elif contents[-1].get("role") == "model":
        if any("functionCall" in p for p in contents[-1].get("parts", [])):
            raise ValueError(
                "Antigravity request is missing tool result(s) for the final assistant tool call."
            )
        _append_turn(contents, "user", [{"text": CONTINUATION_TEXT}])

    if not system_parts:
        system_parts = [
            {"text": ANTIGRAVITY_SYSTEM_INSTRUCTION},
            {"text": ANTIGRAVITY_NO_PREAMBLE_INSTRUCTION},
        ]

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

    request_id, session_id, labels = _envelope(
        messages,
        contents,
        runtime_model,
        model_enum_override,
    )
    request: dict[str, Any] = {
        "contents": contents,
        "systemInstruction": {"role": "user", "parts": system_parts},
        "generationConfig": generation_config,
        "sessionId": session_id,
        "labels": labels,
    }
    converted_tools = _tools(tools or [], runtime_model)
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
        "requestId": request_id,
    }

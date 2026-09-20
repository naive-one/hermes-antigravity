from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable

from .cloudcode import MODEL_ENDPOINTS, request_headers
from .errors import AntigravityError, TokenExpired

STREAM_PATH = "/v1internal:streamGenerateContent?alt=sse"


def _sse_json_lines(response: Iterable[bytes]) -> Iterable[dict[str, Any]]:
    data_lines: list[str] = []
    for raw in response:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload != "[DONE]":
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                return


def _meaningful(response: dict[str, Any]) -> bool:
    for candidate in response.get("candidates") or []:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        for part in (content or {}).get("parts") or []:
            if not isinstance(part, dict):
                continue
            if part.get("functionCall"):
                return True
            if isinstance(part.get("text"), str) and part["text"].strip() and not part.get("thought"):
                return True
    return False


def _friendly_error(status: int, detail: str) -> str:
    try:
        parsed = json.loads(detail)
        error = parsed.get("error") if isinstance(parsed, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str) and message:
            return message
    except Exception:
        pass
    return detail.strip() or f"HTTP {status}"


class AntigravityClient:
    def __init__(
        self,
        *,
        endpoints: list[str] | tuple[str, ...] | None = None,
        stream_timeout: float = 300,
        post_json: Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]] | None = None,
    ):
        self.endpoints = [e.rstrip("/") for e in (endpoints or MODEL_ENDPOINTS)]
        self.stream_timeout = stream_timeout
        self.post_json = post_json

    def _headers(self, access_token: str) -> dict[str, str]:
        return request_headers(access_token, accept="text/event-stream")

    def stream_generate(self, *, access_token: str, body: dict[str, Any]) -> Iterable[dict[str, Any]]:
        payload = json.dumps(body).encode("utf-8")
        headers = self._headers(access_token)
        last_error: AntigravityError | None = None

        for endpoint in self.endpoints:
            req = urllib.request.Request(
                endpoint + STREAM_PATH,
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.stream_timeout) as resp:
                    for event in _sse_json_lines(resp):
                        if event.get("error"):
                            error = event.get("error") or {}
                            code = int(error.get("code") or 500) if isinstance(error, dict) else 500
                            message = error.get("message") if isinstance(error, dict) else None
                            if code == 401:
                                raise TokenExpired()
                            raise AntigravityError(str(message or "Antigravity stream error"), status=code)
                        yield event.get("response") if isinstance(event.get("response"), dict) else event
                    return
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                if exc.code == 401:
                    raise TokenExpired() from exc
                last_error = AntigravityError(
                    f"Cloud Code Assist API error ({exc.code}): {_friendly_error(exc.code, detail)}",
                    status=exc.code,
                )
                # 404 means the runtime id itself failed: let runtime.py try the
                # next model candidate instead of probing other endpoints forever.
                if exc.code == 404:
                    break
                # Quota/auth/client failures are normally endpoint-independent.
                if exc.code in {400, 403, 429}:
                    break
                # 5xx continues to the next production/sandbox endpoint.
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = AntigravityError(f"Cloud Code Assist connection failed: {exc}", status=502)

        if last_error:
            raise last_error
        raise AntigravityError("Antigravity request failed without a response", status=502)

    def generate(self, *, access_token: str, body: dict[str, Any], empty_retries: int = 2) -> dict[str, Any]:
        if self.post_json is not None:
            return self.post_json(self.endpoints[0] + STREAM_PATH, body, self._headers(access_token))

        last: dict[str, Any] = {
            "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}]
        }
        for attempt in range(empty_retries + 1):
            parts: list[dict[str, Any]] = []
            finish = "STOP"
            usage: dict[str, Any] = {}
            response_id: str | None = None
            for chunk in self.stream_generate(access_token=access_token, body=body):
                response_id = chunk.get("responseId") or response_id
                usage = chunk.get("usageMetadata") or usage
                candidate = (chunk.get("candidates") or [{}])[0]
                content = candidate.get("content") if isinstance(candidate, dict) else None
                parts.extend((content or {}).get("parts") or [])
                finish = candidate.get("finishReason") or finish if isinstance(candidate, dict) else finish
            last = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": parts},
                        "finishReason": finish,
                    }
                ],
                "usageMetadata": usage,
            }
            if response_id:
                last["responseId"] = response_id
            if _meaningful(last) or attempt >= empty_retries:
                return last
        return last

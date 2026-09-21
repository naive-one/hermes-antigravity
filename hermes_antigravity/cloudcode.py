from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .errors import AntigravityError
from .models import register_discovered_model_enums

CLOUD_CODE_ENDPOINT = "https://cloudcode-pa.googleapis.com"
MODEL_ENDPOINTS = (
    "https://daily-cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    CLOUD_CODE_ENDPOINT,
)
ANTIGRAVITY_LOAD_CODE_ASSIST_METADATA = {
    "ideType": "ANTIGRAVITY",
    "platform": "PLATFORM_UNSPECIFIED",
    "pluginType": "GEMINI",
}
TIER_LEGACY = "legacy-tier"
PROJECT_ONBOARD_MAX_ATTEMPTS = 5
PROJECT_ONBOARD_INTERVAL_SECONDS = 2
DISCOVERY_TIMEOUT_SECONDS = 8
CATALOG_TTL_SECONDS = 30 * 60
_CATALOG_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def antigravity_user_agent() -> str:
    explicit = (os.getenv("ANTIGRAVITY_USER_AGENT") or "").strip()
    if explicit:
        return explicit

    system = platform.system().lower()
    os_name = "windows" if system.startswith("win") else ("darwin" if system == "darwin" else system or "linux")
    machine = platform.machine().lower()
    arch = "amd64" if machine in {"x86_64", "x64"} else ("386" if machine in {"i386", "i686"} else machine or "arm64")

    legacy_version = (os.getenv("PI_AI_ANTIGRAVITY_VERSION") or "").strip()
    if legacy_version:
        return f"antigravity/hub/{legacy_version} {os_name}/{arch}"

    cli_version = (os.getenv("ANTIGRAVITY_CLI_VERSION") or "1.1.23").strip()
    client_cl = (os.getenv("ANTIGRAVITY_CLIENT_CL") or "974125021").strip()
    return (
        f"antigravity/cli/{cli_version} "
        f"(aidev_client; os_type={os_name}; arch={arch}; cl={client_cl}; auth_method=consumer)"
    )


def request_headers(access_token: str, *, accept: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": antigravity_user_agent(),
    }
    if accept:
        headers["Accept"] = accept
    return headers


def _post_json(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float = 120,
) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise AntigravityError(f"Cloud Code Assist API error ({exc.code}): {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AntigravityError(f"Cloud Code Assist connection failed: {exc}", status=502) from exc


def read_project_id(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"]:
        return value["id"]
    return None


def extract_project_id(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in (
        "antigravityProjectId",
        "projectId",
        "backendProjectId",
        "userDefinedCloudaicompanionProject",
        "cloudaicompanionProject",
        "project",
    ):
        project = read_project_id(payload.get(key))
        if project:
            return project
    for key in ("projects", "projectIds", "cloudaicompanionProjects"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                project = extract_project_id(item) or read_project_id(item)
                if project:
                    return project
    return None


def read_default_tier(allowed_tiers: object) -> str:
    if not isinstance(allowed_tiers, list):
        return TIER_LEGACY
    for tier in allowed_tiers:
        if isinstance(tier, dict) and tier.get("isDefault") and isinstance(tier.get("id"), str) and tier["id"]:
            return tier["id"]
    return TIER_LEGACY


def stable_project_id(seed: str = "antigravity-default") -> str:
    digest = bytearray(hashlib.sha1(f"antigravity:{seed}".encode("utf-8")).digest()[:16])
    digest[6] = (digest[6] & 0x0F) | 0x50
    digest[8] = (digest[8] & 0x3F) | 0x80
    hexed = digest.hex()
    return f"{hexed[:8]}-{hexed[8:12]}-{hexed[12:16]}-{hexed[16:20]}-{hexed[20:]}"


def default_project_id(seed: str = "antigravity-default") -> str:
    return (os.getenv("ANTIGRAVITY_PROJECT_ID") or "").strip() or stable_project_id(seed)


def _list_cloud_projects(access_token: str) -> str | None:
    headers = request_headers(access_token)
    for endpoint in MODEL_ENDPOINTS:
        try:
            payload = _post_json(
                f"{endpoint}/v1internal:listCloudAICompanionProjects",
                {},
                headers,
                timeout=DISCOVERY_TIMEOUT_SECONDS,
            )
        except Exception:
            continue
        project = extract_project_id(payload)
        if project:
            return project
    return None


def load_or_onboard_project(
    access_token: str,
    *,
    seed: str = "antigravity-default",
    post_json: Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    explicit = (os.getenv("ANTIGRAVITY_PROJECT_ID") or "").strip()
    if explicit:
        return explicit

    sender = post_json
    headers = request_headers(access_token)
    load_payload: dict[str, Any] | None = None

    # Match pi-antigravity: production first, then sandbox/cloudcode fallback,
    # with short metadata timeouts.
    for endpoint in MODEL_ENDPOINTS:
        try:
            if sender is None:
                payload = _post_json(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    {"metadata": {"ideType": "ANTIGRAVITY"}},
                    headers,
                    timeout=DISCOVERY_TIMEOUT_SECONDS,
                )
            else:
                payload = sender(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    {"metadata": {"ideType": "ANTIGRAVITY"}},
                    headers,
                )
        except Exception:
            continue
        if isinstance(payload, dict):
            load_payload = payload
            project = extract_project_id(payload)
            if project:
                return project
            if sender is None:
                listed = _list_cloud_projects(access_token)
                if listed:
                    return listed
            break

    # Keep the legacy onboarding route as an additional compatibility path.
    if isinstance(load_payload, dict):
        onboard_body = {
            "tierId": read_default_tier(load_payload.get("allowedTiers")),
            "metadata": ANTIGRAVITY_LOAD_CODE_ASSIST_METADATA,
        }
        for endpoint in MODEL_ENDPOINTS:
            for attempt in range(PROJECT_ONBOARD_MAX_ATTEMPTS):
                if attempt:
                    sleep(PROJECT_ONBOARD_INTERVAL_SECONDS)
                try:
                    if sender is None:
                        op = _post_json(
                            f"{endpoint}/v1internal:onboardUser",
                            onboard_body,
                            headers,
                            timeout=DISCOVERY_TIMEOUT_SECONDS,
                        )
                    else:
                        op = sender(f"{endpoint}/v1internal:onboardUser", onboard_body, headers)
                except Exception:
                    break
                project = extract_project_id((op.get("response") or {}) if isinstance(op, dict) else None)
                if isinstance(op, dict) and op.get("done") and project:
                    return project

    # pi-antigravity deliberately has a deterministic final fallback so a
    # transient project-discovery outage does not block inference entirely.
    return default_project_id(seed)


def fetch_available_models(
    access_token: str,
    project_id: str,
    *,
    force: bool = False,
    post_json: Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache_key = (access_token, project_id)
    now = time.time()
    if post_json is None and not force:
        cached = _CATALOG_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return dict(cached[1])

    merged: dict[str, Any] = {}
    headers = request_headers(access_token)
    body = {"project": project_id}

    def fetch_one(endpoint: str) -> dict[str, Any]:
        if post_json is None:
            return _post_json(
                f"{endpoint}/v1internal:fetchAvailableModels",
                body,
                headers,
                timeout=DISCOVERY_TIMEOUT_SECONDS,
            )
        return post_json(f"{endpoint}/v1internal:fetchAvailableModels", body, headers)

    # pi-antigravity merges the account catalog across endpoint candidates.
    # Probe them concurrently so a slow sandbox endpoint does not add its
    # timeout serially to every catalog refresh.
    with ThreadPoolExecutor(max_workers=len(MODEL_ENDPOINTS)) as pool:
        futures = [pool.submit(fetch_one, endpoint) for endpoint in MODEL_ENDPOINTS]
        for future in as_completed(futures):
            try:
                payload = future.result()
            except Exception:
                continue
            models = payload.get("models") if isinstance(payload, dict) else None
            if isinstance(models, dict):
                merged.update(models)

    if merged:
        register_discovered_model_enums(merged)
        if post_json is None:
            _CATALOG_CACHE[cache_key] = (now + CATALOG_TTL_SECONDS, dict(merged))
    return merged


def clear_catalog_cache() -> None:
    _CATALOG_CACHE.clear()


def fetch_quota_summary(
    access_token: str,
    project_id: str | None = None,
    *,
    post_json: Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    headers = request_headers(access_token)
    bodies = [{"project": project_id}] if project_id else [{}]
    bodies.append({})
    for endpoint in MODEL_ENDPOINTS:
        for body in bodies:
            try:
                if post_json is None:
                    return _post_json(
                        f"{endpoint}/v1internal:retrieveUserQuota",
                        body,
                        headers,
                        timeout=DISCOVERY_TIMEOUT_SECONDS,
                    )
                return post_json(f"{endpoint}/v1internal:retrieveUserQuota", body, headers)
            except Exception:
                continue
    return {}

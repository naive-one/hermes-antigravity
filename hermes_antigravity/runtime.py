from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .client import AntigravityClient
from .cloudcode import fetch_available_models, fetch_quota_summary, load_or_onboard_project
from .credentials import CredentialStore, load_agy_keychain_credentials
from .errors import AntigravityError, TokenExpired
from .models import (
    DEFAULT_MODEL,
    KNOWN_MODELS,
    get_fallback_runtime_model,
    get_model_enum,
    normalize_effort,
    public_models_from_catalog,
    resolve_wire_model_id,
    strip_provider_prefix,
)
from .oauth import fetch_user_email, refresh_access_token
from .openai_compat import ChatRequest, openai_completion_object, parse_chat_request, to_openai_completion
from .transform import build_generate_content_request

def _is_hard_quota_error(exc: AntigravityError) -> bool:
    if exc.status != 429:
        return False
    text = str(exc).lower()
    if (
        "individual quota reached" in text
        or "quota reached" in text
        or "resets in " in text
        or "reset in " in text
    ):
        return True
    if "rate limit" in text or "rate-limit" in text:
        return False
    return any(
        marker in text
        for marker in (
            "quota exceeded",
            "exceeded your",
            "daily limit",
            "limit reached",
            "reached your",
        )
    )


def _generate_with_transient_retry(
    client: AntigravityClient,
    *,
    access_token: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    last_error: AntigravityError | None = None
    for attempt in range(3):
        try:
            return client.generate(access_token=access_token, body=body)
        except AntigravityError as exc:
            last_error = exc
            if exc.status != 429 or _is_hard_quota_error(exc) or attempt >= 2:
                raise
            time.sleep(0.5 * (2 ** attempt))
    if last_error:
        raise last_error
    raise AntigravityError("Antigravity request failed without a response", status=502)


def _needs_refresh(credentials: dict[str, Any]) -> bool:
    access = credentials.get("access_token") or credentials.get("access") or credentials.get("token")
    refresh = credentials.get("refresh_token") or credentials.get("refresh")
    expires = credentials.get("expires_at") or credentials.get("expires")
    if not access:
        return bool(refresh)
    return bool(refresh and isinstance(expires, (int, float)) and time.time() + 60 >= float(expires))


def _prepare_credentials(
    credentials: dict[str, Any],
    *,
    store: CredentialStore,
    persist: bool,
) -> dict[str, Any]:
    creds = dict(credentials)
    refresh = creds.get("refresh_token") or creds.get("refresh")
    if _needs_refresh(creds):
        if not refresh:
            raise AntigravityError("Antigravity access token expired and no refresh token is available", status=401)
        creds.update(refresh_access_token(str(refresh)))

    access = creds.get("access_token") or creds.get("access") or creds.get("token")
    if not access:
        raise AntigravityError("Antigravity credential does not contain an access token", status=401)

    if not creds.get("email"):
        email = fetch_user_email(str(access))
        if email:
            creds["email"] = email
    if not (creds.get("project_id") or creds.get("projectId")):
        creds["project_id"] = load_or_onboard_project(
            str(access),
            seed=str(creds.get("email") or "antigravity-default"),
        )

    if persist:
        store.upsert(creds, activate=False)
    return creds


def _credential_candidates(store: CredentialStore) -> list[tuple[str, dict[str, Any], bool]]:
    candidates: list[tuple[str, dict[str, Any], bool]] = []
    seen_tokens: set[str] = set()

    # The profile-local active account must win. Keychain is a compatibility
    # fallback, otherwise "hermes agy use" would be ignored on macOS.
    for key, creds in store.ordered_credentials():
        token = str(creds.get("refresh_token") or creds.get("access_token") or "")
        if token and token in seen_tokens:
            continue
        if token:
            seen_tokens.add(token)
        candidates.append((key, creds, True))

    keychain = load_agy_keychain_credentials()
    if keychain:
        token = str(keychain.get("refresh_token") or keychain.get("access_token") or "")
        if not token or token not in seen_tokens:
            candidates.append(("agy-keychain", keychain, False))

    return candidates


def _catalog_runtime(
    catalog: dict[str, Any],
    logical_model: str,
    effort: str | None,
) -> str | None:
    if not catalog:
        return None
    logical = strip_provider_prefix(logical_model)
    level = normalize_effort(effort)
    requested = resolve_wire_model_id(logical, level)
    if requested in catalog:
        return requested

    bases = [logical]
    if logical == "gemini-3.8-flash":
        bases.extend(["gemini-3.7-flash", "gemini-3.6-flash"])
    elif logical == "gemini-3.7-flash":
        bases.append("gemini-3.6-flash")

    for base in bases:
        runtime_ids: list[str] = []
        if base.startswith("gemini-3.") and "flash" in base:
            suffix = "high" if level == "high" else "medium" if level == "medium" else "low"
            runtime_ids.extend([f"{base}-{suffix}", f"{base}-tiered", base])
        else:
            runtime_ids.append(resolve_wire_model_id(base, level))
            runtime_ids.append(base)
        for runtime_id in runtime_ids:
            if runtime_id in catalog:
                return runtime_id

    normalized_base = logical.replace("-", " ").lower()
    level_word = "high" if level == "high" else "medium" if level == "medium" else "low"
    for runtime_id, info in catalog.items():
        label = ""
        if isinstance(info, dict):
            label = str(info.get("displayName") or info.get("label") or info.get("name") or "")
        haystack = f"{runtime_id} {label}".replace("-", " ").lower()
        if normalized_base in haystack and (level_word in haystack or "thinking" in haystack):
            return str(runtime_id)
    return None


def _runtime_candidates(
    request: ChatRequest,
    catalog: dict[str, Any],
) -> list[str]:
    static_runtime = resolve_wire_model_id(request.model, request.reasoning_effort)
    dynamic_runtime = _catalog_runtime(catalog, request.model, request.reasoning_effort)

    candidates: list[str] = []
    ordered_initial = (
        (static_runtime, dynamic_runtime)
        if request.model in KNOWN_MODELS
        else (dynamic_runtime, static_runtime)
    )
    for runtime_id in ordered_initial:
        if runtime_id and runtime_id not in candidates:
            candidates.append(runtime_id)

    current = static_runtime
    for _ in range(3):
        fallback = get_fallback_runtime_model(current, request.reasoning_effort)
        if not fallback:
            break
        if fallback not in candidates:
            candidates.append(fallback)
        current = fallback
    return candidates


def _access_token(credentials: dict[str, Any]) -> str:
    value = credentials.get("access_token") or credentials.get("access") or credentials.get("token")
    if not value:
        raise AntigravityError("Missing Antigravity access token", status=401)
    return str(value)


def _project_id(credentials: dict[str, Any]) -> str:
    value = credentials.get("project_id") or credentials.get("projectId")
    if not value:
        raise AntigravityError("Missing Antigravity project id", status=401)
    return str(value)


def build_upstream_body(
    request: ChatRequest,
    *,
    credentials: dict[str, Any],
    runtime_model: str,
) -> dict[str, Any]:
    return build_generate_content_request(
        model=request.model,
        project_id=_project_id(credentials),
        messages=request.messages,
        tools=request.tools,
        reasoning_effort=request.reasoning_effort,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        tool_choice=request.tool_choice,
        runtime_model_override=runtime_model,
        model_enum_override=get_model_enum(runtime_model),
    )


def generate_chat_completion(
    payload: dict[str, Any],
    *,
    client: AntigravityClient | None = None,
    store: CredentialStore | None = None,
) -> dict[str, Any]:
    request = parse_chat_request(payload)
    client = client or AntigravityClient()
    store = store or CredentialStore.default()

    candidates = _credential_candidates(store)
    if not candidates:
        raise AntigravityError(
            "Missing Antigravity credentials. Run hermes agy login.",
            status=401,
            error_type="invalid_request_error",
        )

    last_error: Exception | None = None

    for account_key, raw_credentials, persist in candidates:
        try:
            credentials = _prepare_credentials(raw_credentials, store=store, persist=persist)
        except Exception as exc:
            last_error = exc
            continue

        access_token = _access_token(credentials)
        project_id = _project_id(credentials)
        is_known_model = request.model in KNOWN_MODELS
        catalog: dict[str, Any] = {}
        discovery_attempted = False

        # Known models use their verified static route immediately for lower
        # time-to-first-token. Unknown/future models need discovery up front.
        if not is_known_model:
            discovery_attempted = True
            try:
                catalog = fetch_available_models(access_token, project_id)
            except Exception:
                catalog = {}

        runtimes = _runtime_candidates(request, catalog)
        refreshed_once = False
        runtime_index = 0

        def append_dynamic_runtime_if_needed() -> None:
            nonlocal catalog, discovery_attempted
            if discovery_attempted:
                return
            discovery_attempted = True
            try:
                catalog = fetch_available_models(access_token, project_id)
            except Exception:
                catalog = {}
            dynamic_runtime = _catalog_runtime(
                catalog,
                request.model,
                request.reasoning_effort,
            )
            if dynamic_runtime and dynamic_runtime not in runtimes:
                runtimes.append(dynamic_runtime)

        while runtime_index < len(runtimes):
            runtime_model = runtimes[runtime_index]
            runtime_index += 1
            body = build_upstream_body(
                request,
                credentials=credentials,
                runtime_model=runtime_model,
            )
            try:
                upstream = _generate_with_transient_retry(client, access_token=access_token, body=body)
                if persist:
                    store.upsert(credentials, activate=True)
                return to_openai_completion(request.model, upstream)
            except TokenExpired as exc:
                last_error = exc
                refresh = credentials.get("refresh_token") or credentials.get("refresh")
                if refreshed_once or not refresh:
                    break
                refreshed_once = True
                credentials.update(refresh_access_token(str(refresh)))
                access_token = _access_token(credentials)
                if persist:
                    store.upsert(credentials, activate=True)
                body = build_upstream_body(
                    request,
                    credentials=credentials,
                    runtime_model=runtime_model,
                )
                try:
                    upstream = _generate_with_transient_retry(client, access_token=access_token, body=body)
                    return to_openai_completion(request.model, upstream)
                except Exception as retry_exc:
                    last_error = retry_exc
                    if isinstance(retry_exc, AntigravityError) and retry_exc.status == 404:
                        if runtime_index >= len(runtimes):
                            append_dynamic_runtime_if_needed()
                        continue
                    if isinstance(retry_exc, AntigravityError) and _is_hard_quota_error(retry_exc):
                        break
                    raise
            except AntigravityError as exc:
                last_error = exc
                if exc.status == 404:
                    # Known routes intentionally skip discovery on the fast
                    # path. Only after static candidates fail do we ask the
                    # account catalog for rollout aliases/tiered runtimes.
                    if runtime_index >= len(runtimes):
                        append_dynamic_runtime_if_needed()
                    continue
                if _is_hard_quota_error(exc):
                    break
                raise

    if last_error:
        raise last_error
    raise AntigravityError("No usable Antigravity account or runtime model was available", status=503)


def get_live_catalog(
    *,
    store: CredentialStore | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    store = store or CredentialStore.default()
    candidates = _credential_candidates(store)
    if not candidates:
        return {}, {}
    _, raw, persist = candidates[0]
    credentials = _prepare_credentials(raw, store=store, persist=persist)
    catalog = fetch_available_models(
        _access_token(credentials),
        _project_id(credentials),
        force=force,
    )
    return catalog, credentials


def get_available_model_ids(*, store: CredentialStore | None = None) -> list[str]:
    try:
        catalog, _ = get_live_catalog(store=store)
    except Exception:
        catalog = {}
    return public_models_from_catalog(catalog)


def get_quota(*, store: CredentialStore | None = None) -> dict[str, Any]:
    try:
        _, credentials = get_live_catalog(store=store)
        if not credentials:
            return {}
        return fetch_quota_summary(_access_token(credentials), _project_id(credentials))
    except Exception:
        return {}


def ensure_provider_profile_files(root: Path | None = None) -> Path:
    if root is None:
        try:
            from hermes_constants import get_hermes_home

            root = get_hermes_home()
        except Exception:
            root = Path.home() / ".hermes"

    plugin_dir = Path(root).expanduser() / "plugins" / "model-providers" / "antigravity"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent.parent
    shim = (
        "import sys\n"
        f"_SOURCE_ROOT = {str(source_root)!r}\n"
        "if _SOURCE_ROOT not in sys.path:\n"
        "    sys.path.insert(0, _SOURCE_ROOT)\n"
        "from hermes_antigravity.hermes_provider import register_provider_profile\n"
        "register_provider_profile()\n"
    )
    (plugin_dir / "__init__.py").write_text(shim, encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(
        "name: antigravity\n"
        "kind: model-provider\n"
        "version: 1.0.0\n"
        "description: Google Antigravity provider supplied by hermes-antigravity\n",
        encoding="utf-8",
    )
    return plugin_dir


__all__ = [
    "DEFAULT_MODEL",
    "generate_chat_completion",
    "get_available_model_ids",
    "get_live_catalog",
    "get_quota",
    "ensure_provider_profile_files",
    "openai_completion_object",
]

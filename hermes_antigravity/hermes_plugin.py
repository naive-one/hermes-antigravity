from __future__ import annotations

import argparse
import json
from typing import Any

from .credentials import CredentialStore
from .hermes_provider import (
    DEFAULT_MODEL,
    PLACEHOLDER_API_KEY,
    PLACEHOLDER_API_KEY_ENV,
    PROVIDER_NAME,
    register_provider_profile,
)
from .models import KNOWN_MODELS, normalize_model_id
from .runtime import (
    ensure_provider_profile_files,
    generate_chat_completion,
    get_available_model_ids,
    get_live_catalog,
    get_quota,
    openai_completion_object,
)


def _is_antigravity_request(provider: str | None, request: dict[str, Any]) -> bool:
    del request
    return (provider or "").strip().lower() in {
        PROVIDER_NAME,
        "google-antigravity",
        "agy",
    }


def _error_message(exc: Exception) -> str:
    message = " ".join(str(exc).split()) or type(exc).__name__
    return f"Antigravity request failed: {message}"


def antigravity_llm_execution(**kwargs: Any) -> Any:
    request = kwargs.get("request") or {}
    next_call = kwargs.get("next_call")
    if not _is_antigravity_request(kwargs.get("provider"), request):
        return next_call(request) if callable(next_call) else request

    try:
        completion = generate_chat_completion(request)
    except Exception as exc:
        completion = {
            "model": str(request.get("model") or DEFAULT_MODEL),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": _error_message(exc),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    return openai_completion_object(completion)


def _save_placeholder_api_key() -> None:
    """Satisfy Hermes' generic api_key provider gate.

    The value never leaves Hermes: llm_execution is short-circuited and the
    real Google OAuth token is loaded from hermes-antigravity's account store.
    """
    try:
        from hermes_cli.config import get_env_value, save_env_value

        if not (get_env_value(PLACEHOLDER_API_KEY_ENV) or "").strip():
            save_env_value(PLACEHOLDER_API_KEY_ENV, PLACEHOLDER_API_KEY)
    except Exception:
        return


def _setup_cli(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="antigravity_command")

    login = sub.add_parser("login", help="add or refresh a Google Antigravity account")
    login.add_argument("--no-keychain", action="store_true", help="skip macOS agy Keychain import")
    login.add_argument("--no-browser", action="store_true", help="print the OAuth URL instead of opening it")
    login.add_argument(
        "--manual",
        action="store_true",
        help="VPS/headless mode: paste the localhost callback URL after browser authorization",
    )
    login.add_argument("--timeout", type=int, default=300, help="seconds to wait for the OAuth callback")

    sub.add_parser("status", help="show Antigravity account and catalog status")
    sub.add_parser("accounts", help="list saved Google accounts")

    use = sub.add_parser("use", help="make a saved account active")
    use.add_argument("account", help="account key or email shown by 'hermes agy accounts'")

    remove = sub.add_parser("remove-account", help="remove one saved account")
    remove.add_argument("account", help="account key shown by 'hermes agy accounts'")

    models = sub.add_parser("models", help="show live Antigravity models for the active account")
    models.add_argument("--refresh", action="store_true", help="ignore the model-catalog cache")

    sub.add_parser("quota", help="show the raw Antigravity quota summary")

    select = sub.add_parser("select", help="set Antigravity as the active Hermes provider/model")
    select.add_argument("model", nargs="?", default=DEFAULT_MODEL)

    logout = sub.add_parser("logout", help="remove saved browser OAuth accounts")
    logout.add_argument(
        "--all",
        action="store_true",
        help="remove all locally saved Antigravity accounts (default behavior currently does the same)",
    )


def _select_model(model_id: str) -> None:
    model_id = normalize_model_id(model_id)
    try:
        from hermes_cli.config import load_config, save_config
    except Exception as exc:
        raise SystemExit(f"Hermes config helpers are not available: {exc}") from exc

    _save_placeholder_api_key()
    ensure_provider_profile_files()
    register_provider_profile()

    config = load_config()
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {"default": model_cfg} if model_cfg else {}

    model_cfg["provider"] = PROVIDER_NAME
    model_cfg["default"] = model_id
    model_cfg["base_url"] = "http://127.0.0.1:8765/v1"
    model_cfg["api_mode"] = "chat_completions"
    config["model"] = model_cfg
    save_config(config)
    print(f"Default Hermes model set to {model_id} via provider '{PROVIDER_NAME}'.")


def _print_accounts(store: CredentialStore) -> None:
    rows = store.list_accounts()
    if not rows:
        print("No saved Antigravity accounts.")
        return
    for key, creds, active in rows:
        marker = "*" if active else " "
        email = creds.get("email") or key
        project = creds.get("project_id") or creds.get("projectId") or "-"
        print(f"{marker} {key}  {email}  project={project}")


def _status() -> None:
    store = CredentialStore.default()
    rows = store.list_accounts()
    print(f"saved accounts: {len(rows)}")
    if rows:
        _print_accounts(store)
    try:
        catalog, credentials = get_live_catalog(store=store)
    except Exception as exc:
        print(f"live catalog: unavailable ({exc})")
        return
    if credentials:
        print(f"active account: {credentials.get('email') or 'unknown'}")
    print(f"live runtime models: {len(catalog)}")
    print(f"Hermes public models: {len(get_available_model_ids(store=store))}")


def _print_models(refresh: bool) -> None:
    try:
        catalog, credentials = get_live_catalog(force=refresh)
    except Exception as exc:
        print(f"Live model discovery failed: {exc}")
        print("Static fallback models:")
        for model in KNOWN_MODELS:
            print(f"  {model}")
        return

    if credentials.get("email"):
        print(f"account: {credentials['email']}")
    public = get_available_model_ids()
    print("Hermes models:")
    for model in public:
        print(f"  {model}")
    if catalog:
        print("\nAntigravity runtime ids:")
        for runtime_id in sorted(catalog):
            info = catalog[runtime_id]
            label = ""
            if isinstance(info, dict):
                label = str(info.get("displayName") or info.get("name") or "")
            suffix = f"  # {label}" if label else ""
            print(f"  {runtime_id}{suffix}")


def _print_quota() -> None:
    payload = get_quota()
    if not payload:
        print("Quota information is unavailable for the active account.")
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _handle_cli(args: argparse.Namespace) -> None:
    command = getattr(args, "antigravity_command", None) or "status"
    store = CredentialStore.default()

    if command == "login":
        from .oauth import run_login

        ensure_provider_profile_files()
        register_provider_profile()
        _save_placeholder_api_key()
        credentials = run_login(
            open_browser=not args.no_browser,
            timeout=args.timeout,
            store=store,
            prefer_keychain=not args.no_keychain,
            manual=args.manual,
        )
        if credentials.get("email"):
            print(f"Antigravity login complete: {credentials['email']}")
        else:
            print("Antigravity login complete.")
        return

    if command == "accounts":
        _print_accounts(store)
        return

    if command == "use":
        if not store.activate(args.account):
            raise SystemExit(f"Antigravity account not found or ambiguous: {args.account}")
        print(f"Active Antigravity account: {args.account}")
        return

    if command == "remove-account":
        if not store.remove(args.account):
            raise SystemExit(f"Antigravity account not found: {args.account}")
        print(f"Removed Antigravity account: {args.account}")
        return

    if command == "models":
        _print_models(bool(args.refresh))
        return

    if command == "quota":
        _print_quota()
        return

    if command == "select":
        _select_model(args.model)
        return

    if command == "logout":
        store.delete()
        print("Saved Antigravity browser OAuth accounts removed.")
        return

    _status()


def register(ctx: Any) -> None:
    # Register in-process for this Hermes session and materialize a tiny
    # model-provider shim so the provider is discoverable on the next startup.
    register_provider_profile()
    ensure_provider_profile_files()
    _save_placeholder_api_key()

    ctx.register_cli_command(
        name="agy",
        help="Manage the Google Antigravity provider",
        description="OAuth, accounts, models, quota, and model selection for Google Antigravity.",
        setup_fn=_setup_cli,
        handler_fn=_handle_cli,
    )
    ctx.register_middleware("llm_execution", antigravity_llm_execution)

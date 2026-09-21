from __future__ import annotations

from typing import Any

from .models import (
    DEFAULT_MODEL,
    KNOWN_MODELS,
    clamp_reasoning_effort,
    strip_provider_prefix,
)
from .runtime import get_available_model_ids

PROVIDER_NAME = "antigravity"
PLACEHOLDER_API_KEY_ENV = "ANTIGRAVITY_HERMES_API_KEY"
PLACEHOLDER_API_KEY = "hermes-antigravity"
DUMMY_BASE_URL = "http://127.0.0.1:8765/v1"


def _reasoning_effort(reasoning_config: dict | None, model: str | None = None) -> str | None:
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return "off"
    effort = str(reasoning_config.get("effort") or "").strip().lower().replace("_", "-")
    if not effort:
        return None
    return clamp_reasoning_effort(model or DEFAULT_MODEL, effort)


def _efforts_for(model: str | None) -> tuple[str, ...] | None:
    logical = strip_provider_prefix(model or DEFAULT_MODEL)
    if logical in {"gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"}:
        return ("low", "medium", "high")
    if logical == "gemini-3.1-pro":
        return ("low", "high")
    if logical in {"claude-sonnet-4-6", "claude-opus-4-6"}:
        return ("high",)
    if logical == "gpt-oss-120b":
        return ("medium",)
    return None


def register_provider_profile() -> bool:
    """Register the Hermes-visible Antigravity model provider."""
    try:
        from providers import register_provider
        from providers.base import OMIT_TEMPERATURE, ProviderProfile
    except Exception:
        return False

    class AntigravityProfile(ProviderProfile):
        def build_api_kwargs_extras(
            self,
            *,
            reasoning_config: dict | None = None,
            **context: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            top_level: dict[str, Any] = {}
            effort = _reasoning_effort(reasoning_config, context.get("model"))
            if effort:
                top_level["reasoning_effort"] = effort
            return {}, top_level

        def get_max_tokens(self, model: str | None) -> int | None:
            return KNOWN_MODELS.get(model or "") or KNOWN_MODELS.get(DEFAULT_MODEL)

        def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
            return _efforts_for(model)

        def fetch_models(
            self,
            *,
            api_key: str | None = None,
            base_url: str | None = None,
            timeout: float = 8.0,
        ) -> list[str] | None:
            del api_key, base_url, timeout
            return get_available_model_ids()

        def default_vision_model(self) -> str | None:
            return DEFAULT_MODEL

    register_provider(
        AntigravityProfile(
            name=PROVIDER_NAME,
            aliases=("google-antigravity", "agy"),
            display_name="Google Antigravity",
            description="Google Antigravity / Cloud Code Assist via hermes-antigravity",
            env_vars=(PLACEHOLDER_API_KEY_ENV,),
            base_url=DUMMY_BASE_URL,
            auth_type="api_key",
            supports_health_check=False,
            supports_model_listing=True,
            supports_vision=True,
            fallback_models=tuple(KNOWN_MODELS),
            default_aux_model=DEFAULT_MODEL,
            fixed_temperature=OMIT_TEMPERATURE,
        )
    )
    return True


register_provider_profile()

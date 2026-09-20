from __future__ import annotations

from typing import Any

ANTIGRAVITY_PREFIX = "google-antigravity/"
DEFAULT_MODEL = f"{ANTIGRAVITY_PREFIX}gemini-3.8-flash"

# Public Hermes model ids -> maximum output tokens.
KNOWN_MODELS: dict[str, int] = {
    f"{ANTIGRAVITY_PREFIX}gemini-3.8-flash": 65536,
    f"{ANTIGRAVITY_PREFIX}gemini-3.7-flash": 65536,
    f"{ANTIGRAVITY_PREFIX}gemini-3.6-flash": 65536,
    f"{ANTIGRAVITY_PREFIX}gemini-3.5-flash": 65536,
    f"{ANTIGRAVITY_PREFIX}gemini-3.1-pro": 65535,
    f"{ANTIGRAVITY_PREFIX}claude-sonnet-4-6": 64000,
    f"{ANTIGRAVITY_PREFIX}claude-opus-4-6": 64000,
    f"{ANTIGRAVITY_PREFIX}gpt-oss-120b": 32768,
}

ROUTING: dict[str, dict[str, str]] = {
    "gemini-3.8-flash": {
        "off": "gemini-3.8-flash-low",
        "low": "gemini-3.8-flash-low",
        "medium": "gemini-3.8-flash-medium",
        "high": "gemini-3.8-flash-high",
    },
    "gemini-3.7-flash": {
        "off": "gemini-3.7-flash-low",
        "low": "gemini-3.7-flash-low",
        "medium": "gemini-3.7-flash-medium",
        "high": "gemini-3.7-flash-high",
    },
    "gemini-3.6-flash": {
        "off": "gemini-3.6-flash-low",
        "low": "gemini-3.6-flash-low",
        "medium": "gemini-3.6-flash-medium",
        "high": "gemini-3.6-flash-high",
    },
    "gemini-3.5-flash": {
        "off": "gemini-3.5-flash-extra-low",
        "low": "gemini-3.5-flash-extra-low",
        "medium": "gemini-3.5-flash-low",
        "high": "gemini-3-flash-agent",
    },
    "gemini-3.1-pro": {
        "off": "gemini-3.1-pro-low",
        "low": "gemini-3.1-pro-low",
        "medium": "gemini-3.1-pro-low",
        "high": "gemini-pro-agent",
    },
    "claude-sonnet-4-6": {
        "off": "claude-sonnet-4-6",
        "low": "claude-sonnet-4-6",
        "medium": "claude-sonnet-4-6",
        "high": "claude-sonnet-4-6",
    },
    "claude-opus-4-6": {
        "off": "claude-opus-4-6-thinking",
        "low": "claude-opus-4-6-thinking",
        "medium": "claude-opus-4-6-thinking",
        "high": "claude-opus-4-6-thinking",
    },
    "gpt-oss-120b": {
        "off": "gpt-oss-120b-medium",
        "low": "gpt-oss-120b-medium",
        "medium": "gpt-oss-120b-medium",
        "high": "gpt-oss-120b-medium",
    },
}

RUNTIME_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "gemini-3.8-flash": 65536,
    "gemini-3.8-flash-low": 65536,
    "gemini-3.8-flash-medium": 65536,
    "gemini-3.8-flash-high": 65536,
    "gemini-3.7-flash": 65536,
    "gemini-3.7-flash-tiered": 65536,
    "gemini-3.7-flash-low": 65536,
    "gemini-3.7-flash-medium": 65536,
    "gemini-3.7-flash-high": 65536,
    "gemini-3.6-flash": 65536,
    "gemini-3.6-flash-tiered": 65536,
    "gemini-3.6-flash-low": 65536,
    "gemini-3.6-flash-medium": 65536,
    "gemini-3.6-flash-high": 65536,
    "gemini-3.5-flash": 65536,
    "gemini-3.5-flash-extra-low": 65536,
    "gemini-3.5-flash-low": 65536,
    "gemini-3-flash-agent": 65536,
    "gemini-3.1-pro": 65535,
    "gemini-3.1-pro-low": 65535,
    "gemini-3.1-pro-high": 65535,
    "gemini-pro-agent": 65535,
    "claude-sonnet-4-6": 64000,
    "claude-opus-4-6": 64000,
    "claude-opus-4-6-thinking": 64000,
    "gpt-oss-120b": 32768,
    "gpt-oss-120b-medium": 32768,
    "openai/gpt-oss-120b-maas": 32768,
}

# Static fallbacks copied from the current pi-antigravity catalog. Live
# fetchAvailableModels values always take precedence.
MODEL_ENUM_FALLBACKS: dict[str, str] = {
    "gemini-3.8-flash": "MODEL_PLACEHOLDER_M318",
    "gemini-3.8-flash-high": "MODEL_PLACEHOLDER_M318",
    "gemini-3.8-flash-medium": "MODEL_PLACEHOLDER_M319",
    "gemini-3.8-flash-low": "MODEL_PLACEHOLDER_M320",
    "gemini-3.8-flash-tiered": "MODEL_PLACEHOLDER_M322",
    "gemini-3.7-flash": "MODEL_PLACEHOLDER_M298",
    "gemini-3.7-flash-high": "MODEL_PLACEHOLDER_M298",
    "gemini-3.7-flash-medium": "MODEL_PLACEHOLDER_M299",
    "gemini-3.7-flash-low": "MODEL_PLACEHOLDER_M300",
    "gemini-3.7-flash-tiered": "MODEL_PLACEHOLDER_M301",
    "gemini-3.6-flash": "MODEL_PLACEHOLDER_M71",
    "gemini-3.6-flash-high": "MODEL_PLACEHOLDER_M71",
    "gemini-3.6-flash-medium": "MODEL_PLACEHOLDER_M72",
    "gemini-3.6-flash-low": "MODEL_PLACEHOLDER_M73",
    "gemini-3.6-flash-tiered": "MODEL_PLACEHOLDER_M196",
    "gemini-3.5-flash": "MODEL_PLACEHOLDER_M20",
    "gemini-3.5-flash-extra-low": "MODEL_PLACEHOLDER_M187",
    "gemini-3.5-flash-low": "MODEL_PLACEHOLDER_M20",
    "gemini-3-flash-agent": "MODEL_PLACEHOLDER_M84",
    "gemini-3.1-pro": "MODEL_PLACEHOLDER_M36",
    "gemini-3.1-pro-low": "MODEL_PLACEHOLDER_M36",
    "gemini-3.1-pro-high": "MODEL_PLACEHOLDER_M37",
    "gemini-pro-agent": "MODEL_PLACEHOLDER_M16",
    "claude-sonnet-4-6": "MODEL_PLACEHOLDER_M35",
    "claude-opus-4-6": "MODEL_PLACEHOLDER_M26",
    "claude-opus-4-6-thinking": "MODEL_PLACEHOLDER_M26",
    "gpt-oss-120b": "MODEL_OPENAI_GPT_OSS_120B_MEDIUM",
    "gpt-oss-120b-medium": "MODEL_OPENAI_GPT_OSS_120B_MEDIUM",
    "openai/gpt-oss-120b-maas": "MODEL_OPENAI_GPT_OSS_120B_MEDIUM",
}

_DYNAMIC_MODEL_ENUMS: dict[str, str] = {}


def strip_provider_prefix(model: str) -> str:
    model = (model or "").strip()
    return model[len(ANTIGRAVITY_PREFIX):] if model.startswith(ANTIGRAVITY_PREFIX) else model


def normalize_model_id(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return DEFAULT_MODEL
    if "/" not in model:
        return f"{ANTIGRAVITY_PREFIX}{model}"
    return model


def normalize_effort(value: str | None) -> str:
    effort = (value or "low").strip().lower().replace("_", "-")
    if effort in {"off", "none", "disabled"}:
        return "off"
    if effort in {"minimum", "minimal"}:
        return "low"
    if effort in {"normal", "medium"}:
        return "medium"
    if effort in {"high", "xhigh", "max", "extra-high"}:
        return "high"
    return "low"


def clamp_reasoning_effort(model: str, effort: str | None = None) -> str:
    logical = strip_provider_prefix(normalize_model_id(model))
    level = normalize_effort(effort)
    if level == "off":
        return "off"
    if logical == "gemini-3.1-pro":
        return "high" if level == "high" else "low"
    if logical == "gpt-oss-120b":
        return "medium"
    return level


def resolve_wire_model_id(model: str, effort: str | None = None) -> str:
    logical = strip_provider_prefix(normalize_model_id(model))
    level = clamp_reasoning_effort(model, effort)
    route = ROUTING.get(logical)
    if not route:
        return logical
    return route.get(level) or route.get("low") or logical


def get_max_output_tokens(model: str, runtime_model: str | None = None) -> int:
    if runtime_model and runtime_model in RUNTIME_MAX_OUTPUT_TOKENS:
        return RUNTIME_MAX_OUTPUT_TOKENS[runtime_model]
    logical = strip_provider_prefix(normalize_model_id(model))
    return KNOWN_MODELS.get(f"{ANTIGRAVITY_PREFIX}{logical}", 8192)


def get_fallback_runtime_model(runtime_model: str, effort: str | None = None) -> str | None:
    if runtime_model.startswith("gemini-3.8-flash-"):
        return runtime_model.replace("gemini-3.8-flash-", "gemini-3.7-flash-", 1)
    if runtime_model == "gemini-3.8-flash":
        return resolve_wire_model_id("gemini-3.7-flash", effort)
    if runtime_model == "gemini-3.7-flash-tiered":
        return resolve_wire_model_id("gemini-3.6-flash", effort)
    if runtime_model.startswith("gemini-3.7-flash-"):
        return runtime_model.replace("gemini-3.7-flash-", "gemini-3.6-flash-", 1)
    if runtime_model == "gemini-3.7-flash":
        return resolve_wire_model_id("gemini-3.6-flash", effort)
    return None


def register_discovered_model_enums(models: dict[str, Any] | None) -> None:
    if not isinstance(models, dict):
        return
    for runtime_id, info in models.items():
        if isinstance(info, dict) and isinstance(info.get("model"), str) and info["model"]:
            _DYNAMIC_MODEL_ENUMS[str(runtime_id)] = info["model"]


def get_model_enum(runtime_model: str) -> str | None:
    return _DYNAMIC_MODEL_ENUMS.get(runtime_model) or MODEL_ENUM_FALLBACKS.get(runtime_model)


def clear_model_enum_cache() -> None:
    _DYNAMIC_MODEL_ENUMS.clear()


def public_model_from_runtime(runtime_id: str) -> str | None:
    rid = (runtime_id or "").strip()
    aliases = {
        "gemini-3-flash-agent": "gemini-3.5-flash",
        "gemini-pro-agent": "gemini-3.1-pro",
        "claude-opus-4-6-thinking": "claude-opus-4-6",
        "gpt-oss-120b-medium": "gpt-oss-120b",
        "openai/gpt-oss-120b-maas": "gpt-oss-120b",
    }
    if rid in aliases:
        return aliases[rid]
    for suffix in ("-extra-low", "-extra-high", "-minimal", "-medium", "-high", "-low", "-thinking", "-tiered"):
        if rid.endswith(suffix):
            base = rid[:-len(suffix)]
            if f"{ANTIGRAVITY_PREFIX}{base}" in KNOWN_MODELS:
                return base
    if f"{ANTIGRAVITY_PREFIX}{rid}" in KNOWN_MODELS:
        return rid
    return None


def public_models_from_catalog(models: dict[str, Any] | None) -> list[str]:
    if not isinstance(models, dict):
        return list(KNOWN_MODELS)
    seen: set[str] = set()
    for runtime_id in models:
        public = public_model_from_runtime(str(runtime_id))
        if public:
            seen.add(f"{ANTIGRAVITY_PREFIX}{public}")
    # Conservative fallbacks remain selectable even when a rollout-specific
    # authenticated catalog omits a family.
    seen.update(KNOWN_MODELS)
    return [model for model in KNOWN_MODELS if model in seen]

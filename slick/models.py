from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import os
 


 # legacy function-based provider helpers removed; now using class-based providers

# Provider class-based implementation (preferred)
def _load_provider(name: str):
    # Lazy import providers to avoid importing optional SDKs at CLI/help time
    if name == "openai":
        from .providers.openai import OpenAIProvider

        return OpenAIProvider
    if name == "anthropic":
        from .providers.anthropic import AnthropicProvider

        return AnthropicProvider
    if name == "google":
        from .providers.google import GoogleProvider

        return GoogleProvider
    if name == "mistral":
        from .providers.mistral import MistralProvider

        return MistralProvider
    if name == "groq":
        from .providers.groq import GroqProvider

        return GroqProvider
    if name == "together":
        from .providers.together import TogetherProvider

        return TogetherProvider
    if name == "fireworks":
        from .providers.fireworks import FireworksProvider

        return FireworksProvider
    raise ValueError(f"Unknown provider: {name}")


PROVIDERS = {name: name for name in ["openai", "anthropic", "google", "mistral", "groq", "together", "fireworks"]}

# In-memory defaults (overridden by resolution logic)
_default_model: Optional[str] = None
_default_provider: Optional[str] = None


def set_default_model(model: str, provider: Optional[str] = None) -> None:
    """Set the global default model and optional provider in-memory.

    TODO: Persist to user config (e.g., ~/.config/slick/config.toml) via a config helper.
    """

    global _default_model, _default_provider
    _default_model = model
    _default_provider = provider


def get_default_model() -> Tuple[Optional[str], Optional[str]]:
    """Return the currently resolved default (in-memory only for now).

    TODO: Load from env, user config, project config if in-memory is not set.
    """

    return _default_model, _default_provider


def list_providers() -> List[str]:
    """Return provider keys supported by this module."""

    return list(PROVIDERS.keys())


def list_models(provider: str) -> List[str]:
    """List models for a given provider (best-effort).

    This aims to call the provider SDK's list models endpoint if available.
    For example (to be implemented):
    - openai: client.models.list()
    - anthropic: client.models.list() (if available)
    - google: genai.list_models()
    - mistral: client.models.list()
    - cohere: cohere.Client(...).models.list()
    - groq: client.models.list()

    TODO: Implement per-provider listing using their official SDKs, with graceful fallbacks.
    """

    provider_key = PROVIDERS.get(provider)
    if not provider_key:
        raise ValueError(f"Unknown provider: {provider}")
    try:
        provider_cls = _load_provider(provider_key)
        return provider_cls.list_models()
    except Exception:
        # Keep errors non-fatal for listing. Callers can inspect logs later.
        return []


def is_available(provider: str) -> bool:
    """Check if a provider is 'available' (package importable and env keys present).

    TODO: Implement dynamic import check + env key presence.
    """

    return True


def _dynamic_import(dotted_path: str) -> Any:
    """Import a symbol by dotted path (kept for future use)."""

    module_name, _, attr = dotted_path.rpartition(".")
    if not module_name:
        return __import__(dotted_path)
    mod = __import__(module_name, fromlist=[attr])
    return getattr(mod, attr)


def create_chat_model(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create and return a LangChain chat model instance for the resolved provider+model.

    Resolution order (highest first):
    1) Explicit args (model/provider)
    2) In-memory defaults (set_default_model)
    3) Env vars (SLICK_MODEL, SLICK_PROVIDER)
    4) User config (~/.config/slick/config.toml, or macOS Application Support path)
    5) Project config (pyproject.toml [tool.slick])
    6) Built-in default (openai:gpt-4o-mini)

    TODO:
    - Implement the full resolution order.
    - Resolve provider from registry if only model provided.
    - Dynamic import of provider class; instantiate with kwargs and API key from env.
    - Support Azure deployment nuances (deployment name vs model).
    - Respect 'streaming' flag if passed; allow callbacks passthrough.
    """

    # Resolve provider/model
    sel_provider = provider or _default_provider or os.getenv("SLICK_PROVIDER") or "openai"
    sel_model = model or _default_model or os.getenv("SLICK_MODEL") or "gpt-4o-mini"

    provider_key = PROVIDERS.get(sel_provider)
    if not provider_key:
        raise ValueError(f"Unknown provider: {sel_provider}")
    provider_cls = _load_provider(provider_key)
    return provider_cls.make_chat(sel_model, **kwargs)



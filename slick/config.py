from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


def load_env_defaults() -> Tuple[Optional[str], Optional[str]]:
    """Read SLICK_MODEL and SLICK_PROVIDER from environment variables."""

    return os.getenv("SLICK_MODEL"), os.getenv("SLICK_PROVIDER")


def load_user_config() -> Dict[str, Any]:
    """Load user-level config (skeleton).

    Intended locations:
    - Linux: ~/.config/slick/config.toml
    - macOS: ~/Library/Application Support/slick/config.toml
    - Windows: %APPDATA%/slick/config.toml

    TODO: Implement TOML parsing and path resolution; avoid heavy deps.
    """

    return {}


def load_project_config() -> Dict[str, Any]:
    """Load project-level config from pyproject.toml under [tool.slick] (skeleton).

    TODO: Implement TOML parsing and extraction; avoid heavy deps.
    """

    return {}


def resolve_defaults() -> Tuple[Optional[str], Optional[str]]:
    """Apply precedence to derive default model/provider when not set in-memory.

    Order: env > user config > project config > None

    This function intentionally does not supply a built-in default; callers may add one.
    """

    env_model, env_provider = load_env_defaults()
    if env_model or env_provider:
        return env_model, env_provider

    user = load_user_config()
    if "default_model" in user or "default_provider" in user:
        return user.get("default_model"), user.get("default_provider")

    proj = load_project_config()
    if "default_model" in proj or "default_provider" in proj:
        return proj.get("default_model"), proj.get("default_provider")

    return None, None



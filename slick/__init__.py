"""Slick: Jinja templates to LLMs and back.

Use Prompt or render with an ordinary provider.call/acall, then parse when
needed. The existing @prompt decorator remains available as typed
shorthand. CLI and API provider implementations live in slick.providers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .prompts import Prompt, PromptError, parse, prompt, render
from .providers import (
    ExecutionResult,
    Provider,
    ProviderError,
    get_command,
    get_default,
    set_default,
)
from .tools import Tool, ToolError

try:
    __version__ = version("slick-ai")
except PackageNotFoundError:  # running from source without an install
    __version__ = "0.0.0"

__all__ = [
    "ExecutionResult",
    "Prompt",
    "PromptError",
    "Provider",
    "ProviderError",
    "Tool",
    "ToolError",
    "__version__",
    "get_command",
    "get_default",
    "parse",
    "prompt",
    "render",
    "set_default",
]

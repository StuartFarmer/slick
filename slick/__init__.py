"""Slick: Jinja templates to LLMs and back.

Use Prompt or render with an ordinary backend.call/acall, then parse when
needed. The existing @prompt decorator remains available as typed
shorthand. Optional API adapters live in slick.backends. Legacy CLI model
exports remain available here.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .models import (
    ClaudeModel,
    CodexModel,
    ExecutionResult,
    Model,
    ModelError,
    get_default,
    get_model,
    set_default,
)
from .prompts import Prompt, PromptError, parse, prompt, render
from .tools import Tool, ToolError

try:
    __version__ = version("slick-ai")
except PackageNotFoundError:  # running from source without an install
    __version__ = "0.0.0"

__all__ = [
    "ClaudeModel",
    "CodexModel",
    "ExecutionResult",
    "Model",
    "ModelError",
    "Prompt",
    "PromptError",
    "Tool",
    "ToolError",
    "__version__",
    "get_default",
    "get_model",
    "parse",
    "prompt",
    "render",
    "set_default",
]

"""Slick: prompts as typed Python functions, run on CLI-backed models.

    from slick import prompt

    @prompt(template="summarize.md.j2")
    def summarize(document: str, audience: str = "an engineer") -> Summary:
        '''Summarize a document for one audience.'''

The template file supplies the words, the signature supplies the
variables, and the return annotation supplies the output contract. The
decorator lives in `slick.prompts`, the model primitives in
`slick.models`. Both are re-exported here.
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
from .prompts import PromptError, prompt

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
    "PromptError",
    "__version__",
    "get_default",
    "get_model",
    "prompt",
    "set_default",
]

"""Compatibility imports for the callable-tool API.

The implementation lives with the native turn protocol in ``slick.turns.tools``.
"""

from .turns import tools as _implementation
from .turns.tools import Tool, ToolError, prepare_tools

__all__ = ["Tool", "ToolError", "prepare_tools"]


def __getattr__(name):
    """Keep private helper imports working while the implementation lives in turns."""
    return getattr(_implementation, name)

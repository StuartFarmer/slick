"""Compatibility module; Anthropic conversion lives in :mod:`slick.turns._anthropic`."""

from .turns import _anthropic as _implementation


def __getattr__(name):
    return getattr(_implementation, name)

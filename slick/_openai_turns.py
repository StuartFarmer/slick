"""Compatibility module; OpenAI conversion lives in :mod:`slick.turns._openai`."""

from .turns import _openai as _implementation


def __getattr__(name):
    return getattr(_implementation, name)

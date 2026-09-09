"""SDK-shaped response objects used by offline transport boundary tests."""

from copy import deepcopy
from types import SimpleNamespace


def sdk_response(value):
    if isinstance(value, dict):
        return SimpleNamespace(
            **{
                key: deepcopy(item) if key in {"arguments", "input"} else sdk_response(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return [sdk_response(item) for item in value]
    return value

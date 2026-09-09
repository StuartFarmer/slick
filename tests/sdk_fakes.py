"""SDK-shaped response objects used by offline transport boundary tests."""

from types import SimpleNamespace


def _json(value, exclude_none):
    if isinstance(value, SimpleNamespace):
        return {
            key: _json(item, exclude_none)
            for key, item in vars(value).items()
            if not (exclude_none and item is None)
        }
    if isinstance(value, list):
        return [_json(item, exclude_none) for item in value]
    return value


class JsonNamespace(SimpleNamespace):
    def model_dump(self, *, mode="json", exclude_none=False):
        return _json(self, exclude_none)

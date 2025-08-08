from __future__ import annotations

from typing import Any, List

import fireworks  # type: ignore
import langchain_fireworks  # type: ignore

from .base import Provider


class FireworksProvider(Provider):
    name = "fireworks"
    env_key = "FIREWORKS_API_KEY"

    @staticmethod
    def list_models() -> List[str]:
        key = FireworksProvider.get_api_key()
        if not key:
            return []
        data = fireworks.Client(api_key=key).models.list()
        items = data.get("data", []) if isinstance(data, dict) else getattr(data, "data", [])
        return sorted(m.get("id") for m in items)

    @staticmethod
    def make_chat(model: str, **kwargs: Any):
        return langchain_fireworks.ChatFireworks(model=model, api_key=FireworksProvider.get_api_key(), **kwargs)



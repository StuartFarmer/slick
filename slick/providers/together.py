from __future__ import annotations

from typing import Any, List

import langchain_together
import together  # type: ignore
from .base import Provider

class TogetherProvider(Provider):
    name = "together"
    env_key = "TOGETHER_API_KEY"

    @staticmethod
    def list_models() -> List[str]:
        client = together.Together(api_key=TogetherProvider.get_api_key())
        data = client.models.list()
        return sorted(m.get("id") for m in data.get("data", []))

    @staticmethod
    def make_chat(model: str, **kwargs: Any):
        return langchain_together.ChatTogetherAI(model=model, api_key=TogetherProvider.get_api_key(), **kwargs)

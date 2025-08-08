from __future__ import annotations

from typing import Any, List

import mistralai  # type: ignore
import langchain_mistralai  # type: ignore

from .base import Provider


class MistralProvider(Provider):
    name = "mistral"
    env_key = "MISTRAL_API_KEY"

    @staticmethod
    def list_models() -> List[str]:
        key = MistralProvider.get_api_key()
        if not key:
            return []
        client = mistralai.Mistral(api_key=key)
        data = client.models.list()
        return sorted(m.id for m in data.data)

    @staticmethod
    def make_chat(model: str, **kwargs: Any):
        return langchain_mistralai.ChatMistralAI(model=model, api_key=MistralProvider.get_api_key(), **kwargs)



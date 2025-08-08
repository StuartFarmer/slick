from __future__ import annotations

from typing import Any, List

import langchain_openai  # type: ignore
from openai import OpenAI  # type: ignore

from .base import Provider


class OpenAIProvider(Provider):
    name = "openai"
    env_key = "OPENAI_API_KEY"

    @staticmethod
    def list_models() -> List[str]:
        key = OpenAIProvider.get_api_key()
        if not key:
            return []
        client = OpenAI(api_key=key)
        models = client.models.list()
        return sorted(m.id for m in models.data)

    @staticmethod
    def make_chat(model: str, **kwargs: Any):
        return langchain_openai.ChatOpenAI(model=model, api_key=OpenAIProvider.get_api_key(), **kwargs)



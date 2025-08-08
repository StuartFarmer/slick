from __future__ import annotations

from typing import Any, List

import groq  # type: ignore
import langchain_groq  # type: ignore

from .base import Provider


class GroqProvider(Provider):
    name = "groq"
    env_key = "GROQ_API_KEY"

    @staticmethod
    def list_models() -> List[str]:
        key = GroqProvider.get_api_key()
        if not key:
            return []
        client = groq.Groq(api_key=key)
        models = client.models.list()
        return sorted(m.id for m in models.data)

    @staticmethod
    def make_chat(model: str, **kwargs: Any):
        return langchain_groq.ChatGroq(model=model, api_key=GroqProvider.get_api_key(), **kwargs)



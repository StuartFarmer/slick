from __future__ import annotations

from typing import Any, List

import google.generativeai as genai  # type: ignore
import langchain_google_genai  # type: ignore

from .base import Provider


class GoogleProvider(Provider):
    name = "google"
    env_key = "GOOGLE_API_KEY"

    @staticmethod
    def list_models() -> List[str]:
        key = GoogleProvider.get_api_key()
        if not key:
            return []
        genai.configure(api_key=key)
        return sorted(getattr(m, "name", "") for m in genai.list_models())

    @staticmethod
    def make_chat(model: str, **kwargs: Any):
        return langchain_google_genai.ChatGoogleGenerativeAI(
            model=model, api_key=GoogleProvider.get_api_key(), **kwargs
        )



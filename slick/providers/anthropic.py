from typing import Any, List

import anthropic
import langchain_anthropic

from .base import Provider


class AnthropicProvider(Provider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"

    @staticmethod
    def list_models() -> List[str]:
        key = AnthropicProvider.get_api_key()
        if not key:
            return []
        client = anthropic.Anthropic(api_key=key)
        models = client.models.list()
        return sorted(m.id for m in models.data)

    @staticmethod
    def make_chat(model: str, **kwargs: Any):
        return langchain_anthropic.ChatAnthropic(model=model, api_key=AnthropicProvider.get_api_key(), **kwargs)



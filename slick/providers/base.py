from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any, List


class Provider(ABC):
    """Abstract base class for all providers.

    Implementations should use static methods to avoid import-time side effects.
    """

    name: str
    env_key: str

    @classmethod
    def get_api_key(cls) -> str:
        return os.getenv(cls.env_key)

    @staticmethod
    @abstractmethod
    def list_models() -> List[str]:
        """Return a list of model identifiers available for this provider.

        Should gracefully return an empty list if the API key is missing.
        """

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def make_chat(model: str, **kwargs: Any):  # -> BaseChatModel
        """Create a LangChain chat model instance for the given model id.

        Should raise a clear RuntimeError with installation guidance if the provider
        integration package is not installed.
        """

        raise NotImplementedError



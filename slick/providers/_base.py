"""The common text-call contract; optional SDKs and execution live in adapters."""

from abc import ABC, abstractmethod


class Provider(ABC):
    """Built-in providers accept a complete prompt and return final text.

    Pass input positionally for compatibility with legacy CLI keyword names.
    Custom objects can implement either call mode without inheriting this class.
    """

    @abstractmethod
    def call(self, text: str) -> str:
        """Return final text synchronously."""
        raise NotImplementedError

    @abstractmethod
    async def acall(self, text: str) -> str:
        """Return final text asynchronously."""
        raise NotImplementedError


class ProviderError(Exception):
    """A provider could not complete a request."""

"""The common context/tool-result contract; execution remains in application code."""

from abc import ABC, abstractmethod

from ..tools import ToolRequest


class Provider(ABC):
    """One request in, (text, tool_requests) out; no implicit history or execution.

    Custom objects can implement either call mode without inheriting this class.
    """

    @abstractmethod
    def call(self, context: str, *, tools=None, tool_results=None) -> tuple[str, list[ToolRequest]]:
        """Return text and tool requests synchronously."""
        raise NotImplementedError

    @abstractmethod
    async def acall(
        self, context: str, *, tools=None, tool_results=None
    ) -> tuple[str, list[ToolRequest]]:
        """Return text and tool requests asynchronously."""
        raise NotImplementedError


class ProviderError(Exception):
    """A provider could not complete a request."""

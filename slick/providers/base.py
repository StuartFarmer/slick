"""The common context/tool-result contract; execution remains in application code."""

from contextlib import contextmanager

from ..tools import prepare_tools


class Provider:
    """One request in, (text, tool_requests) out; no implicit history or execution.

    Custom objects can implement either call mode without inheriting this class.
    """

    def call(self, context: str, *, tools=None, tool_results=None) -> tuple[str, list[dict]]:
        """Return text and tool requests synchronously."""
        raise NotImplementedError

    async def acall(self, context: str, *, tools=None, tool_results=None) -> tuple[str, list[dict]]:
        """Return text and tool requests asynchronously."""
        raise NotImplementedError

    def identity(self) -> dict:
        return {"provider": self.provider, "model": self.model}


class ProviderError(Exception):
    """A provider could not complete a request."""


class APIProvider(Provider):
    """Prepare a portable call, send its native payload, and decode the response."""

    def call(self, context: str, *, tools=None, tool_results=None):
        with self._request_errors():
            prepared = prepare_tools(tools or [])
            return self._decode(self._send(self._encode(context, prepared, tool_results or [])))

    async def acall(self, context: str, *, tools=None, tool_results=None):
        with self._request_errors():
            prepared = prepare_tools(tools or [])
            response = await self._asend(self._encode(context, prepared, tool_results or []))
            return self._decode(response)

    @contextmanager
    def _request_errors(self):
        """Keep SDK details in the cause; cancellation and process exits propagate."""
        try:
            yield
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.provider} request failed.") from exc

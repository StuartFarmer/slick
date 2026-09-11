"""The common context/tool-result contract; execution remains in application code."""

from contextlib import contextmanager
from typing import Any

from ..prompts import Prompt, _render_prompt, parse
from ..tools import prepare_tools


class Provider:
    """One request in, (text, tool_requests) out; no implicit history or execution.

    Custom objects can implement either call mode without inheriting this class.
    Inherit it to also get prompt/aprompt rendering and output parsing.
    """

    def call(self, context: str, *, tools=None, tool_results=None) -> tuple[str, list[dict]]:
        """Return text and tool requests synchronously."""
        raise NotImplementedError

    async def acall(self, context: str, *, tools=None, tool_results=None) -> tuple[str, list[dict]]:
        """Return text and tool requests asynchronously."""
        raise NotImplementedError

    def prompt(
        self,
        prompt: Prompt,
        /,
        *,
        output_type: Any = None,
        tools=None,
        tool_results=None,
        **variables,
    ) -> tuple[Any, list[dict]]:
        """Render, call once, and return (parsed output or raw text, requests).

        output_type supplies the template's schema variable and the parse type.
        None leaves text and variables unchanged. Tool requests pass through
        without execution or tracking. Provider and parsing errors propagate.
        """
        context = _render_prompt(prompt, output_type, variables)
        text, requests = self.call(context, tools=tools, tool_results=tool_results)
        return parse(text, str if output_type is None else output_type), requests

    async def aprompt(
        self,
        prompt: Prompt,
        /,
        *,
        output_type: Any = None,
        tools=None,
        tool_results=None,
        **variables,
    ) -> tuple[Any, list[dict]]:
        """Like prompt(), using acall asynchronously."""
        context = _render_prompt(prompt, output_type, variables)
        text, requests = await self.acall(context, tools=tools, tool_results=tool_results)
        return parse(text, str if output_type is None else output_type), requests

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

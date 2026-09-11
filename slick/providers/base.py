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

    def turn(self, context, *, tools=None, tool_results=None, continuation=None):
        """Execute one native conversation turn; continuation belongs to the Session."""
        with self._request_errors():
            request = self._encode_turn(
                context, prepare_tools(tools or []), tool_results or [], continuation
            )
            return self._decode_turn(self._send(request), request)

    async def aturn(self, context, *, tools=None, tool_results=None, continuation=None):
        """Async counterpart of turn(); call/acall remain stateless tuple APIs."""
        with self._request_errors():
            request = self._encode_turn(
                context, prepare_tools(tools or []), tool_results or [], continuation
            )
            return self._decode_turn(await self._asend(request), request)

    def _encode_turn(self, context, prepared, results, continuation):
        request = self._encode(context, prepared, results)
        field = "input" if self.provider == "openai" else "messages"
        if continuation is not None:
            if continuation["identity"] != self.identity():
                raise ValueError("Continuation requires the same provider and model")
            messages = continuation["messages"][:]
            if results:
                encoded = self._encode("", prepared, results)[field]
                messages.extend(
                    encoded[-1:] if self.provider == "anthropic" else encoded[-len(results) :]
                )
            if context:
                messages.append({"role": "user", "content": context})
            request[field] = messages
        elif isinstance(request[field], str):
            request[field] = [{"role": "user", "content": context}] if context else []
        if self.provider == "openai":
            request["include"] = ["reasoning.encrypted_content"]
        return request

    def _decode_turn(self, response, request):
        text, requests = self._decode(response)
        if self.provider == "openai":
            messages = request["input"] + _jsonable(response.output)
            status = "complete" if response.status == "completed" else "incomplete"
        elif self.provider == "anthropic":
            messages = request["messages"] + [
                {"role": "assistant", "content": _jsonable(response.content)}
            ]
            status = {
                "end_turn": "complete",
                "stop_sequence": "complete",
                "tool_use": "tools",
                "pause_turn": "continue",
            }.get(response.stop_reason, "incomplete")
        else:
            choice = response.choices[0]
            messages = request["messages"] + [_jsonable(choice.message)]
            status = {"stop": "complete", "tool_calls": "tools"}.get(
                choice.finish_reason, "incomplete"
            )
        if requests and status == "complete":
            status = "tools"
        if status == "tools" and not requests:
            status = "incomplete"
        return {
            "text": text,
            "requests": requests,
            "status": status,
            "continuation": {"identity": self.identity(), "messages": messages},
        }

    @contextmanager
    def _request_errors(self):
        """Keep SDK details in the cause; cancellation and process exits propagate."""
        try:
            yield
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.provider} request failed.") from exc


def _jsonable(value):
    """Preserve SDK blocks, including opaque continuation fields, as plain data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        value = vars(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value

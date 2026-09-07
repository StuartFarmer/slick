"""Optional API backends: one complete prompt in, one complete text response out.

SDKs are imported on first use. Clients created here live for one call;
clients supplied by the application remain the application's responsibility.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import KW_ONLY, dataclass, field
from importlib import import_module
from typing import Any

from . import _anthropic_turns, _openai_turns
from .models import ModelError
from .tools import prepare_tools
from .turns import ModelTurn, ToolResult, UserMessage, validate_history

BackendError = ModelError


def _validate(model, timeout, max_output_tokens, max_retries):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a finite positive number of seconds")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")


def _client(provider, name, injected, timeout, max_retries):
    options = {"timeout": timeout, "max_retries": max_retries}
    if injected is not None:
        # SDK option copies share the application's transport; do not close them.
        return nullcontext(injected.with_options(**options))
    try:
        sdk = import_module(provider)
    except ImportError as exc:
        raise BackendError(f"Install slick-ai[{provider}] to use this backend.") from exc
    return getattr(sdk, name)(**options)


def _openai_text(response):
    if response.status != "completed":
        raise BackendError(f"OpenAI response did not complete: {response.status}.")
    text = []
    for item in response.output:
        if item.type == "reasoning":
            continue
        if item.type != "message":
            raise BackendError(f"Unexpected OpenAI output: {item.type}.")
        if item.status != "completed":
            raise BackendError(f"OpenAI message did not complete: {item.status}.")
        for block in item.content:
            if block.type != "output_text":
                raise BackendError(f"Unexpected OpenAI content: {block.type}.")
            text.append(block.text)
    if not text:
        raise BackendError("OpenAI returned no text content.")
    return "".join(text)


def _anthropic_text(response):
    if response.stop_reason not in {"end_turn", "stop_sequence"}:
        raise BackendError(f"Anthropic response did not complete: {response.stop_reason}.")
    text = []
    for block in response.content:
        if block.type in {"thinking", "redacted_thinking"}:
            continue
        if block.type != "text":
            raise BackendError(f"Unexpected Anthropic content: {block.type}.")
        text.append(block.text)
    if not text:
        raise BackendError("Anthropic returned no text content.")
    return "".join(text)


@dataclass
class OpenAI:
    """OpenAI Responses API. Supply an explicit model and optionally SDK clients."""

    model: str
    _: KW_ONLY
    timeout: float = 60
    max_output_tokens: int = 2048
    max_retries: int = 0
    client: Any = field(default=None, repr=False, compare=False)
    async_client: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        _validate(self.model, self.timeout, self.max_output_tokens, self.max_retries)

    def identity(self) -> dict:
        """Public configuration for opt-in cache keys; excludes credentials."""
        return {
            "provider": "openai",
            "model": self.model,
            "timeout": self.timeout,
            "max_output_tokens": self.max_output_tokens,
            "max_retries": self.max_retries,
        }

    def _request(self, text):
        return {
            "model": self.model,
            "input": text,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }

    def call(self, text: str) -> str:
        try:
            with _client("openai", "OpenAI", self.client, self.timeout, self.max_retries) as client:
                return _openai_text(client.responses.create(**self._request(text)))
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError("OpenAI request failed.") from exc

    async def acall(self, text: str) -> str:
        try:
            async with _client(
                "openai", "AsyncOpenAI", self.async_client, self.timeout, self.max_retries
            ) as client:
                return _openai_text(await client.responses.create(**self._request(text)))
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError("OpenAI request failed.") from exc

    def _turn_request(self, history, tools, instructions):
        prepared = prepare_tools(tools)
        validate_history(history, provider="openai", model=self.model)
        request = self._request(_openai_turns.encode_history(history))
        request.update(
            tools=_openai_turns.tool_definitions(prepared), include=["reasoning.encrypted_content"]
        )
        if instructions:
            request["instructions"] = instructions
        if prepared:
            request["parallel_tool_calls"] = False
        return request

    async def aturn(
        self,
        history: list[UserMessage | ModelTurn | ToolResult],
        *,
        tools: list,
        instructions: str = "",
    ) -> ModelTurn:
        """Make one native request; never execute tools or mutate history."""
        try:
            request = self._turn_request(history, tools, instructions)
            async with _client(
                "openai", "AsyncOpenAI", self.async_client, self.timeout, self.max_retries
            ) as client:
                response = await client.responses.create(**request)
            turn = _openai_turns.decode_turn(response.model_dump(mode="json"), model=self.model)
            if turn.tool_calls and not request["tools"]:
                raise ValueError("OpenAI returned tool calls with no tools available")
            return turn
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"OpenAI native turn failed: {exc}") from exc


@dataclass
class Anthropic:
    """Anthropic Messages API. Supply an explicit model and optionally SDK clients."""

    model: str
    _: KW_ONLY
    timeout: float = 60
    max_output_tokens: int = 2048
    max_retries: int = 0
    client: Any = field(default=None, repr=False, compare=False)
    async_client: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        _validate(self.model, self.timeout, self.max_output_tokens, self.max_retries)

    def identity(self) -> dict:
        """Public configuration for opt-in cache keys; excludes credentials."""
        return {
            "provider": "anthropic",
            "model": self.model,
            "timeout": self.timeout,
            "max_output_tokens": self.max_output_tokens,
            "max_retries": self.max_retries,
        }

    def _request(self, text):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": self.max_output_tokens,
        }

    def call(self, text: str) -> str:
        try:
            with _client(
                "anthropic", "Anthropic", self.client, self.timeout, self.max_retries
            ) as client:
                return _anthropic_text(client.messages.create(**self._request(text)))
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError("Anthropic request failed.") from exc

    async def acall(self, text: str) -> str:
        try:
            async with _client(
                "anthropic", "AsyncAnthropic", self.async_client, self.timeout, self.max_retries
            ) as client:
                return _anthropic_text(await client.messages.create(**self._request(text)))
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError("Anthropic request failed.") from exc

    def _turn_request(self, history, tools, instructions):
        prepared = prepare_tools(tools)
        validate_history(history, provider="anthropic", model=self.model)
        request = {
            "model": self.model,
            "messages": _anthropic_turns.encode_history(history),
            "max_tokens": self.max_output_tokens,
            "tools": _anthropic_turns.tool_definitions(prepared),
        }
        if instructions:
            request["system"] = instructions
        if prepared:
            request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        return request

    async def aturn(
        self,
        history: list[UserMessage | ModelTurn | ToolResult],
        *,
        tools: list,
        instructions: str = "",
    ) -> ModelTurn:
        """Make one native request; never execute tools or mutate history."""
        try:
            request = self._turn_request(history, tools, instructions)
            async with _client(
                "anthropic", "AsyncAnthropic", self.async_client, self.timeout, self.max_retries
            ) as client:
                response = await client.messages.create(**request)
            turn = _anthropic_turns.decode_turn(response.model_dump(mode="json"), model=self.model)
            if turn.tool_calls and not request["tools"]:
                raise ValueError("Anthropic returned tool calls with no tools available")
            return turn
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Anthropic native turn failed: {exc}") from exc

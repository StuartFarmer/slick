"""Native API providers: one complete prompt in, one complete text response out.

SDKs are imported on first use. Native clients created here live for one call;
injected native clients belong to the application.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import KW_ONLY, dataclass, field
from importlib import import_module
from typing import Any

from ..turns import ModelTurn, ToolResult, UserMessage, _chat_completions, validate_history
from ..turns import _anthropic as _anthropic_turns
from ..turns import _openai as _openai_turns
from ..turns.tools import prepare_tools
from ._base import Provider, ProviderError


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


def _client(provider, name, injected, timeout, max_retries, **sdk_options):
    options = {"timeout": timeout, "max_retries": max_retries, **sdk_options}
    if injected is not None:
        # SDK option copies share the application's transport; do not close them.
        return nullcontext(injected.with_options(**options))
    try:
        sdk = import_module(provider)
    except ImportError as exc:
        raise ProviderError(f"Install slick-ai[{provider}] to use this provider.") from exc
    return getattr(sdk, name)(**options)


def _openai_text(response):
    if response.status != "completed":
        raise ProviderError(f"OpenAI response did not complete: {response.status}.")
    text = []
    for item in response.output:
        if item.type == "reasoning":
            continue
        if item.type != "message":
            raise ProviderError(f"Unexpected OpenAI output: {item.type}.")
        if item.status != "completed":
            raise ProviderError(f"OpenAI message did not complete: {item.status}.")
        for block in item.content:
            if block.type != "output_text":
                raise ProviderError(f"Unexpected OpenAI content: {block.type}.")
            text.append(block.text)
    if not text:
        raise ProviderError("OpenAI returned no text content.")
    return "".join(text)


def _anthropic_text(response):
    if response.stop_reason not in {"end_turn", "stop_sequence"}:
        raise ProviderError(f"Anthropic response did not complete: {response.stop_reason}.")
    text = []
    for block in response.content:
        if block.type in {"thinking", "redacted_thinking"}:
            continue
        if block.type != "text":
            raise ProviderError(f"Unexpected Anthropic content: {block.type}.")
        text.append(block.text)
    if not text:
        raise ProviderError("Anthropic returned no text content.")
    return "".join(text)


def _chat_text(response, provider):
    if len(response.choices) != 1:
        raise ProviderError(f"{provider} must return exactly one choice.")
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ProviderError(f"{provider} response did not complete as text.")
    message = choice.message
    if any(getattr(message, field, None) for field in ("tool_calls", "function_call", "refusal")):
        raise ProviderError(f"{provider} returned tool calls or a refusal instead of final text.")
    if not isinstance(message.content, str):
        raise ProviderError(f"{provider} returned no text content.")
    return message.content


@dataclass
class OpenRouterAPI(Provider):
    """Direct OpenRouter Chat Completions through the optional OpenAI SDK."""

    model: str
    _: KW_ONLY
    api_key: str | None = field(default=None, repr=False, compare=False)
    timeout: float = 60
    max_output_tokens: int = 2048
    max_retries: int = 0
    client: Any = field(default=None, repr=False, compare=False)
    async_client: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        _validate(self.model, self.timeout, self.max_output_tokens, self.max_retries)
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("api_key must be a nonempty string")

    def identity(self) -> dict:
        return {
            "provider": "openrouter",
            "model": self.model,
            "timeout": self.timeout,
            "max_output_tokens": self.max_output_tokens,
            "max_retries": self.max_retries,
        }

    def _client(self, asynchronous=False):
        injected = self.async_client if asynchronous else self.client
        key = self.api_key if self.api_key is not None else os.getenv("OPENROUTER_API_KEY")
        options = {"base_url": "https://openrouter.ai/api/v1"}
        if key is not None:
            if not key.strip():
                raise ProviderError("OPENROUTER_API_KEY must be nonempty.")
            options["api_key"] = key
        else:
            raise ProviderError("Set OPENROUTER_API_KEY or pass api_key to OpenRouterAPI.")
        return _client(
            "openai",
            "AsyncOpenAI" if asynchronous else "OpenAI",
            injected,
            self.timeout,
            self.max_retries,
            **options,
        )

    def _request(self, text):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": self.max_output_tokens,
            "stream": False,
            "n": 1,
        }

    def call(self, text: str) -> str:
        try:
            with self._client() as client:
                return _chat_text(
                    client.chat.completions.create(**self._request(text)), "OpenRouter"
                )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenRouter request failed.") from exc

    async def acall(self, text: str) -> str:
        try:
            async with self._client(asynchronous=True) as client:
                response = await client.chat.completions.create(**self._request(text))
                return _chat_text(response, "OpenRouter")
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenRouter request failed.") from exc

    async def aturn(
        self,
        history: list[UserMessage | ModelTurn | ToolResult],
        *,
        tools: list,
        instructions: str = "",
    ) -> ModelTurn:
        """Make one Chat Completions turn without executing tools."""
        try:
            prepared = prepare_tools(tools)
            validate_history(history, provider="openrouter", model=self.model)
            request = self._request("")
            request["messages"] = _chat_completions.encode_history(
                history, instructions=instructions
            )
            if prepared:
                request["tools"] = _chat_completions.tool_definitions(prepared)
            async with self._client(asynchronous=True) as client:
                response = await client.chat.completions.create(**request)
            turn = _chat_completions.decode_turn(
                response.model_dump(mode="json", exclude_none=True),
                provider="openrouter",
                model=self.model,
            )
            if turn.tool_calls and not prepared:
                raise ValueError("OpenRouter returned tool calls with no tools available")
            return turn
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenRouter native turn failed.") from exc


@dataclass
class OpenAIAPI(Provider):
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
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenAI request failed.") from exc

    async def acall(self, text: str) -> str:
        try:
            async with _client(
                "openai", "AsyncOpenAI", self.async_client, self.timeout, self.max_retries
            ) as client:
                return _openai_text(await client.responses.create(**self._request(text)))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenAI request failed.") from exc

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
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"OpenAI native turn failed: {exc}") from exc


@dataclass
class AnthropicAPI(Provider):
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
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Anthropic request failed.") from exc

    async def acall(self, text: str) -> str:
        try:
            async with _client(
                "anthropic", "AsyncAnthropic", self.async_client, self.timeout, self.max_retries
            ) as client:
                return _anthropic_text(await client.messages.create(**self._request(text)))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Anthropic request failed.") from exc

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
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Anthropic native turn failed: {exc}") from exc

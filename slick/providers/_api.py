"""Native API providers: explicit context and tool results in, text and requests out.

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

from ..tools._protocol import prepare_call
from ._base import Provider, ProviderError
from ._tools import _anthropic, _chat_completions, _openai


def _validate(model, timeout, max_output_tokens, max_retries):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
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


def _decode(codec, response, prepared):
    try:
        text, requests = codec.decode_response(response.model_dump(mode="json", exclude_none=True))
    except (ValueError, TypeError) as exc:
        raise ProviderError(f"Invalid provider response: {exc}") from exc
    if requests and not prepared:
        raise ValueError("Provider returned tool requests with no tools available")
    return text, requests


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

    def _request(self, context, tools, tool_results):
        prepared, results = prepare_call(context, tools, tool_results)
        request = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "stream": False,
            "n": 1,
        }
        request.update(_chat_completions.encode_request(context, prepared, results))
        return request, prepared

    def call(self, context: str, *, tools=None, tool_results=None):
        try:
            request, prepared = self._request(context, tools, tool_results)
            with self._client() as client:
                response = client.chat.completions.create(**request)
                return _decode(_chat_completions, response, prepared)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenRouter request failed.") from exc

    async def acall(self, context: str, *, tools=None, tool_results=None):
        try:
            request, prepared = self._request(context, tools, tool_results)
            async with self._client(asynchronous=True) as client:
                response = await client.chat.completions.create(**request)
                return _decode(_chat_completions, response, prepared)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenRouter request failed.") from exc


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

    def _request(self, context, tools, tool_results):
        prepared, results = prepare_call(context, tools, tool_results)
        request = {"model": self.model, "max_output_tokens": self.max_output_tokens, "store": False}
        request.update(_openai.encode_request(context, prepared, results))
        return request, prepared

    def call(self, context: str, *, tools=None, tool_results=None):
        try:
            request, prepared = self._request(context, tools, tool_results)
            with _client("openai", "OpenAI", self.client, self.timeout, self.max_retries) as client:
                response = client.responses.create(**request)
                return _decode(_openai, response, prepared)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenAI request failed.") from exc

    async def acall(self, context: str, *, tools=None, tool_results=None):
        try:
            request, prepared = self._request(context, tools, tool_results)
            async with _client(
                "openai", "AsyncOpenAI", self.async_client, self.timeout, self.max_retries
            ) as client:
                response = await client.responses.create(**request)
                return _decode(_openai, response, prepared)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenAI request failed.") from exc


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

    def _request(self, context, tools, tool_results):
        prepared, results = prepare_call(context, tools, tool_results)
        request = {"model": self.model, "max_tokens": self.max_output_tokens}
        request.update(_anthropic.encode_request(context, prepared, results))
        return request, prepared

    def call(self, context: str, *, tools=None, tool_results=None):
        try:
            request, prepared = self._request(context, tools, tool_results)
            with _client(
                "anthropic", "Anthropic", self.client, self.timeout, self.max_retries
            ) as client:
                response = client.messages.create(**request)
                return _decode(_anthropic, response, prepared)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Anthropic request failed.") from exc

    async def acall(self, context: str, *, tools=None, tool_results=None):
        try:
            request, prepared = self._request(context, tools, tool_results)
            async with _client(
                "anthropic", "AsyncAnthropic", self.async_client, self.timeout, self.max_retries
            ) as client:
                response = await client.messages.create(**request)
                return _decode(_anthropic, response, prepared)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Anthropic request failed.") from exc

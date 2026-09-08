"""Optional LiteLLM text completion; no SDK import until a call is made."""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field
from importlib import import_module

from ._base import Provider, ProviderError

_RESERVED_OPTIONS = {
    "model",
    "messages",
    "api_base",
    "base_url",
    "api_key",
    "timeout",
    "num_retries",
    "max_retries",
    "stream",
    "stream_options",
    "n",
    "drop_params",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "parallel_tool_calls",
    "input",
    "previous_response_id",
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "model_list",
    "router",
    "client",
    "acompletion",
}
_TOKEN_LIMITS = {"max_tokens", "max_output_tokens", "max_completion_tokens"}


def _sdk():
    if sys.version_info >= (3, 15):
        raise ProviderError("The LiteLLM extra requires Python >=3.10,<3.15.")
    try:
        return import_module("litellm")
    except ImportError as exc:
        message = (
            "Install slick-ai[litellm] to use this provider."
            if exc.name == "litellm"
            else "LiteLLM could not import a dependency."
        )
        raise ProviderError(message) from exc


def _text(response):
    if len(response.choices) != 1:
        raise ProviderError("LiteLLM must return exactly one choice.")
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ProviderError("LiteLLM response did not complete as text.")
    message = choice.message
    if any(getattr(message, field, None) for field in ("tool_calls", "function_call", "refusal")):
        raise ProviderError("LiteLLM returned tool calls or a refusal instead of final text.")
    if not isinstance(message.content, str):
        raise ProviderError("LiteLLM returned no text content.")
    return message.content


@dataclass
class LiteLLMGateway(Provider):
    """Provider-prefixed model access through the optional LiteLLM SDK.

    Options are provider-specific inference settings. Calls do not configure
    tools, streaming, fallback routing, or persistent conversation state.
    """

    model: str
    _: KW_ONLY
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 60
    max_retries: int = 0
    options: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        for name in ("model", "api_base", "api_key"):
            value = getattr(self, name)
            if value is None and name != "model":
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number of seconds")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if self.options is None:
            self.options = {}
        if not isinstance(self.options, dict) or any(not isinstance(k, str) for k in self.options):
            raise ValueError("options must be a dictionary with string keys")
        if _RESERVED_OPTIONS.intersection(self.options):
            raise ValueError("options cannot override call, tool, or routing controls")
        if self.model.startswith("chatgpt/") and _TOKEN_LIMITS.intersection(self.options):
            raise ValueError("chatgpt/ does not enforce output token limits")
        self.options = deepcopy(self.options)

    def _request(self, text):
        request = {
            **deepcopy(self.options),
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "timeout": self.timeout,
            "num_retries": self.max_retries,
            "drop_params": False,
            "stream": False,
            "n": 1,
        }
        if self.api_base is not None:
            request["api_base"] = self.api_base
        if self.api_key is not None:
            request["api_key"] = self.api_key
        return request

    def call(self, text: str) -> str:
        try:
            return _text(_sdk().completion(**self._request(text)))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("LiteLLM request failed.") from exc

    async def acall(self, text: str) -> str:
        try:
            return _text(await _sdk().acompletion(**self._request(text)))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("LiteLLM request failed.") from exc

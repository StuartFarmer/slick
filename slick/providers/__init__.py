"""Providers for CLI tools, remote inference, and local inference endpoints.

Importing or constructing a provider does not import optional SDKs or execute
requests. A model ID selects a model within the configured provider.
"""

from ._api import AnthropicAPI, OpenAIAPI, OpenRouterAPI
from ._base import Provider, ProviderError
from ._command import (
    ClaudeCLI,
    CodexCLI,
    Command,
    ExecutionResult,
    get_command,
    get_default,
    set_default,
)
from ._litellm import LiteLLMAPI

__all__ = [
    "AnthropicAPI",
    "ClaudeCLI",
    "CodexCLI",
    "Command",
    "ExecutionResult",
    "LiteLLMAPI",
    "OpenAIAPI",
    "OpenRouterAPI",
    "Provider",
    "ProviderError",
    "get_command",
    "get_default",
    "set_default",
]

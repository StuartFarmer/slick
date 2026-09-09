"""Canonical provider classes share one execution contract."""

import subprocess
import sys
from dataclasses import is_dataclass
from inspect import signature


def test_providers_use_handwritten_constructors_without_client_injection():
    from slick.providers import (
        AnthropicAPI,
        Command,
        ExecutionResult,
        LiteLLMAPI,
        OpenAIAPI,
        OpenRouterAPI,
    )

    for cls in (AnthropicAPI, Command, ExecutionResult, LiteLLMAPI, OpenAIAPI, OpenRouterAPI):
        assert not is_dataclass(cls)
        assert "client" not in signature(cls).parameters
        assert "async_client" not in signature(cls).parameters


def test_provider_names_and_ownership():
    import slick
    from slick import providers

    assert slick.Provider is providers.Provider
    assert slick.ProviderError is providers.ProviderError
    for name in (
        "CodexCLI",
        "ClaudeCLI",
        "OpenAIAPI",
        "AnthropicAPI",
        "LiteLLMAPI",
        "OpenRouterAPI",
    ):
        cls = getattr(providers, name)
        assert cls.__name__ == name
        assert cls.__module__.startswith("slick.providers.")
        assert issubclass(cls, providers.Provider)


def test_base_import_and_cli_do_not_load_optional_sdks_or_start_io():
    code = """
import sys

def forbid(event, args):
    if event in {'subprocess.Popen', 'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError(event)
    if event == 'import' and args[0].split('.')[0] in {'openai', 'anthropic', 'litellm'}:
        raise AssertionError(args[0])

sys.addaudithook(forbid)
import slick
from slick import cli
from examples import _cli
from examples.coding_harness import __main__
from slick.providers import Provider, CodexCLI, ClaudeCLI
for provider in (CodexCLI(), ClaudeCLI()):
    assert isinstance(provider, Provider)
assert cli.build_parser().parse_args(['call', '--provider', 'codex']).provider == 'codex'
assert not {'openai', 'anthropic', 'litellm'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_optional_sdks_load_only_when_a_request_is_sent():
    code = """
import sys
sys.modules['openai'] = None
sys.modules['anthropic'] = None
sys.modules['litellm'] = None
import slick
from slick import providers
for name in ('OpenAIAPI', 'OpenRouterAPI', 'AnthropicAPI', 'LiteLLMAPI'):
    try:
        options = {"api_key": "offline"} if name == "OpenRouterAPI" else {}
        getattr(providers, name)("model", **options).call("input")
    except providers.ProviderError as exc:
        assert isinstance(exc.__cause__, ModuleNotFoundError)
    else:
        raise AssertionError(name)
assert providers.CodexCLI().provider == 'codex'
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

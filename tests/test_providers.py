"""Canonical provider classes share one execution contract."""

import inspect
import subprocess
import sys


def test_provider_names_and_ownership():
    import slick
    from slick import providers

    assert slick.Provider is providers.Provider
    assert slick.ProviderError is providers.ProviderError
    assert inspect.isabstract(providers.Provider)
    assert inspect.isabstract(providers.Command)
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
    assert slick.get_command is providers.get_command


def test_public_defaults_share_state(monkeypatch):
    import slick
    from slick import providers
    from slick.providers import _command

    monkeypatch.setattr(_command, "_default_provider", None)
    monkeypatch.setattr(_command, "_default_model", None)
    monkeypatch.delenv("SLICK_PROVIDER", raising=False)
    monkeypatch.delenv("SLICK_MODEL", raising=False)
    slick.set_default(provider="claude", model="selected")
    provider = providers.get_command()
    assert isinstance(provider, providers.ClaudeCLI)
    assert provider.model == "selected"
    providers.set_default(provider="codex")
    assert slick.get_default() == ("codex", "selected")
    assert isinstance(slick.get_command(), providers.CodexCLI)


def test_imports_and_construction_do_not_load_sdks_or_start_io():
    code = """
import sys

def forbid(event, args):
    if event in {'subprocess.Popen', 'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError(event)
    if event == 'import' and args[0].split('.')[0] in {'openai', 'anthropic', 'litellm'}:
        raise AssertionError(args[0])

sys.addaudithook(forbid)
import slick
from slick.providers import (
    Provider, Command, CodexCLI, ClaudeCLI, OpenAIAPI, AnthropicAPI, LiteLLMAPI, OpenRouterAPI,
)
for provider in (
    CodexCLI(), ClaudeCLI(), OpenAIAPI('test'), AnthropicAPI('test'), LiteLLMAPI('test'),
    OpenRouterAPI('vendor/model'),
):
    assert isinstance(provider, Provider)
assert not {'openai', 'anthropic', 'litellm'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

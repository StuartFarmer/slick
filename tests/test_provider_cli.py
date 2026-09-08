"""Provider CLI routing and output tests never launch an agent or contact a provider."""

from io import StringIO
from types import SimpleNamespace as NS

import pytest

from slick import cli
from slick.providers import ProviderError


@pytest.mark.parametrize(
    "name,cls",
    [
        ("litellm", "LiteLLMAPI"),
        ("openai", "OpenAIAPI"),
        ("openrouter", "OpenRouterAPI"),
        ("anthropic", "AnthropicAPI"),
    ],
)
def test_explicit_api_provider_receives_stdin_and_model(monkeypatch, capsys, tmp_path, name, cls):
    seen = []

    def factory(**kwargs):
        seen.append(kwargs)
        return NS(call=lambda text: "answer:" + text)

    monkeypatch.setattr(cli, cls, factory)
    monkeypatch.setattr("sys.stdin", StringIO("question\n"))
    output = tmp_path / "nested" / "answer.txt"
    args = ["call", "--provider", name, "--model", "private", "--output", str(output)]
    if name == "litellm":
        args.extend(["--api-base", "http://localhost:8000/v1"])
    assert cli.main(args) == 0
    expected = {"model": "private"}
    if name == "litellm":
        expected["api_base"] = "http://localhost:8000/v1"
    assert seen == [expected]
    assert capsys.readouterr().out == "answer:question\n\n"
    assert output.read_text() == "answer:question\n"


@pytest.mark.parametrize("name", ["litellm", "openai", "anthropic", "openrouter"])
def test_api_model_is_required_even_when_cli_default_exists(monkeypatch, capsys, name):
    monkeypatch.setenv("SLICK_MODEL", "cli-default")
    assert cli.main(["call", "prompt", "--provider", name]) == 2
    assert "--model" in capsys.readouterr().err


@pytest.mark.parametrize("name", [None, "codex", "claude", "openai", "anthropic", "openrouter"])
def test_endpoint_flag_is_rejected_for_other_providers(capsys, name):
    args = ["call", "prompt", "--api-base", "http://localhost:8000", "--model", "test"]
    if name:
        args.extend(["--provider", name])
    assert cli.main(args) == 2
    assert "--api-base" in capsys.readouterr().err


def test_omitted_provider_uses_legacy_resolution(monkeypatch, capsys):
    seen = []

    def get_command(provider, model):
        seen.append((provider, model))
        return NS(call=lambda text: "legacy:" + text)

    monkeypatch.setattr(cli, "get_command", get_command)
    assert cli.main(["call", "prompt"]) == 0
    assert seen == [(None, None)]
    assert capsys.readouterr().out == "legacy:prompt\n"


def test_provider_command_shows_default_selection(monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_default", lambda: ("claude", None))
    assert cli.main(["provider"]) == 0
    assert capsys.readouterr().out == "provider=claude model=(provider default)\n"


@pytest.mark.parametrize("error", [ValueError("bad configuration"), ProviderError("missing SDK")])
def test_api_failures_are_reported_without_traceback(monkeypatch, capsys, error):
    def factory(**kwargs):
        raise error

    monkeypatch.setattr(cli, "LiteLLMAPI", factory)
    assert cli.main(["call", "prompt", "--provider", "litellm", "--model", "test"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == f"{error}\n"


def test_empty_input_is_rejected_before_constructing_provider(monkeypatch, capsys):
    def factory(**kwargs):
        raise AssertionError("provider constructed for empty input")

    monkeypatch.setattr(cli, "LiteLLMAPI", factory)
    monkeypatch.setattr("sys.stdin", StringIO("  \n"))
    assert cli.main(["call", "--provider", "litellm", "--model", "test"]) == 2
    assert "No prompt" in capsys.readouterr().err

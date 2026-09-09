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
        return NS(call=lambda text: ("answer:" + text, []))

    monkeypatch.setattr(cli.providers, cls, factory)
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
def test_api_model_is_required(capsys, name):
    assert cli.main(["call", "prompt", "--provider", name]) == 2
    assert "--model" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["codex", "claude", "openai", "anthropic", "openrouter"])
def test_endpoint_flag_is_rejected_for_other_providers(capsys, name):
    args = ["call", "prompt", "--api-base", "http://localhost:8000", "--model", "test"]
    args.extend(["--provider", name])
    assert cli.main(args) == 2
    assert "--api-base" in capsys.readouterr().err


@pytest.mark.parametrize("name,cls", [("codex", "CodexCLI"), ("claude", "ClaudeCLI")])
@pytest.mark.parametrize("model", [None, "selected"])
def test_cli_provider_receives_explicit_model(monkeypatch, capsys, name, cls, model):
    seen = []

    def factory(**kwargs):
        seen.append(kwargs)
        return NS(call=lambda text: ("answer:" + text, []))

    monkeypatch.setattr(cli.providers, cls, factory)
    args = ["call", "prompt", "--provider", name]
    if model is not None:
        args.extend(["--model", model])
    assert cli.main(args) == 0
    assert seen == [{"model": model}]
    assert capsys.readouterr().out == "answer:prompt\n"


def test_provider_is_required(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["call", "prompt"])
    assert exc.value.code == 2
    assert "--provider" in capsys.readouterr().err


@pytest.mark.parametrize("error", [ValueError("bad configuration"), ProviderError("missing SDK")])
def test_api_failures_are_reported_without_traceback(monkeypatch, capsys, error):
    def factory(**kwargs):
        raise error

    monkeypatch.setattr(cli.providers, "LiteLLMAPI", factory)
    assert cli.main(["call", "prompt", "--provider", "litellm", "--model", "test"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == f"{error}\n"


def test_empty_input_is_rejected_before_constructing_provider(monkeypatch, capsys):
    def factory(**kwargs):
        raise AssertionError("provider constructed for empty input")

    monkeypatch.setattr(cli.providers, "LiteLLMAPI", factory)
    monkeypatch.setattr("sys.stdin", StringIO("  \n"))
    assert cli.main(["call", "--provider", "litellm", "--model", "test"]) == 2
    assert "No prompt" in capsys.readouterr().err

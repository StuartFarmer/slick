import sys
from types import SimpleNamespace as NS

import pytest
from sdk_fakes import sdk_response
from test_tool_providers import provider, reply, request


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_decoders_extract_available_output_without_checking_status(name):
    raw = reply(name)
    if name == "openai":
        raw["status"] = "incomplete"
        raw["output"].append({"type": "reasoning"})
    elif name == "anthropic":
        raw["stop_reason"] = "max_tokens"
        raw["content"].append({"type": "thinking"})
    else:
        raw["choices"][0]["finish_reason"] = "length"
        raw["choices"][0]["message"]["reasoning"] = "internal"
    assert provider(name)._decode(sdk_response(raw)) == ("  answer\n", [request()])


def test_litellm_options_are_passed_to_the_sdk(monkeypatch):
    from slick.providers import LiteLLMAPI

    captured = []

    def completion(**kwargs):
        captured.append(kwargs)
        return NS(choices=[NS(message=NS(content="partial", tool_calls=None))])

    monkeypatch.setitem(sys.modules, "litellm", NS(completion=completion))
    instance = LiteLLMAPI("chatgpt/example", options={"max_tokens": 20, "n": 2})
    assert instance.call("hello") == ("partial", [])
    assert captured[0]["max_tokens"] == 20
    assert captured[0]["n"] == 2


def test_command_passes_context_without_inspecting_tools():
    from test_command_providers import FakeCommand

    command = FakeCommand()
    assert command.call("hello", tools=[object()], tool_results=[{}]) == ("report body", [])
    assert command.runs[0][1] == "hello"

"""LiteLLM harness turns use offline responses and real workspace tools."""

import asyncio
import json
import socket
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from examples.coding_harness import __main__ as cli
from examples.coding_harness.session import load_session, restore_session, save_session
from slick.providers import ProviderError
from slick.turns import ToolResult, UserMessage

MODEL = "openrouter/openai/gpt-oss-120b:nitro"


def reply(content="Done", calls=None, finish=None):
    return {
        "choices": [
            {
                "finish_reason": finish or ("tool_calls" if calls else "stop"),
                "message": {"role": "assistant", "content": content, "tool_calls": calls},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


def tool_call(arguments="{}"):
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": "list_files", "arguments": arguments},
    }


def install_sdk(monkeypatch, responses, requests):
    async def acompletion(**request):
        requests.append(deepcopy(request))
        response = responses.pop(0)
        return SimpleNamespace(model_dump=lambda **kwargs: deepcopy(response))

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=acompletion))


def test_cli_litellm_executes_tools_and_returns_the_final_answer(repo, monkeypatch, capsys):
    requests = []
    first = reply(None, [tool_call()])
    first["choices"][0]["message"]["reasoning_content"] = "Inspect the files first."
    install_sdk(monkeypatch, [first, reply("Workspace inspected.")], requests)
    assert (
        cli.main(
            [
                "--provider",
                "litellm",
                "--model",
                MODEL,
                "--workspace",
                str(repo),
                "--headless",
                "--task",
                "List the files",
            ]
        )
        == 2
    )  # A live workspace without configured checks is unverified.
    assert "Workspace inspected." in capsys.readouterr().out
    assert len(requests) == 2
    assert requests[0]["model"] == MODEL
    assert requests[0]["messages"][0]["role"] == "system"
    assert requests[0]["stream"] is False
    assert requests[0]["drop_params"] is False
    assert requests[0]["tools"][0]["type"] == "function"
    assert requests[1]["messages"][-2] == first["choices"][0]["message"]
    result = requests[1]["messages"][-1]
    assert result["role"] == "tool" and result["tool_call_id"] == "call-1"
    assert "sample.py" in result["content"]


def test_litellm_session_round_trip_preserves_history(repo, tmp_path, monkeypatch):
    from examples.coding_harness.state import HarnessConfig
    from slick.providers import LiteLLMAPI as LiteLLMProvider

    requests = []
    install_sdk(monkeypatch, [reply("First answer"), reply("Second answer")], requests)

    async def run():
        provider = LiteLLMProvider(MODEL)
        agent = await cli.create_agent(
            provider, repo, HarnessConfig(), decide=cli.deny, emit=lambda event: None
        )
        await agent.run("First task")
        path = tmp_path.parent / f"{tmp_path.name}-session.json"
        save_session(path, agent)
        saved = load_session(path)
        restored = await restore_session(
            saved, LiteLLMProvider(MODEL), decide=cli.deny, emit=lambda event: None
        )
        result = await restored.run("Follow up")
        assert result.answer == "Second answer"
        assert restored.state.provider == "litellm"
        assert restored.state.model == MODEL

    asyncio.run(run())
    assert any(message.get("content") == "First answer" for message in requests[1]["messages"])


@pytest.mark.parametrize(
    "response",
    [
        reply(finish="length"),
        reply(finish="content_filter"),
        reply(None),
        reply(None, [tool_call(), tool_call()]),
        reply("Bad", [tool_call()], finish="stop"),
    ],
)
def test_invalid_turns_fail_before_tool_execution(monkeypatch, response):
    from slick.providers import LiteLLMAPI as LiteLLMProvider

    def list_files() -> list[str]:
        """List files."""
        raise AssertionError("Providers must not execute tools")

    install_sdk(monkeypatch, [response], [])
    with pytest.raises(ProviderError):
        asyncio.run(LiteLLMProvider(MODEL).aturn([UserMessage("Go")], tools=[list_files]))


def test_bad_arguments_are_recoverable_and_no_tools_are_allowed_in_summary(monkeypatch):
    from slick.providers import LiteLLMAPI as LiteLLMProvider

    def list_files() -> list[str]:
        """List files."""
        return []

    requests = []
    install_sdk(monkeypatch, [reply(None, [tool_call("{bad")]), reply("Summary")], requests)

    async def run():
        provider = LiteLLMProvider(MODEL)
        history = [UserMessage("Go")]
        turn = await provider.aturn(history, tools=[list_files])
        assert turn.tool_calls[0].argument_error
        assert turn.input_tokens == 12 and turn.output_tokens == 3
        assert history == [UserMessage("Go")]
        await provider.aturn(
            [*history, turn, ToolResult("call-1", "Invalid arguments", True)], tools=[]
        )

    asyncio.run(run())
    assert "tools" not in requests[1]


def test_provider_failures_do_not_expose_credentials(monkeypatch):
    from slick.providers import LiteLLMAPI as LiteLLMProvider

    async def fail(**request):
        raise RuntimeError("private-key")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=fail))
    provider = LiteLLMProvider(MODEL, api_key="private-key")
    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.aturn([UserMessage("Go")], tools=[]))
    assert "private-key" not in str(caught.value)
    assert "private-key" not in json.dumps(provider.identity())


def test_real_sdk_sends_openrouter_key_model_and_tool_results(monkeypatch):
    from slick.providers import LiteLLMAPI as LiteLLMProvider

    httpx = pytest.importorskip("httpx")

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected external network access")

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    sdk = pytest.importorskip("litellm")
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    monkeypatch.setattr(sdk, "telemetry", False)
    monkeypatch.setattr(sdk, "in_memory_llm_clients_cache", type(sdk.in_memory_llm_clients_cache)())
    requests = []

    def respond(request):
        requests.append(request)
        payload = reply(None, [tool_call()]) if len(requests) == 1 else reply("Files checked.")
        payload.update(id="chatcmpl-offline", object="chat.completion", created=1700000000)
        payload["model"] = "openai/gpt-oss-120b"
        payload["choices"][0]["index"] = 0
        payload["usage"]["total_tokens"] = 15
        return httpx.Response(200, json=payload)

    def list_files() -> list[str]:
        """List workspace files."""
        return ["sample.py"]

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            monkeypatch.setattr(AsyncHTTPHandler, "create_client", lambda self, **kwargs: client)
            provider = LiteLLMProvider(MODEL)
            history = [UserMessage("List files")]
            turn = await provider.aturn(history, tools=[list_files], instructions="Inspect files.")
            assert turn.tool_calls[0].name == "list_files"
            final = await provider.aturn(
                [*history, turn, ToolResult("call-1", '["sample.py"]')], tools=[list_files]
            )
            assert final.text == "Files checked."

    asyncio.run(run())
    assert len(requests) == 2
    for request in requests:
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer offline-openrouter-key"
        payload = json.loads(request.content)
        assert payload["model"] == "openai/gpt-oss-120b:nitro"
        assert payload["tools"][0]["function"]["name"] == "list_files"
    assert json.loads(requests[1].content)["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '["sample.py"]',
    }

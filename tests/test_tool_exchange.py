"""Public call/acall: explicit input, portable results, no retained exchange state."""

import asyncio
import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
from sdk_fakes import sdk_response
from test_tool_providers import read, reply

from slick.providers import AnthropicAPI, LiteLLMAPI, OpenAIAPI, OpenRouterAPI, ProviderError


class Client:
    def __init__(self, data, asynchronous=False):
        self.data = data
        self.requests = []
        create = self.acreate if asynchronous else self.create
        self.responses = self.messages = SimpleNamespace(create=create)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if isinstance(self.data, BaseException):
            raise self.data
        return sdk_response(self.data)

    async def acreate(self, **kwargs):
        return self.create(**kwargs)


def build(name, client, asynchronous, monkeypatch):
    cls = {
        "openai": OpenAIAPI,
        "anthropic": AnthropicAPI,
        "openrouter": OpenRouterAPI,
        "litellm": LiteLLMAPI,
    }[name]
    if name == "litellm":
        monkeypatch.setitem(
            sys.modules,
            "litellm",
            SimpleNamespace(
                completion=client.create,
                acompletion=client.acreate,
            ),
        )
        return cls("vendor/model")
    sdk = "Anthropic" if name == "anthropic" else "OpenAI"
    factory = ("Async" if asynchronous else "") + sdk
    monkeypatch.setitem(sys.modules, sdk.lower(), SimpleNamespace(**{factory: lambda **kw: client}))
    return cls("model", **({"api_key": "offline"} if name == "openrouter" else {}))


def invoke(provider, asynchronous, context, **kwargs):
    if asynchronous:
        return asyncio.run(provider.acall(context, **kwargs))
    return provider.call(context, **kwargs)


@pytest.mark.parametrize("name", ["openai", "anthropic", "openrouter", "litellm"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_results_work_on_a_fresh_instance_with_replaced_context(name, asynchronous, monkeypatch):
    format = name if name in {"openai", "anthropic"} else "chat_completions"
    first = Client(reply(format), asynchronous)
    provider = build(name, first, asynchronous, monkeypatch)
    text, requests = invoke(provider, asynchronous, "ORIGINAL_CONTEXT", tools=[read])
    assert text == "  answer\n"
    assert requests[0]["name"] == "read"
    results = json.loads(json.dumps([{"request": requests[0], "content": "found file"}]))
    saved = deepcopy(results)
    second = Client(reply(format, calls=False), asynchronous)
    fresh = build(name, second, asynchronous, monkeypatch)
    assert invoke(fresh, asynchronous, "NEW_SUMMARY", tools=[], tool_results=results) == (
        "  answer\n",
        [],
    )
    wire = json.dumps(second.requests)
    assert "ORIGINAL_CONTEXT" not in wire
    assert "NEW_SUMMARY" in wire and "found file" in wire and "a.py" in wire
    assert len(first.requests) == len(second.requests) == 1
    assert results == saved


@pytest.mark.parametrize("name", ["openai", "anthropic", "openrouter", "litellm"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_invalid_input_fails_before_network(name, asynchronous, monkeypatch):
    client = Client({}, asynchronous)
    provider = build(name, client, asynchronous, monkeypatch)
    with pytest.raises(ProviderError):
        invoke(
            provider, asynchronous, "context", tool_results=[{"request_id": "x", "content": "x"}]
        )
    assert client.requests == []


@pytest.mark.parametrize("name", ["openai", "anthropic", "openrouter", "litellm"])
def test_unadvertised_tools_and_transport_errors(name, monkeypatch):
    format = name if name in {"openai", "anthropic"} else "chat_completions"
    client = Client(reply(format))
    provider = build(name, client, False, monkeypatch)
    _, requests = provider.call("context")
    assert requests[0]["name"] == "read"
    failure = RuntimeError("offline failure")
    client.data = failure
    with pytest.raises(ProviderError) as error:
        provider.call("context")
    assert error.value.__cause__ is failure


@pytest.mark.parametrize("name", ["openai", "anthropic", "openrouter", "litellm"])
def test_interleaved_exchanges_do_not_replace_each_others_requests(name, monkeypatch):
    format = name if name in {"openai", "anthropic"} else "chat_completions"
    client = Client(reply(format))
    provider = build(name, client, False, monkeypatch)
    _, first = provider.call("Task A", tools=[read])
    client.data = reply(format, arguments={"path": "b.py"})
    _, second = provider.call("Task B", tools=[read])
    assert first[0]["arguments"] == {"path": "a.py"}
    assert second[0]["arguments"] == {"path": "b.py"}
    results = [{"request": first[0], "content": "A output"}]
    original = deepcopy(results)
    client.data = RuntimeError("temporary transport failure")
    with pytest.raises(ProviderError):
        provider.call("Resume A", tool_results=results)
    client.data = reply(format, calls=False)
    provider.call("Resume A", tool_results=results)
    wire = json.dumps(client.requests[-1])
    assert "a.py" in wire and "b.py" not in wire
    assert results == original


@pytest.mark.parametrize("name", ["openai", "anthropic", "openrouter", "litellm"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raw", [None, [], {"choices": [None]}])
def test_malformed_sdk_responses_raise_provider_errors(name, asynchronous, raw, monkeypatch):
    client = Client(raw, asynchronous)
    provider = build(name, client, asynchronous, monkeypatch)
    with pytest.raises(ProviderError):
        invoke(provider, asynchronous, "context")

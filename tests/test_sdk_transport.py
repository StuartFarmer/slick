"""Real SDK requests through offline HTTP transports and provider-owned clients."""

import asyncio
import json
import socket
import sys

import pytest
from test_tool_providers import read, reply

from slick import Session
from slick.providers import AnthropicAPI, OpenAIAPI, OpenRouterAPI, ProviderError

httpx = pytest.importorskip("httpx")
PROVIDERS = {"openai": OpenAIAPI, "anthropic": AnthropicAPI, "chat_completions": OpenRouterAPI}


def wire_reply(name, *, calls=False):
    raw = reply(name, calls=calls)
    raw.update(id="offline", model="test-model")
    if name == "anthropic":
        raw.update(type="message", role="assistant", usage={"input_tokens": 1, "output_tokens": 2})
    elif name == "openai":
        raw.update(object="response", created_at=0)
    else:
        raw.update(object="chat.completion", created=0)
        raw["choices"][0]["index"] = 0
    return raw


def install_transport(monkeypatch, names, asynchronous, respond):
    """Replace SDK constructors only in tests; each gets a fresh offline transport."""
    clients = []
    monkeypatch.setenv("OPENAI_API_KEY", "offline-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-anthropic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-router")

    def no_network(*args, **kwargs):
        raise AssertionError("unexpected external network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    for sdk_name in {"anthropic" if name == "anthropic" else "openai" for name in names}:
        sdk = pytest.importorskip(sdk_name)
        class_name = ("Async" if asynchronous else "") + (
            "Anthropic" if sdk_name == "anthropic" else "OpenAI"
        )
        original = getattr(sdk, class_name)

        def factory(_original=original, **options):
            http = httpx.AsyncClient if asynchronous else httpx.Client
            client = http(transport=httpx.MockTransport(respond))
            clients.append(client)
            return _original(http_client=client, **options)

        monkeypatch.setattr(sdk, class_name, factory)
    return clients


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sdk_exchange_creates_and_closes_clients(name, asynchronous, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=wire_reply(name, calls=len(requests) == 1))

    clients = install_transport(monkeypatch, [name], asynchronous, respond)
    provider = PROVIDERS[name]("test-model", timeout=12.5)

    def invoke(context, **kwargs):
        return (
            asyncio.run(provider.acall(context, **kwargs))
            if asynchronous
            else provider.call(context, **kwargs)
        )

    text, calls = invoke("OLD_CONTEXT", tools=[read])
    assert text == "  answer\n" and calls[0]["arguments"] == {"path": "a.py"}
    assert invoke("NEW_CONTEXT", tool_results=[{"request": calls[0], "content": "found"}]) == (
        "  answer\n",
        [],
    )
    assert len(clients) == len(requests) == 2
    assert all(client.is_closed for client in clients)
    body = json.loads(requests[1].content)
    assert "NEW_CONTEXT" in json.dumps(body) and "OLD_CONTEXT" not in json.dumps(body)
    assert "found" in json.dumps(body) and "tools" not in body
    assert requests[0].extensions["timeout"]["read"] == 12.5
    if name == "chat_completions":
        assert str(requests[0].url) == "https://openrouter.ai/api/v1/chat/completions"
        assert requests[0].headers["authorization"] == "Bearer offline-router"


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_sdk_rate_limit_is_not_retried_and_clients_close(name, asynchronous, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            429, json={"error": {"type": "rate_limit_error", "message": "offline"}}
        )

    clients = install_transport(monkeypatch, [name], asynchronous, respond)
    provider = PROVIDERS[name]("test-model")
    with pytest.raises(ProviderError):
        asyncio.run(provider.acall("input")) if asynchronous else provider.call("input")
    assert len(requests) == 1
    assert clients[0].is_closed


@pytest.mark.parametrize("name", PROVIDERS)
def test_async_cancellation_closes_real_sdk_client(name, monkeypatch):
    async def scenario():
        started = asyncio.Event()

        async def respond(request):
            started.set()
            await asyncio.Event().wait()

        clients = install_transport(monkeypatch, [name], True, respond)
        task = asyncio.create_task(PROVIDERS[name]("test-model").acall("input"))
        try:
            await asyncio.wait_for(started.wait(), 2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert clients[0].is_closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "source,target", [("openai", "anthropic"), ("anthropic", "chat_completions")]
)
def test_session_switches_provider_using_real_sdks(source, target, monkeypatch):
    requests, effects = [], []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json=wire_reply(source if len(requests) == 1 else target, calls=len(requests) == 1)
        )

    clients = install_transport(monkeypatch, [source, target], True, respond)

    def read(path: str):
        effects.append(path)
        return "file content"

    async def scenario():
        session = Session(provider=PROVIDERS[source]("test-model"), tools=[read])
        await session.acall("OLD_CONTEXT")
        await session.resolve_pending()
        assert await session.acall("NEW_CONTEXT", provider=PROVIDERS[target]("test-model")) == (
            "  answer\n",
            [],
        )
        assert effects == ["a.py"] and not session.ready_results

    asyncio.run(scenario())
    assert all(client.is_closed for client in clients)
    assert "file content" in requests[1].content.decode()


@pytest.mark.skipif(sys.version_info >= (3, 15), reason="LiteLLM Python range")
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length", "rate_limit"])
def test_litellm_real_sdk_uses_offline_transport(monkeypatch, asynchronous, finish_reason):
    from slick.providers import LiteLLMAPI, ProviderError

    blocked = []

    def no_network(*args, **kwargs):
        blocked.append(True)
        raise AssertionError("unexpected external network access")

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    sdk = pytest.importorskip("litellm")
    monkeypatch.setattr(sdk, "telemetry", False)
    # Isolate SDK-owned client caching between transports/event loops.
    monkeypatch.setattr(sdk, "in_memory_llm_clients_cache", type(sdk.in_memory_llm_clients_cache)())
    requests = []
    reply = {
        "id": "chatcmpl-offline",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "private-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": "  answer\n",
                },
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }

    def respond(request):
        requests.append(request)
        if finish_reason == "rate_limit":
            return httpx.Response(
                429,
                json={"error": {"message": "offline rate limit", "type": "rate_limit_error"}},
                headers={"retry-after": "0"},
            )
        return httpx.Response(200, json=reply)

    provider_instance = LiteLLMAPI(
        "openai/private-model",
        api_base="http://offline.test/v1",
        api_key="offline",
        timeout=12.5,
    )

    transport = httpx.MockTransport(respond)
    if asynchronous:

        async def run():
            async with httpx.AsyncClient(transport=transport) as http_client:
                monkeypatch.setattr(sdk, "aclient_session", http_client)
                if finish_reason != "rate_limit":
                    assert await provider_instance.acall("prompt\n") == ("  answer\n", [])
                else:
                    with pytest.raises(ProviderError):
                        await provider_instance.acall("prompt\n")
                assert not http_client.is_closed

        asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as http_client:
            monkeypatch.setattr(sdk, "client_session", http_client)
            if finish_reason != "rate_limit":
                assert provider_instance.call("prompt\n") == ("  answer\n", [])
            else:
                with pytest.raises(ProviderError):
                    provider_instance.call("prompt\n")
            assert not http_client.is_closed

    assert len(requests) == 1
    assert requests[0].url == "http://offline.test/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == "private-model"
    assert body["messages"] == [{"role": "user", "content": "prompt\n"}]
    assert requests[0].extensions["timeout"]["read"] == 12.5
    assert not blocked

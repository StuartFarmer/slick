"""Real optional SDK serialization/parsing against an offline HTTP transport."""

import asyncio
import json
import socket
import sys

import pytest

from slick.providers import AnthropicAPI, OpenAIAPI

httpx = pytest.importorskip("httpx")


@pytest.mark.parametrize("source,target", [("openai", "anthropic"), ("anthropic", "chat")])
def test_session_switches_provider_with_real_sdk_transports(source, target, monkeypatch):
    from test_tool_providers import reply

    from slick import Session
    from slick.providers import OpenRouterAPI

    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    effects, captured = [], []

    def no_network(*args, **kwargs):
        raise AssertionError("unexpected network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)

    def read(path: str) -> str:
        """Read a file."""
        effects.append(path)
        return "actual file content"

    def respond(request):
        captured.append(json.loads(request.content))
        name = source if len(captured) == 1 else target
        raw = reply("chat_completions" if name == "chat" else name, calls=len(captured) == 1)
        raw.update(id="offline", model="test-model")
        if name == "anthropic":
            raw.update(
                type="message", role="assistant", usage={"input_tokens": 1, "output_tokens": 2}
            )
        elif name == "openai":
            raw.update(object="response", created_at=0)
        else:
            raw.update(object="chat.completion", created=0)
            raw["choices"][0]["index"] = 0
        return httpx.Response(200, json=raw)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            providers = {
                "openai": OpenAIAPI(
                    "test-model",
                    async_client=openai.AsyncOpenAI(api_key="offline", http_client=client),
                ),
                "anthropic": AnthropicAPI(
                    "test-model",
                    async_client=anthropic.AsyncAnthropic(api_key="offline", http_client=client),
                ),
                "chat": OpenRouterAPI(
                    "test-model",
                    api_key="offline",
                    async_client=openai.AsyncOpenAI(api_key="offline", http_client=client),
                ),
            }
            session = Session(provider=providers[source], tools=[read])
            await session.acall("OLD_CONTEXT")
            await session.resolve_pending()
            assert await session.acall("NEW_CONTEXT", provider=providers[target]) == (
                "  answer\n",
                [],
            )
            assert effects == ["a.py"]
            assert not session.ready_results and not client.is_closed

    asyncio.run(scenario())
    assert len(captured) == 2
    assert "OLD_CONTEXT" not in json.dumps(captured[1])
    assert "actual file content" in json.dumps(captured[1])
    if target == "anthropic":
        assert captured[1]["messages"][-1]["content"][0]["type"] == "tool_result"
    else:
        assert captured[1]["messages"][-1]["role"] == "tool"


@pytest.mark.parametrize("name", ["openai", "anthropic", "openrouter", "litellm"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sdk_tool_exchange_with_fresh_provider(name, asynchronous, monkeypatch):
    from test_tool_providers import read, reply

    from slick.providers import LiteLLMAPI, OpenRouterAPI

    def no_network(*args, **kwargs):
        raise AssertionError("unexpected external network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    format = name if name in {"openai", "anthropic"} else "chat_completions"
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        raw = reply(format, calls=len(captured) == 1)
        raw.update(id="offline", model="test-model")
        if format == "chat_completions":
            raw.update(object="chat.completion", created=0)
            raw["choices"][0]["index"] = 0
        elif format == "anthropic":
            raw.update(
                type="message", role="assistant", usage={"input_tokens": 1, "output_tokens": 2}
            )
        else:
            raw.update(object="response", created_at=0)
        return httpx.Response(200, json=raw)

    if name == "litellm":
        if sys.version_info >= (3, 15):
            pytest.skip("LiteLLM Python range")
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        sdk = pytest.importorskip("litellm")
        monkeypatch.setattr(sdk, "telemetry", False)
        monkeypatch.setattr(
            sdk, "in_memory_llm_clients_cache", type(sdk.in_memory_llm_clients_cache)()
        )
    else:
        sdk = pytest.importorskip("anthropic" if name == "anthropic" else "openai")

    def build(http_client):
        if name == "litellm":
            monkeypatch.setattr(
                sdk, "aclient_session" if asynchronous else "client_session", http_client
            )
            return LiteLLMAPI(
                "openai/test-model", api_base="http://offline.test/v1", api_key="offline"
            )
        client_class = ("Async" if asynchronous else "") + (
            "Anthropic" if name == "anthropic" else "OpenAI"
        )
        client = getattr(sdk, client_class)(api_key="offline", http_client=http_client)
        kwargs = {"async_client" if asynchronous else "client": client}
        if name == "openrouter":
            kwargs["api_key"] = "offline"
        cls = {"openai": OpenAIAPI, "anthropic": AnthropicAPI, "openrouter": OpenRouterAPI}[name]
        return cls("test-model", **kwargs)

    transport = httpx.MockTransport(respond)
    if asynchronous:

        async def run():
            async with httpx.AsyncClient(transport=transport) as http_client:
                _, requests = await build(http_client).acall("OLD_CONTEXT", tools=[read])
                results = json.loads(json.dumps([{"request": requests[0], "content": "found"}]))
                assert await build(http_client).acall("NEW_CONTEXT", tool_results=results) == (
                    "  answer\n",
                    [],
                )
                assert not http_client.is_closed

        asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as http_client:
            _, requests = build(http_client).call("OLD_CONTEXT", tools=[read])
            results = json.loads(json.dumps([{"request": requests[0], "content": "found"}]))
            assert build(http_client).call("NEW_CONTEXT", tool_results=results) == (
                "  answer\n",
                [],
            )
            assert not http_client.is_closed
    assert len(captured) == 2
    assert "OLD_CONTEXT" not in json.dumps(captured[1])
    assert "NEW_CONTEXT" in json.dumps(captured[1])
    assert "found" in json.dumps(captured[1])
    assert "tools" not in captured[1]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["stop", "length", "rate_limit"])
def test_openrouter_sdk_uses_fixed_endpoint_and_explicit_credentials(
    monkeypatch, asynchronous, outcome
):
    from slick.providers import OpenRouterAPI, ProviderError

    sdk = pytest.importorskip("openai")
    requests = []

    def no_network(*args, **kwargs):
        raise AssertionError("unexpected external network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")

    def respond(request):
        requests.append(request)
        if outcome == "rate_limit":
            return httpx.Response(
                429, json={"error": {"message": "offline"}}, headers={"retry-after": "0"}
            )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "vendor/model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": outcome,
                        "message": {
                            "role": "assistant",
                            "content": "  answer\n",
                        },
                    }
                ],
            },
        )

    options = {"model": "vendor/model", "api_key": "explicit-router-key", "timeout": 12.5}
    transport = httpx.MockTransport(respond)
    if asynchronous:

        async def run():
            async with httpx.AsyncClient(transport=transport) as http_client:
                async with sdk.AsyncOpenAI(
                    api_key="wrong-client-key",
                    base_url="http://wrong.test/v1",
                    http_client=http_client,
                ) as client:
                    provider = OpenRouterAPI(**options, async_client=client)
                    if outcome == "stop":
                        assert await provider.acall("prompt\n") == ("  answer\n", [])
                    else:
                        with pytest.raises(ProviderError):
                            await provider.acall("prompt\n")
                    assert not http_client.is_closed

        asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as http_client:
            with sdk.OpenAI(
                api_key="wrong-client-key",
                base_url="http://wrong.test/v1",
                http_client=http_client,
            ) as client:
                provider = OpenRouterAPI(**options, client=client)
                if outcome == "stop":
                    assert provider.call("prompt\n") == ("  answer\n", [])
                else:
                    with pytest.raises(ProviderError):
                        provider.call("prompt\n")
                assert not http_client.is_closed
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer explicit-router-key"
    assert request.extensions["timeout"]["read"] == 12.5
    assert json.loads(request.content) == {
        "model": "vendor/model",
        "messages": [{"role": "user", "content": "prompt\n"}],
        "max_tokens": 2048,
        "stream": False,
        "n": 1,
    }


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
                if finish_reason == "stop":
                    assert await provider_instance.acall("prompt\n") == ("  answer\n", [])
                else:
                    with pytest.raises(ProviderError):
                        await provider_instance.acall("prompt\n")
                assert not http_client.is_closed

        asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as http_client:
            monkeypatch.setattr(sdk, "client_session", http_client)
            if finish_reason == "stop":
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


CASES = [
    (
        "openai",
        "OpenAI",
        OpenAIAPI,
        "/v1/responses",
        {"model": "test-model", "input": "prompt\n", "max_output_tokens": 123, "store": False},
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "model": "test-model",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "  answer\n", "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "temperature": 1.0,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "metadata": {},
            "usage": {
                "input_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 5,
            },
        },
    ),
    (
        "anthropic",
        "Anthropic",
        AnthropicAPI,
        "/v1/messages",
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "prompt\n"}],
            "max_tokens": 123,
        },
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [{"type": "text", "text": "  answer\n"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        },
    ),
]


@pytest.mark.parametrize("provider,sdk_class,provider_class,path,body,reply", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sdk_request_and_response_leave_injected_transport_open(
    provider, sdk_class, provider_class, path, body, reply, asynchronous
):
    sdk = pytest.importorskip(provider)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=reply)

    transport = httpx.MockTransport(respond)
    options = {"timeout": 12.5, "max_output_tokens": 123}
    if asynchronous:

        async def run():
            async with httpx.AsyncClient(transport=transport) as http_client:
                client = getattr(sdk, "Async" + sdk_class)(
                    api_key="offline-test-key", http_client=http_client
                )
                provider_instance = provider_class("test-model", async_client=client, **options)
                assert await provider_instance.acall("prompt\n") == ("  answer\n", [])
                assert not http_client.is_closed
                assert not client.is_closed()

        asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as http_client:
            client = getattr(sdk, sdk_class)(api_key="offline-test-key", http_client=http_client)
            provider_instance = provider_class("test-model", client=client, **options)
            assert provider_instance.call("prompt\n") == ("  answer\n", [])
            assert not http_client.is_closed
            assert not client.is_closed()

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == path
    assert json.loads(requests[0].content) == body
    assert requests[0].extensions["timeout"]["read"] == 12.5


@pytest.mark.parametrize("provider,sdk_class,provider_class,path,body,reply", CASES)
def test_real_sdk_native_turns_replay_original_call_and_correlated_result(
    provider, sdk_class, provider_class, path, body, reply
):
    from copy import deepcopy

    sdk = pytest.importorskip(provider)
    first = deepcopy(reply)
    if provider == "openai":
        first["output"] = [
            {
                "type": "function_call",
                "id": "fc1",
                "call_id": "call_1",
                "name": "read",
                "arguments": '{"key":"intro"}',
                "status": "completed",
            },
        ]
    else:
        first["stop_reason"] = "tool_use"
        first["content"] = [
            {"type": "tool_use", "id": "call_1", "name": "read", "input": {"key": "intro"}},
        ]
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=first if len(requests) == 1 else reply)

    def read(key: str) -> str:
        """Read a document."""
        raise AssertionError("provider executed a tool")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
            client = getattr(sdk, "Async" + sdk_class)(api_key="offline", http_client=http_client)
            provider_instance = provider_class("test-model", async_client=client, timeout=12.5)
            text, calls = await provider_instance.acall("read intro", tools=[read])
            assert calls[0]["arguments"] == {"key": "intro"}
            final = await provider_instance.acall(
                "New context",
                tools=[read],
                tool_results=[{"request": calls[0], "content": "Introduction"}],
            )
            assert final == ("  answer\n", [])
            assert not http_client.is_closed and not client.is_closed()

    asyncio.run(run())
    assert len(requests) == 2
    assert all(request.url.path == path for request in requests)
    assert all(request.extensions["timeout"]["read"] == 12.5 for request in requests)
    first_body, second = (json.loads(request.content) for request in requests)
    assert "instructions" not in first_body and "system" not in first_body
    if provider == "openai":
        assert first_body["tools"][0]["strict"] is False
        assert second["input"][-1] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "Introduction",
        }
        calls = [item for item in second["input"] if item.get("type") == "function_call"]
        assert len(calls) == 1 and calls[0]["call_id"] == "call_1"
        assert json.loads(calls[0]["arguments"]) == {"key": "intro"}
    else:
        assert "input_schema" in first_body["tools"][0]
        assert second["messages"][-1] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "Introduction",
                    "is_error": False,
                },
            ],
        }
        calls = [item for item in second["messages"][-2]["content"] if item["type"] == "tool_use"]
        assert len(calls) == 1 and calls[0]["id"] == "call_1"
        assert calls[0]["input"] == {"key": "intro"}

"""Direct OpenRouter requests use OpenRouter credentials and the optional OpenAI SDK."""

import asyncio
import sys

import pytest
from sdk_fakes import JsonNamespace as NS
from test_api_providers import Client

from slick.providers import ProviderError


class ChatClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat = NS(completions=self.responses)

    def with_options(self, **options):
        return ChatClient(self.reply, asynchronous=self.asynchronous, state=self.state, **options)


def reply(finish="stop", content="  answer\n", **message):
    return NS(
        choices=[NS(finish_reason=finish, message=NS(role="assistant", content=content, **message))]
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_openrouter_routes_and_authenticates_at_call_time(monkeypatch, asynchronous, owned):
    from slick.providers import OpenRouterAPI

    seen = []

    def factory(**options):
        client = ChatClient(reply(), asynchronous=asynchronous, **options)
        seen.append(client)
        return client

    monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=factory, AsyncOpenAI=factory))
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    injected = {} if owned else {"async_client" if asynchronous else "client": factory()}
    provider = OpenRouterAPI("vendor/model", timeout=12.5, max_output_tokens=123, **injected)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    answer = asyncio.run(provider.acall("input\n")) if asynchronous else provider.call("input\n")
    assert answer == ("  answer\n", [])
    assert len(seen) == 1
    state = seen[0].state
    assert len(state["requests"]) == 1
    options, request = state["requests"][0]
    assert request == {
        "model": "vendor/model",
        "messages": [{"role": "user", "content": "input\n"}],
        "max_tokens": 123,
        "stream": False,
        "n": 1,
    }
    assert options == {
        "timeout": 12.5,
        "max_retries": 0,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "router-key",
    }
    assert state["closed"] == int(owned)
    assert provider.identity() == {
        "provider": "openrouter",
        "model": "vendor/model",
        "timeout": 12.5,
        "max_output_tokens": 123,
        "max_retries": 0,
    }
    assert not hasattr(provider, "aturn")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "response",
    [
        reply("length"),
        reply("content_filter"),
        reply(content=None),
        reply(tool_calls=[object()]),
        reply(function_call={"name": "read"}),
        reply(refusal="no"),
        NS(choices=[]),
        NS(choices=[reply().choices[0]] * 2),
        NS(),
    ],
)
def test_openrouter_rejects_incomplete_or_nontext_results(response, asynchronous):
    from slick.providers import OpenRouterAPI

    client = ChatClient(response, asynchronous=asynchronous)
    provider = OpenRouterAPI(
        "vendor/model", api_key="offline", **{"async_client" if asynchronous else "client": client}
    )
    with pytest.raises(ProviderError):
        asyncio.run(provider.acall("prompt")) if asynchronous else provider.call("prompt")
    assert len(client.state["requests"]) == 1


@pytest.mark.parametrize("injected", [False, True])
def test_openrouter_requires_its_own_key_for_all_clients(monkeypatch, injected):
    from slick.providers import OpenRouterAPI

    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = ChatClient(reply(), api_key="wrong-client-key")
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        OpenRouterAPI("vendor/model", client=client if injected else None).call("prompt")
    assert not client.state["requests"]


def test_openrouter_explicit_key_is_private_and_missing_sdk_is_actionable(monkeypatch):
    from slick.providers import OpenRouterAPI

    provider = OpenRouterAPI("vendor/model", api_key="explicit-key")
    assert "explicit-key" not in repr(provider)
    assert "explicit-key" not in str(provider.identity())
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ProviderError, match=r"slick-ai\[openai\]"):
        provider.call("prompt")


def test_openrouter_async_cancellation_propagates():
    from slick.providers import OpenRouterAPI

    async def run():
        started = asyncio.Event()

        async def create(**request):
            started.set()
            await asyncio.Event().wait()

        client = NS(chat=NS(completions=NS(create=create)))
        client.with_options = lambda **options: client
        task = asyncio.create_task(
            OpenRouterAPI("vendor/model", api_key="offline", async_client=client).acall("prompt")
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

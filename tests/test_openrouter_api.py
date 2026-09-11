"""Direct OpenRouter requests use OpenRouter credentials and the optional OpenAI SDK."""

import asyncio
from types import SimpleNamespace as NS

import pytest
from test_api_providers import Client, install_sdk

from slick.providers import ProviderError


class ChatClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat = NS(completions=self.responses)


def reply(finish="stop", content="  answer\n", **message):
    return NS(
        choices=[NS(finish_reason=finish, message=NS(role="assistant", content=content, **message))]
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_openrouter_routes_and_authenticates_at_call_time(monkeypatch, asynchronous):
    from slick.providers import OpenRouterAPI

    seen = []

    def factory(**options):
        client = ChatClient(reply(), asynchronous=asynchronous, **options)
        seen.append(client)
        return client

    install_sdk(monkeypatch, "OpenAI", factory, asynchronous)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterAPI("vendor/model", timeout=12.5, max_output_tokens=123)
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
    assert state["closed"] == 1
    assert provider.identity() == {
        "provider": "openrouter",
        "model": "vendor/model",
    }


def test_openrouter_requires_its_own_key(monkeypatch):
    from slick.providers import OpenRouterAPI

    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = ChatClient(reply())
    install_sdk(monkeypatch, "OpenAI", lambda **kw: client)
    with pytest.raises(ProviderError) as raised:
        OpenRouterAPI("vendor/model").call("prompt")
    assert isinstance(raised.value.__cause__, KeyError)
    assert raised.value.__cause__.args == ("OPENROUTER_API_KEY",)
    assert not client.state["requests"]


def test_openrouter_explicit_key_is_private():
    from slick.providers import OpenRouterAPI

    provider = OpenRouterAPI("vendor/model", api_key="explicit-key")
    assert "explicit-key" not in repr(provider)
    assert "explicit-key" not in str(provider.identity())


def test_openrouter_async_cancellation_propagates(monkeypatch):
    from slick.providers import OpenRouterAPI

    async def run():
        started = asyncio.Event()

        async def create(**request):
            started.set()
            await asyncio.Event().wait()

        client = ChatClient(reply(), asynchronous=True)
        client.chat.completions.create = create
        install_sdk(monkeypatch, "OpenAI", lambda **kw: client, asynchronous=True)
        task = asyncio.create_task(OpenRouterAPI("vendor/model", api_key="offline").acall("prompt"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

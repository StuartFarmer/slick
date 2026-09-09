"""LiteLLM's text boundary, tested without provider access or the optional SDK."""

import asyncio
import sys
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from slick.providers import ProviderError

supported_python = pytest.mark.skipif(sys.version_info >= (3, 15), reason="LiteLLM Python range")


def response(content="  answer\n", finish="stop", **message_fields):
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": None,
        "function_call": None,
        "refusal": None,
    }
    message.update(message_fields)
    return NS(choices=[NS(finish_reason=finish, message=NS(**message))])


def install_sdk(monkeypatch, reply, requests):
    def completion(**request):
        requests.append(deepcopy(request))
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def acompletion(**request):
        return completion(**request)

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        NS(
            completion=completion,
            acompletion=acompletion,
        ),
    )


@supported_python
@pytest.mark.parametrize("asynchronous", [False, True])
def test_exact_text_options_and_configuration_are_preserved(monkeypatch, asynchronous):
    from slick.providers import LiteLLMAPI

    requests = []
    install_sdk(monkeypatch, response(), requests)
    options = {"temperature": 0.2, "response_format": {"type": "json_object"}}
    provider = LiteLLMAPI(
        "openai/private-model",
        api_base="http://localhost:8000/v1",
        api_key="offline",
        timeout=12.5,
        options=options,
    )
    options["response_format"]["type"] = "updated-option"
    result = asyncio.run(provider.acall("input\n")) if asynchronous else provider.call("input\n")
    assert result == ("  answer\n", [])
    assert requests == [
        {
            "model": "openai/private-model",
            "messages": [{"role": "user", "content": "input\n"}],
            "api_base": "http://localhost:8000/v1",
            "api_key": "offline",
            "timeout": 12.5,
            "num_retries": 0,
            "drop_params": False,
            "stream": False,
            "n": 1,
            "temperature": 0.2,
            "response_format": {"type": "updated-option"},
        }
    ]


@supported_python
def test_provider_resolves_missing_credentials_and_unknown_model(monkeypatch):
    from slick.providers import LiteLLMAPI

    requests = []
    install_sdk(monkeypatch, response("ok"), requests)
    assert LiteLLMAPI("unusual/model", max_retries=2).call("prompt") == ("ok", [])
    assert requests[0]["model"] == "unusual/model"
    assert requests[0]["num_retries"] == 2
    assert "api_base" not in requests[0] and "api_key" not in requests[0]
    assert "max_tokens" not in requests[0]


@supported_python
def test_options_use_normal_dictionary_reference_semantics(monkeypatch):
    from slick.providers import LiteLLMAPI

    seen = []

    def completion(**request):
        seen.append(request["response_format"]["type"])
        request["response_format"]["type"] = "changed by SDK"
        return response()

    monkeypatch.setitem(sys.modules, "litellm", NS(completion=completion))
    provider = LiteLLMAPI("openai/test", options={"response_format": {"type": "json_object"}})
    provider.call("one")
    provider.call("two")
    assert seen == ["json_object", "changed by SDK"]


@supported_python
@pytest.mark.parametrize("asynchronous", [False, True])
def test_provider_exception_is_chained_without_exposing_its_message(monkeypatch, asynchronous):
    from slick.providers import LiteLLMAPI

    error = RuntimeError("sensitive-provider-detail")
    install_sdk(monkeypatch, error, [])
    provider = LiteLLMAPI("openai/test")
    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.acall("prompt")) if asynchronous else provider.call("prompt")
    assert caught.value.__cause__ is error
    assert "sensitive-provider-detail" not in str(caught.value)


def test_representation_and_persistence_do_not_expose_credentials():
    from slick.providers import LiteLLMAPI

    provider = LiteLLMAPI(
        "openai/test", api_key="private-key", options={"secret": "private-option"}
    )
    assert "private-key" not in repr(provider) and "private-option" not in repr(provider)
    identity = provider.identity()
    assert "private-key" not in str(identity)
    assert "private-option" not in str(identity)


@supported_python
def test_async_cancellation_propagates_without_sync_fallback(monkeypatch):
    from slick.providers import LiteLLMAPI

    async def run():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def acompletion(**request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        def completion(**request):
            raise AssertionError("sync fallback")

        monkeypatch.setitem(
            sys.modules,
            "litellm",
            NS(
                completion=completion,
                acompletion=acompletion,
            ),
        )
        task = asyncio.create_task(LiteLLMAPI("openai/test").acall("prompt"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())

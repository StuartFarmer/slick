"""LiteLLM's text boundary, tested without provider access or the optional SDK."""

import asyncio
import sys
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from slick.providers import ProviderError

supported_python = pytest.mark.skipif(sys.version_info >= (3, 15), reason="LiteLLM Python range")


def response(content="  answer\n", finish="stop", **message_fields):
    message = {"content": content, "tool_calls": None, "function_call": None, "refusal": None}
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
    from slick.providers import LiteLLMGateway

    requests = []
    install_sdk(monkeypatch, response(), requests)
    options = {"temperature": 0.2, "response_format": {"type": "json_object"}}
    provider = LiteLLMGateway(
        "openai/private-model",
        api_base="http://localhost:8000/v1",
        api_key="offline",
        timeout=12.5,
        options=options,
    )
    options["response_format"]["type"] = "changed by caller"
    result = asyncio.run(provider.acall("input\n")) if asynchronous else provider.call("input\n")
    assert result == "  answer\n"
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
            "response_format": {"type": "json_object"},
        }
    ]


@supported_python
def test_provider_resolves_missing_credentials_and_unknown_model(monkeypatch):
    from slick.providers import LiteLLMGateway

    requests = []
    install_sdk(monkeypatch, response(""), requests)
    assert LiteLLMGateway("unusual/model", max_retries=2).call("prompt") == ""
    assert requests[0]["model"] == "unusual/model"
    assert requests[0]["num_retries"] == 2
    assert "api_base" not in requests[0] and "api_key" not in requests[0]
    assert "max_tokens" not in requests[0]


@supported_python
def test_sdk_mutation_does_not_change_the_next_request(monkeypatch):
    from slick.providers import LiteLLMGateway

    seen = []

    def completion(**request):
        seen.append(request["response_format"]["type"])
        request["response_format"]["type"] = "changed by SDK"
        return response()

    monkeypatch.setitem(sys.modules, "litellm", NS(completion=completion))
    provider = LiteLLMGateway("openai/test", options={"response_format": {"type": "json_object"}})
    provider.call("one")
    provider.call("two")
    assert seen == ["json_object", "json_object"]


@supported_python
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "reply",
    [
        response("partial", "length"),
        response("", "content_filter"),
        response(None, "tool_calls", tool_calls=[NS(id="call")]),
        response(None),
        response(refusal="refused"),
        response(tool_calls=[NS(id="call")]),
        response(function_call=NS(name="read")),
        NS(choices=[]),
        NS(choices=[response().choices[0], response().choices[0]]),
        NS(choices=[NS(finish_reason="stop", message=NS())]),
        NS(),
    ],
)
def test_only_final_text_is_accepted(monkeypatch, asynchronous, reply):
    from slick.providers import LiteLLMGateway

    install_sdk(monkeypatch, reply, [])
    provider = LiteLLMGateway("openai/test")
    with pytest.raises(ProviderError):
        asyncio.run(provider.acall("prompt")) if asynchronous else provider.call("prompt")


@supported_python
@pytest.mark.parametrize("asynchronous", [False, True])
def test_provider_exception_is_chained_without_exposing_its_message(monkeypatch, asynchronous):
    from slick.providers import LiteLLMGateway

    error = RuntimeError("sensitive-provider-detail")
    install_sdk(monkeypatch, error, [])
    provider = LiteLLMGateway("openai/test")
    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.acall("prompt")) if asynchronous else provider.call("prompt")
    assert caught.value.__cause__ is error
    assert "sensitive-provider-detail" not in str(caught.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"model": "  "},
        {"model": None},
        {"api_base": ""},
        {"api_key": 1},
        {"api_key": " "},
        {"timeout": True},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"max_retries": True},
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"options": []},
        {"options": {1: "bad"}},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    from slick.providers import LiteLLMGateway

    with pytest.raises(ValueError):
        LiteLLMGateway(**{"model": "openai/test", **kwargs})


@pytest.mark.parametrize(
    "key",
    [
        "model",
        "messages",
        "api_base",
        "base_url",
        "api_key",
        "timeout",
        "num_retries",
        "max_retries",
        "stream",
        "stream_options",
        "n",
        "drop_params",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "parallel_tool_calls",
        "input",
        "previous_response_id",
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "model_list",
        "router",
        "client",
        "acompletion",
    ],
)
def test_options_cannot_override_execution_contract(key):
    from slick.providers import LiteLLMGateway

    with pytest.raises(ValueError):
        LiteLLMGateway("openai/test", options={key: "private-value"})


@pytest.mark.parametrize("limit", ["max_tokens", "max_output_tokens", "max_completion_tokens"])
def test_chatgpt_rejects_limits_that_upstream_discards(limit):
    from slick.providers import LiteLLMGateway

    with pytest.raises(ValueError, match="chatgpt"):
        LiteLLMGateway("chatgpt/example", options={limit: 20})


def test_representation_and_persistence_do_not_expose_credentials():
    from slick import PromptError
    from slick.prompts import _digest
    from slick.providers import LiteLLMGateway

    provider = LiteLLMGateway(
        "openai/test", api_key="private-key", options={"secret": "private-option"}
    )
    assert "private-key" not in repr(provider) and "private-option" not in repr(provider)
    with pytest.raises(PromptError, match="identity"):
        _digest("prompt", provider)


@supported_python
def test_missing_sdk_and_broken_dependency_have_distinct_errors(monkeypatch):
    from slick.providers import _litellm as adapter

    for name, message in [("litellm", r"slick-ai\[litellm\]"), ("dependency", "import")]:
        error = ModuleNotFoundError("missing", name=name)

        def fail_import(module, error=error):
            raise error

        monkeypatch.setattr(adapter, "import_module", fail_import)
        with pytest.raises(ProviderError, match=message) as caught:
            adapter.LiteLLMGateway("openai/test").call("prompt")
        assert caught.value.__cause__ is error


def test_unsupported_python_fails_before_loading_sdk(monkeypatch):
    from slick.providers import _litellm as adapter

    monkeypatch.setattr(adapter, "sys", NS(version_info=(3, 15)))
    with pytest.raises(ProviderError, match=r"3\.10.*3\.15"):
        adapter.LiteLLMGateway("openai/test").call("prompt")


@supported_python
def test_async_cancellation_propagates_without_sync_fallback(monkeypatch):
    from slick.providers import LiteLLMGateway

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
        task = asyncio.create_task(LiteLLMGateway("openai/test").acall("prompt"))
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

"""Offline contract tests for the optional SDK adapters."""

import asyncio
import builtins
import importlib
import json
import sys
from types import SimpleNamespace as NS

import pytest


def provider_class(provider):
    module = importlib.import_module("slick.providers")
    return getattr(module, provider + "API")


def response(provider, text="  answer\n"):
    if provider == "OpenAI":
        return NS(
            status="completed",
            output=[
                NS(type="reasoning"),
                NS(type="message", status="completed", content=[NS(type="output_text", text=text)]),
            ],
        )
    return NS(stop_reason="end_turn", content=[NS(type="text", text=text)])


class Client:
    """SDK-shaped client that records requests and cleanup."""

    def __init__(self, reply, *, asynchronous=False, state=None, **options):
        self.reply = reply
        self.asynchronous = asynchronous
        self.state = state if state is not None else {"requests": [], "closed": 0, "entered": 0}
        self.options = options
        self.responses = self.messages = NS(create=self.acreate if asynchronous else self.create)

    def create(self, **request):
        self.state["requests"].append((self.options, request))
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply

    async def acreate(self, **request):
        return self.create(**request)

    def __enter__(self):
        self.state["entered"] += 1
        return self

    def __exit__(self, *args):
        self.state["closed"] += 1

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        self.__exit__(*args)


def invoke(provider_instance, asynchronous):
    return (
        asyncio.run(provider_instance.acall("input\n"))
        if asynchronous
        else provider_instance.call("input\n")
    )


def install_sdk(monkeypatch, provider, factory, asynchronous=False):
    name = ("Async" if asynchronous else "") + provider
    monkeypatch.setitem(sys.modules, provider.lower(), NS(**{name: factory}))


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_requests_create_and_close_their_own_clients(provider, asynchronous, monkeypatch):
    clients = []

    def factory(**options):
        client = Client(response(provider), asynchronous=asynchronous, **options)
        clients.append(client)
        return client

    install_sdk(monkeypatch, provider, factory, asynchronous)
    instance = provider_class(provider)(
        "explicit-model", timeout=12.5, max_output_tokens=123, max_retries=1
    )
    assert invoke(instance, asynchronous) == ("  answer\n", [])
    assert invoke(instance, asynchronous) == ("  answer\n", [])
    expected = {"model": "explicit-model"}
    if provider == "OpenAI":
        expected.update(input="input\n", max_output_tokens=123, store=False)
    else:
        expected.update(messages=[{"role": "user", "content": "input\n"}], max_tokens=123)
    assert len(clients) == 2
    for client in clients:
        assert client.state["requests"] == [({"timeout": 12.5, "max_retries": 1}, expected)]
        assert client.state["entered"] == client.state["closed"] == 1


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_sdk_errors_keep_native_cause_and_close_clients(provider, asynchronous, monkeypatch):
    from slick.providers import ProviderError

    failure = RuntimeError("provider failed")
    client = Client(failure, asynchronous=asynchronous)
    install_sdk(monkeypatch, provider, lambda **kw: client, asynchronous)
    with pytest.raises(ProviderError) as caught:
        invoke(provider_class(provider)("model"), asynchronous)
    assert caught.value.__cause__ is failure
    assert client.state["closed"] == 1


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
def test_cancellation_propagates_and_closes_client(provider, monkeypatch):
    client = Client(asyncio.CancelledError(), asynchronous=True)
    install_sdk(monkeypatch, provider, lambda **kw: client, asynchronous=True)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(provider_class(provider)("model").acall("input"))
    assert client.state["closed"] == 1


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
def test_empty_text_is_preserved_and_multiple_blocks_are_joined(provider):
    reply = response(provider, "")
    instance = provider_class(provider)("model")
    assert instance._decode(reply) == ("", [])
    blocks = reply.output[-1].content if provider == "OpenAI" else reply.content
    kind = "output_text" if provider == "OpenAI" else "text"
    blocks.extend([NS(type=kind, text=" left "), NS(type=kind, text="right\n")])
    assert instance._decode(reply) == (" left right\n", [])


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
def test_configuration_is_passed_to_the_sdk_without_coercion(provider, monkeypatch):
    seen = []

    def factory(**options):
        seen.append(options)
        return Client(response(provider))

    install_sdk(monkeypatch, provider, factory)
    instance = provider_class(provider)("model", timeout="sdk-value")
    assert instance.call("input") == ("  answer\n", [])
    instance.timeout = "updated-value"
    instance.call("input")
    assert [options["timeout"] for options in seen] == ["sdk-value", "updated-value"]


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic", "OpenRouter", "LiteLLM"])
def test_configuration_preserves_model_whitespace_and_integer_timeout(provider):
    instance = provider_class(provider)(" model ", timeout=12)
    assert instance.model == " model "
    assert type(instance.timeout) is int
    assert instance.timeout == 12


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
def test_identity_is_json_configuration_without_client_credentials(provider):
    provider_instance = provider_class(provider)("explicit-model")
    assert json.loads(json.dumps(provider_instance.identity())) == {
        "provider": provider.lower(),
        "model": "explicit-model",
    }


def test_importing_slick_does_not_import_optional_sdks(monkeypatch):
    original_import = builtins.__import__
    attempted = []

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"openai", "anthropic"}:
            attempted.append(name)
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    importlib.reload(importlib.import_module("slick.providers"))
    assert attempted == []

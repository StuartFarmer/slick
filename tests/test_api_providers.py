"""Offline contract tests for the optional SDK adapters."""

import asyncio
import builtins
import importlib
import json
import sys

import pytest
from sdk_fakes import JsonNamespace as NS


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
    """SDK-shaped client: options copies share transport ownership."""

    def __init__(self, reply, *, asynchronous=False, state=None, **options):
        self.reply = reply
        self.asynchronous = asynchronous
        self.state = state if state is not None else {"requests": [], "closed": 0, "entered": 0}
        self.options = options
        self.responses = self.messages = NS(create=self.acreate if asynchronous else self.create)

    def with_options(self, **options):
        return Client(self.reply, asynchronous=self.asynchronous, state=self.state, **options)

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


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_requests_return_exact_text_and_respect_client_ownership(
    provider, asynchronous, owned, monkeypatch
):
    clients = []

    def factory(**options):
        client = Client(response(provider), asynchronous=asynchronous, **options)
        clients.append(client)
        return client

    if owned:
        sdk_name = ("Async" if asynchronous else "") + provider
        monkeypatch.setitem(sys.modules, provider.lower(), NS(**{sdk_name: factory}))
        injected = {}
    else:
        client = factory()
        injected = {"async_client" if asynchronous else "client": client}
    provider_instance = provider_class(provider)(
        "explicit-model", timeout=12.5, max_output_tokens=123, max_retries=1, **injected
    )
    assert invoke(provider_instance, asynchronous) == ("  answer\n", [])
    assert invoke(provider_instance, asynchronous) == ("  answer\n", [])
    expected = {"model": "explicit-model"}
    if provider == "OpenAI":
        expected.update(input="input\n", max_output_tokens=123, store=False)
    else:
        expected.update(messages=[{"role": "user", "content": "input\n"}], max_tokens=123)
    assert len(clients) == (2 if owned else 1)
    assert sum(len(c.state["requests"]) for c in clients) == 2
    for client in clients:
        assert all(
            request == ({"timeout": 12.5, "max_retries": 1}, expected)
            for request in client.state["requests"]
        )
        assert client.state["entered"] == int(owned)
        assert client.state["closed"] == int(owned)


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_sdk_errors_keep_native_cause_and_owned_clients_close(
    provider, asynchronous, owned, monkeypatch
):
    from slick.providers import ProviderError

    native_error = RuntimeError("provider failed")
    client = Client(native_error, asynchronous=asynchronous)
    if owned:
        sdk_name = ("Async" if asynchronous else "") + provider
        monkeypatch.setitem(sys.modules, provider.lower(), NS(**{sdk_name: lambda **kw: client}))
        injected = {}
    else:
        injected = {"async_client" if asynchronous else "client": client}
    with pytest.raises(ProviderError) as caught:
        invoke(provider_class(provider)("model", **injected), asynchronous)
    assert caught.value.__cause__ is native_error
    assert client.state["closed"] == int(owned)


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
def test_cancellation_propagates_and_closes_owned_client(provider, monkeypatch):
    client = Client(asyncio.CancelledError(), asynchronous=True)
    monkeypatch.setitem(
        sys.modules,
        provider.lower(),
        NS(
            **{
                "Async" + provider: lambda **kw: client,
            }
        ),
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(provider_class(provider)("model").acall("input"))
    assert client.state["closed"] == 1


BAD_RESPONSES = [
    ("OpenAI", NS(status="incomplete", output=[])),
    ("OpenAI", NS(status="failed", output=[])),
    ("OpenAI", NS(status="completed", output=[NS(type="function_call")])),
    (
        "OpenAI",
        NS(
            status="completed",
            output=[
                NS(
                    type="message",
                    status="incomplete",
                    content=[NS(type="output_text", text="partial")],
                )
            ],
        ),
    ),
    (
        "OpenAI",
        NS(
            status="completed",
            output=[
                NS(type="message", status="completed", content=[NS(type="refusal", refusal="no")])
            ],
        ),
    ),
    ("OpenAI", NS(status="completed", output=[])),
    ("Anthropic", NS(stop_reason="max_tokens", content=[NS(type="text", text="partial")])),
    ("Anthropic", NS(stop_reason="refusal", content=[NS(type="text", text="no")])),
    ("Anthropic", NS(stop_reason="tool_use", content=[NS(type="tool_use")])),
    ("Anthropic", NS(stop_reason="pause_turn", content=[NS(type="text", text="partial")])),
    ("Anthropic", NS(stop_reason="end_turn", content=[NS(type="tool_use")])),
    ("Anthropic", NS(stop_reason="end_turn", content=[])),
]


@pytest.mark.parametrize("provider,reply", BAD_RESPONSES)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_refused_incomplete_or_nontext_outputs_are_errors(provider, reply, asynchronous):
    from slick.providers import ProviderError

    client = Client(reply, asynchronous=asynchronous)
    provider_instance = provider_class(provider)(
        "model", **{"async_client" if asynchronous else "client": client}
    )
    with pytest.raises(ProviderError):
        invoke(provider_instance, asynchronous)


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
def test_empty_text_is_preserved_and_multiple_blocks_are_joined(provider):
    reply = response(provider, "")
    provider_instance = provider_class(provider)("model", client=Client(reply))
    assert provider_instance.call("input") == ("", [])
    blocks = reply.output[-1].content if provider == "OpenAI" else reply.content
    kind = "output_text" if provider == "OpenAI" else "text"
    blocks.extend([NS(type=kind, text=" left "), NS(type=kind, text="right\n")])
    assert provider_instance.call("input") == (" left right\n", [])


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
@pytest.mark.parametrize(
    "options",
    [
        {"model": ""},
        {"model": "  "},
        {"model": None},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"timeout": True},
        {"timeout": "60"},
        {"max_output_tokens": 0},
        {"max_output_tokens": 1.5},
        {"max_output_tokens": True},
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"max_retries": True},
    ],
)
def test_invalid_configuration_rejected_at_construction(provider, options):
    with pytest.raises((TypeError, ValueError)):
        provider_class(provider)(**{"model": "model", **options})


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
def test_identity_is_json_configuration_without_client_credentials(provider):
    provider_instance = provider_class(provider)("explicit-model", client=NS(api_key="secret"))
    assert json.loads(json.dumps(provider_instance.identity())) == {
        "provider": provider.lower(),
        "model": "explicit-model",
        "timeout": 60,
        "max_output_tokens": 2048,
        "max_retries": 0,
    }


@pytest.mark.parametrize("provider", ["OpenAI", "Anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_sdk_imports_are_lazy_and_missing_extra_has_actionable_error(
    provider, asynchronous, monkeypatch
):
    from slick.providers import ProviderError

    # Missing SDKs must not prevent constructing a provider or importing slick.
    monkeypatch.setitem(sys.modules, provider.lower(), None)
    provider_instance = provider_class(provider)("model")
    with pytest.raises(ProviderError, match=rf"slick-ai\[{provider.lower()}\]") as caught:
        invoke(provider_instance, asynchronous)
    assert isinstance(caught.value.__cause__, ImportError)


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

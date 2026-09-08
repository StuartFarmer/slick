"""SDK-free native codecs and one-request provider boundaries."""

import asyncio
import importlib
from copy import deepcopy

import pytest
from test_api_providers import Client

from slick.providers import AnthropicAPI, OpenAIAPI, ProviderError
from slick.tools import prepare_tools
from slick.turns import ModelTurn, ToolCall, ToolResult, UserMessage


def read(key: str, limit: int = 2) -> str:
    """Read a document."""
    raise AssertionError("A native turn must never execute a tool")


def codec(provider):
    return importlib.import_module(f"slick._{provider}_turns")


def reply(provider, *, calls=True):
    if provider == "openai":
        output = [
            {"type": "reasoning", "id": "r1", "summary": [], "encrypted_content": "opaque"},
            {
                "type": "message",
                "id": "msg1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Looking. ", "annotations": []}],
            },
        ]
        if calls:
            output += [
                {
                    "type": "function_call",
                    "id": "fc1",
                    "call_id": "a",
                    "name": "read",
                    "arguments": '{"key":"x"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc2",
                    "call_id": "b",
                    "name": "unknown",
                    "arguments": "{}",
                    "status": "completed",
                },
            ]
        return {
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    content = [
        {"type": "thinking", "thinking": "private", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "Looking. "},
    ]
    if calls:
        content += [
            {"type": "tool_use", "id": "a", "name": "read", "input": {"key": "x"}},
            {"type": "tool_use", "id": "b", "name": "unknown", "input": {}},
        ]
    return {
        "stop_reason": "tool_use" if calls else "end_turn",
        "content": content,
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }


class JsonResponse:
    def __init__(self, data):
        self.data = data

    def model_dump(self, *, mode):
        assert mode == "json"
        return self.data


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_tool_definitions_preserve_optional_arguments_and_own_schemas(provider):
    prepared = prepare_tools([read])
    definitions = codec(provider).tool_definitions(prepared)
    schema = {
        "additionalProperties": False,
        "properties": {
            "key": {"title": "Key", "type": "string"},
            "limit": {"default": 2, "title": "Limit", "type": "integer"},
        },
        "required": ["key"],
        "title": "readArguments",
        "type": "object",
    }
    expected = {"name": "read", "description": "Read a document."}
    expected.update(
        {"parameters": schema, "type": "function", "strict": False}
        if provider == "openai"
        else {"input_schema": schema}
    )
    assert definitions == [expected]
    definitions[0]["parameters" if provider == "openai" else "input_schema"]["required"].clear()
    assert prepared["read"].parameters["required"] == ["key"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_decode_and_replay_keep_provider_items_and_multiple_correlated_calls(provider):
    original = reply(provider)
    saved = deepcopy(original)
    turn = codec(provider).decode_turn(original, model="test-model")
    assert (turn.provider, turn.model, turn.text, turn.stop_reason) == (
        provider,
        "test-model",
        "Looking. ",
        "tool_calls",
    )
    assert turn.tool_calls == [ToolCall("a", "read", {"key": "x"}), ToolCall("b", "unknown", {})]
    assert (turn.input_tokens, turn.output_tokens) == (2, 3)
    history = [
        UserMessage("read"),
        turn,
        ToolResult("a", "contents"),
        ToolResult("b", "error", True),
    ]
    encoded = codec(provider).encode_history(history)
    if provider == "openai":
        assert encoded == [
            {"role": "user", "content": "read"},
            *saved["output"],
            {"type": "function_call_output", "call_id": "a", "output": "contents"},
            {"type": "function_call_output", "call_id": "b", "output": "error"},
        ]
        encoded[1]["summary"].append({"type": "summary_text", "text": "changed"})
    else:
        assert encoded == [
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": saved["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "a",
                        "content": "contents",
                        "is_error": False,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "b",
                        "content": "error",
                        "is_error": True,
                    },
                ],
            },
        ]
        encoded[1]["content"][0]["signature"] = "changed"
    assert turn.items == saved["output" if provider == "openai" else "content"]
    turn.items.clear()
    assert original == saved


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', "[]", '{"x":NaN}', "bad", {"x": float("inf")}])
def test_malformed_arguments_are_recoverable_and_original_items_retained(provider, raw):
    response = reply(provider)
    items = response["output" if provider == "openai" else "content"]
    items[-1]["arguments" if provider == "openai" else "input"] = raw
    turn = codec(provider).decode_turn(response, model="test-model")
    assert turn.tool_calls[-1].arguments is None
    assert turn.tool_calls[-1].argument_error
    assert turn.items[-1] == items[-1]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "problem",
    [
        "empty_id",
        "duplicate_id",
        "missing_name",
        "missing_arguments",
        "unknown",
        "incomplete",
        "refusal",
        "empty",
    ],
)
def test_invalid_envelopes_fail_without_partial_calls(provider, problem):
    response = reply(provider)
    items = response["output" if provider == "openai" else "content"]
    id_key = "call_id" if provider == "openai" else "id"
    if problem == "empty_id":
        items[-1][id_key] = ""
    elif problem == "duplicate_id":
        items[-1][id_key] = "a"
    elif problem == "missing_name":
        del items[-1]["name"]
    elif problem == "missing_arguments":
        del items[-1]["arguments" if provider == "openai" else "input"]
    elif problem == "unknown":
        items.append({"type": "web_search_call"})
    elif problem == "incomplete":
        response["status" if provider == "openai" else "stop_reason"] = (
            "incomplete" if provider == "openai" else "max_tokens"
        )
    elif problem == "refusal":
        if provider == "openai":
            items[1]["content"].append({"type": "refusal", "refusal": "no"})
        else:
            response["stop_reason"] = "refusal"
    else:
        items.clear()
    with pytest.raises(ValueError):
        codec(provider).decode_turn(response, model="test-model")


@pytest.mark.parametrize("status", ["in_progress", "incomplete"])
def test_openai_rejects_incomplete_tool_items(status):
    response = reply("openai")
    response["output"][-1]["status"] = status
    with pytest.raises(ValueError, match="complete"):
        codec("openai").decode_turn(response, model="test-model")


@pytest.mark.parametrize(
    "stop,calls",
    [("end_turn", True), ("tool_use", False), ("pause_turn", False), ("stop_sequence", False)],
)
def test_anthropic_rejects_inconsistent_or_unsupported_stop_conditions(stop, calls):
    response = reply("anthropic", calls=calls)
    response["stop_reason"] = stop
    with pytest.raises(ValueError):
        codec("anthropic").decode_turn(response, model="test-model")


@pytest.mark.parametrize(
    "provider,provider_class", [("openai", OpenAIAPI), ("anthropic", AnthropicAPI)]
)
@pytest.mark.parametrize("with_tools", [True, False])
def test_aturn_sends_one_request_with_instructions_and_never_executes(
    provider, provider_class, with_tools
):
    client = Client(JsonResponse(reply(provider, calls=with_tools)), asynchronous=True)
    provider_instance = provider_class(
        "test-model", async_client=client, timeout=12.5, max_output_tokens=123
    )
    history = [UserMessage("read")]
    turn = asyncio.run(
        provider_instance.aturn(
            history, tools=[read] if with_tools else [], instructions="Be brief."
        )
    )
    assert turn.text == "Looking. "
    assert history == [UserMessage("read")]
    assert client.state["closed"] == 0
    assert len(client.state["requests"]) == 1
    options, request = client.state["requests"][0]
    assert options == {"timeout": 12.5, "max_retries": 0}
    assert request["model"] == "test-model"
    assert request["tools"] == codec(provider).tool_definitions(
        prepare_tools([read] if with_tools else [])
    )
    if provider == "openai":
        assert request["input"] == [{"role": "user", "content": "read"}]
        assert request["instructions"] == "Be brief."
        assert request["max_output_tokens"] == 123 and request["store"] is False
        assert request["include"] == ["reasoning.encrypted_content"]
        assert request.get("parallel_tool_calls") is (False if with_tools else None)
    else:
        assert request["messages"] == [{"role": "user", "content": "read"}]
        assert request["system"] == "Be brief."
        assert request["max_tokens"] == 123
        assert request.get("tool_choice") == (
            {"type": "auto", "disable_parallel_tool_use": True} if with_tools else None
        )


@pytest.mark.parametrize(
    "provider,provider_class", [("openai", OpenAIAPI), ("anthropic", AnthropicAPI)]
)
@pytest.mark.parametrize("problem", ["definition", "history", "unexpected_calls"])
def test_aturn_rejects_invalid_inputs_before_io_and_unadvertised_calls(
    provider, provider_class, problem
):
    client = Client(JsonResponse(reply(provider)), asynchronous=True)
    provider_instance = provider_class("test-model", async_client=client)
    history, tools = [UserMessage("read")], [read]
    if problem == "definition":
        tools = [lambda missing_annotation: None]
    elif problem == "history":
        history += [
            ModelTurn(provider, "test-model", "", [ToolCall("a", "read", {})], [], "tool_calls")
        ]
    else:
        tools = []
    with pytest.raises(ProviderError):
        asyncio.run(provider_instance.aturn(history, tools=tools))
    assert len(client.state["requests"]) == (1 if problem == "unexpected_calls" else 0)


@pytest.mark.parametrize(
    "provider,provider_class", [("openai", OpenAIAPI), ("anthropic", AnthropicAPI)]
)
@pytest.mark.parametrize("error", [RuntimeError("offline"), asyncio.CancelledError()])
def test_aturn_errors_preserve_causes_and_cancellation(provider, provider_class, error):
    client = Client(error, asynchronous=True)
    provider_instance = provider_class("test-model", async_client=client)
    with pytest.raises(
        asyncio.CancelledError if isinstance(error, asyncio.CancelledError) else ProviderError
    ) as caught:
        asyncio.run(provider_instance.aturn([UserMessage("read")], tools=[]))
    if isinstance(error, RuntimeError):
        assert caught.value.__cause__ is error
    assert client.state["closed"] == 0


@pytest.mark.parametrize(
    "provider,provider_class", [("openai", OpenAIAPI), ("anthropic", AnthropicAPI)]
)
@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
def test_aturn_closes_owned_client_on_every_exit(provider, provider_class, outcome, monkeypatch):
    import sys
    from types import SimpleNamespace

    response = JsonResponse(reply(provider, calls=False))
    if outcome == "error":
        response = RuntimeError("offline")
    elif outcome == "cancelled":
        response = asyncio.CancelledError()
    client = Client(response, asynchronous=True)
    options_seen = []

    def factory(**options):
        options_seen.append(options)
        return client

    name = "AsyncOpenAI" if provider == "openai" else "AsyncAnthropic"
    monkeypatch.setitem(sys.modules, provider, SimpleNamespace(**{name: factory}))
    provider_instance = provider_class("test-model", timeout=7, max_retries=1)
    if outcome == "success":
        assert (
            asyncio.run(provider_instance.aturn([UserMessage("read")], tools=[])).text
            == "Looking. "
        )
    else:
        error = ProviderError if outcome == "error" else asyncio.CancelledError
        with pytest.raises(error):
            asyncio.run(provider_instance.aturn([UserMessage("read")], tools=[]))
    assert options_seen == [{"timeout": 7, "max_retries": 1}]
    assert client.state["entered"] == client.state["closed"] == 1


@pytest.mark.parametrize(
    "provider,provider_class", [("openai", OpenAIAPI), ("anthropic", AnthropicAPI)]
)
def test_aturn_missing_extra_is_lazy_and_preparation_precedes_client(
    provider, provider_class, monkeypatch
):
    import sys

    monkeypatch.setitem(sys.modules, provider, None)
    provider_instance = provider_class("test-model")
    with pytest.raises(ProviderError, match="definition"):
        asyncio.run(provider_instance.aturn([UserMessage("read")], tools=[lambda invalid: None]))
    with pytest.raises(ProviderError, match="unresolved"):
        asyncio.run(
            provider_instance.aturn(
                [
                    UserMessage("read"),
                    ModelTurn(
                        provider, "test-model", "", [ToolCall("a", "read", {})], [], "tool_calls"
                    ),
                ],
                tools=[],
            )
        )
    with pytest.raises(ProviderError, match=rf"slick-ai\[{provider}\]"):
        asyncio.run(provider_instance.aturn([UserMessage("read")], tools=[]))

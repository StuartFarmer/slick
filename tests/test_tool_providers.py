"""Exact native formats for stateless, application-built tool exchanges."""

import importlib
import json
from copy import deepcopy

import pytest


def codec(name):
    return importlib.import_module(f"slick.providers._tools._{name}")


def read(path: str) -> str:
    """Read a file."""
    raise AssertionError("Providers must never execute Python tools")


def request():
    return {"id": "a", "name": "read", "arguments": {"path": "a.py"}}


def reply(name, *, calls=True, text="  answer\n", arguments=None):
    args = {"path": "a.py"} if arguments is None else arguments
    encoded = json.dumps(args) if isinstance(args, dict) else args
    if name == "openai":
        items = (
            [
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
            if text
            else []
        )
        if calls:
            items.append(
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "read",
                    "arguments": encoded,
                    "status": "completed",
                }
            )
        return {"status": "completed", "output": items}
    if name == "anthropic":
        items = [{"type": "text", "text": text}] if text else []
        if calls:
            items.append({"type": "tool_use", "id": "a", "name": "read", "input": args})
        return {"stop_reason": "tool_use" if calls else "end_turn", "content": items}
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = [
            {"type": "function", "id": "a", "function": {"name": "read", "arguments": encoded}}
        ]
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}]}


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
@pytest.mark.parametrize("text,calls", [("  answer\n", False), ("  answer\n", True), ("", True)])
def test_decode_preserves_text_and_requests(name, text, calls):
    original = reply(name, calls=calls, text=text)
    saved = deepcopy(original)
    assert codec(name).decode_response(original) == (text, [request()] if calls else [])
    assert original == saved


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_encode_self_contained_results(name):
    results = [{"request": request(), "content": "contents", "is_error": False}]
    saved = deepcopy(results)
    encoded = codec(name).encode_request("new summary", {}, results)
    if name == "openai":
        assert encoded == {
            "input": [
                {"role": "user", "content": "new summary"},
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "read",
                    "arguments": '{"path": "a.py"}',
                },
                {"type": "function_call_output", "call_id": "a", "output": "contents"},
            ]
        }
    elif name == "anthropic":
        assert encoded == {
            "messages": [
                {"role": "user", "content": "new summary"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "a", "name": "read", "input": {"path": "a.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "a",
                            "content": "contents",
                            "is_error": False,
                        },
                    ],
                },
            ]
        }
    else:
        assert encoded == {
            "messages": [
                {"role": "user", "content": "new summary"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "a",
                            "function": {"name": "read", "arguments": '{"path": "a.py"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "contents"},
            ]
        }
    assert results == saved


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_definitions_are_owned_and_empty_context_does_not_invent_text(name):
    from slick.tools import prepare_tools

    prepared = prepare_tools([read])
    encoded = codec(name).encode_request(
        "",
        prepared,
        [
            {"request": request(), "content": "error", "is_error": True},
            {"request": {**request(), "id": "b"}, "content": "ok", "is_error": False},
        ],
    )
    assert encoded["tools"]
    assert "user" not in json.dumps(encoded.get("input", [])[:1])
    definitions = codec(name).tool_definitions(prepared)
    schema = definitions[0].get("function", definitions[0])
    schema = schema.get("input_schema", schema.get("parameters"))
    schema["required"].clear()
    assert prepared["read"].parameters["required"] == ["path"]


@pytest.mark.parametrize("name", ["openai", "chat_completions"])
@pytest.mark.parametrize("raw", ["bad", "[]", '{"x":NaN}', '{"x":1,"x":2}'])
def test_malformed_json_can_be_returned_as_an_error(name, raw):
    text, requests = codec(name).decode_response(reply(name, arguments=raw))
    assert requests[0]["arguments"] == raw
    assert requests[0]["argument_error"]
    encoded = codec(name).encode_request(
        "continue",
        {},
        [
            {"request": requests[0], "content": "Invalid arguments", "is_error": True},
        ],
    )
    assert raw in json.dumps(encoded).replace('\\"', '"')


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_native_reasoning_with_tools_is_not_silently_discarded(name):
    raw = reply(name)
    if name == "openai":
        raw["output"].insert(0, {"type": "reasoning", "encrypted_content": "opaque"})
    elif name == "anthropic":
        raw["content"].insert(0, {"type": "thinking", "thinking": "private", "signature": "sig"})
    else:
        raw["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.encrypted"}]
    with pytest.raises(ValueError, match="reasoning"):
        codec(name).decode_response(raw)


def test_openai_rejects_a_message_without_text_blocks():
    raw = reply("openai", calls=False)
    raw["output"][0]["content"] = []
    with pytest.raises(ValueError, match="content"):
        codec("openai").decode_response(raw)


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_duplicate_model_requests_are_rejected(name):
    raw = reply(name)
    if name == "openai":
        raw["output"].append(deepcopy(raw["output"][-1]))
    elif name == "anthropic":
        raw["content"].append(deepcopy(raw["content"][-1]))
    else:
        calls = raw["choices"][0]["message"]["tool_calls"]
        calls.append(deepcopy(calls[0]))
    with pytest.raises(ValueError, match="duplicate"):
        codec(name).decode_response(raw)

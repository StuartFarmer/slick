"""Exact native formats for stateless, application-built tool exchanges."""

import json
from copy import deepcopy

import pytest
from sdk_fakes import sdk_response


def provider(name):
    from slick.providers import AnthropicAPI, OpenAIAPI, OpenRouterAPI

    return {"openai": OpenAIAPI, "anthropic": AnthropicAPI, "chat_completions": OpenRouterAPI}[
        name
    ]("model")


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
    assert provider(name)._decode(sdk_response(original)) == (text, [request()] if calls else [])
    assert original == saved


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_encode_self_contained_results(name):
    results = [{"request": request(), "content": "contents", "is_error": False}]
    saved = deepcopy(results)
    encoded = provider(name)._encode("new summary", {}, results)
    if name == "openai":
        assert encoded == {
            "model": "model",
            "max_output_tokens": 2048,
            "store": False,
            "input": [
                {"role": "user", "content": "new summary"},
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "read",
                    "arguments": '{"path": "a.py"}',
                },
                {"type": "function_call_output", "call_id": "a", "output": "contents"},
            ],
        }
    elif name == "anthropic":
        assert encoded == {
            "model": "model",
            "max_tokens": 2048,
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
            ],
        }
    else:
        assert encoded == {
            "model": "model",
            "max_tokens": 2048,
            "stream": False,
            "n": 1,
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
            ],
        }
    assert results == saved


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_definitions_are_owned_and_empty_context_does_not_invent_text(name):
    from slick.tools import prepare_tools

    prepared = prepare_tools([read])
    encoded = provider(name)._encode(
        "",
        prepared,
        [
            {"request": request(), "content": "error", "is_error": True},
            {"request": {**request(), "id": "b"}, "content": "ok", "is_error": False},
        ],
    )
    assert encoded["tools"]
    assert "user" not in json.dumps(encoded.get("input", [])[:1])
    definitions = encoded["tools"]
    schema = definitions[0].get("function", definitions[0])
    schema = schema.get("input_schema", schema.get("parameters"))
    schema["required"].clear()
    assert prepared["read"].parameters["required"] == ["path"]


@pytest.mark.parametrize("name", ["openai", "chat_completions"])
@pytest.mark.parametrize("raw", ["bad", "{"])
def test_malformed_json_can_be_returned_as_an_error(name, raw):
    text, requests = provider(name)._decode(sdk_response(reply(name, arguments=raw)))
    assert requests[0]["arguments"] == raw
    assert requests[0]["argument_error"]
    encoded = provider(name)._encode(
        "continue",
        {},
        [
            {"request": requests[0], "content": "Invalid arguments", "is_error": True},
        ],
    )
    assert raw in json.dumps(encoded).replace('\\"', '"')


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_decoders_extract_text_and_tools_from_reasoning_responses(name):
    raw = reply(name)
    if name == "openai":
        raw["output"].insert(0, {"type": "reasoning", "encrypted_content": "opaque"})
    elif name == "anthropic":
        raw["content"].insert(0, {"type": "thinking", "thinking": "private", "signature": "sig"})
    else:
        raw["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.encrypted"}]
    assert provider(name)._decode(sdk_response(raw)) == ("  answer\n", [request()])


def test_openai_returns_empty_text_when_no_text_blocks_exist():
    raw = reply("openai", calls=False)
    raw["output"][0]["content"] = []
    assert provider("openai")._decode(sdk_response(raw)) == ("", [])


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_duplicate_model_requests_are_preserved(name):
    raw = reply(name)
    if name == "openai":
        raw["output"].append(deepcopy(raw["output"][-1]))
    elif name == "anthropic":
        raw["content"].append(deepcopy(raw["content"][-1]))
    else:
        calls = raw["choices"][0]["message"]["tool_calls"]
        calls.append(deepcopy(calls[0]))
    _, requests = provider(name)._decode(sdk_response(raw))
    assert len(requests) == 2 and requests[0] == requests[1]


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_invalid_response_errors_do_not_include_response_content(name):
    raw = reply(name, text={"private": "sensitive-response-content"})
    if name == "chat_completions":
        response = sdk_response(raw)
        assert provider(name)._decode(response)[0] is response.choices[0].message.content
        return
    with pytest.raises((TypeError, ValueError)) as caught:
        provider(name)._decode(sdk_response(raw))
    assert "sensitive-response-content" not in str(caught.value)


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
@pytest.mark.parametrize("field,value", [("id", ""), ("id", 1), ("name", " "), ("name", None)])
def test_completed_response_preserves_tool_request_fields(name, field, value):
    raw = reply(name)
    if name == "openai":
        raw["output"][-1]["call_id" if field == "id" else field] = value
    elif name == "anthropic":
        raw["content"][-1][field] = value
    else:
        call = raw["choices"][0]["message"]["tool_calls"][0]
        (call if field == "id" else call["function"])[field] = value
    _, requests = provider(name)._decode(sdk_response(raw))
    assert requests[0][field] == value

"""Real optional SDK serialization/parsing against an offline HTTP transport."""

import asyncio
import json

import pytest

from slick.backends import Anthropic, OpenAI

httpx = pytest.importorskip("httpx")


CASES = [
    (
        "openai",
        "OpenAI",
        OpenAI,
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
        Anthropic,
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


@pytest.mark.parametrize("provider,sdk_class,backend_class,path,body,reply", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sdk_request_and_response_leave_injected_transport_open(
    provider, sdk_class, backend_class, path, body, reply, asynchronous
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
                backend = backend_class("test-model", async_client=client, **options)
                assert await backend.acall("prompt\n") == "  answer\n"
                assert not http_client.is_closed
                assert not client.is_closed()

        asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as http_client:
            client = getattr(sdk, sdk_class)(api_key="offline-test-key", http_client=http_client)
            backend = backend_class("test-model", client=client, **options)
            assert backend.call("prompt\n") == "  answer\n"
            assert not http_client.is_closed
            assert not client.is_closed()

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == path
    assert json.loads(requests[0].content) == body
    assert requests[0].extensions["timeout"]["read"] == 12.5


@pytest.mark.parametrize("provider,sdk_class,backend_class,path,body,reply", CASES)
def test_real_sdk_native_turns_replay_original_call_and_correlated_result(
    provider, sdk_class, backend_class, path, body, reply
):
    from copy import deepcopy

    from slick.turns import ToolResult, UserMessage

    sdk = pytest.importorskip(provider)
    first = deepcopy(reply)
    if provider == "openai":
        first["output"] = [
            {"type": "reasoning", "id": "r1", "summary": [], "encrypted_content": "opaque"},
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
            {"type": "thinking", "thinking": "private", "signature": "signed"},
            {"type": "tool_use", "id": "call_1", "name": "read", "input": {"key": "intro"}},
        ]
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=first if len(requests) == 1 else reply)

    def read(key: str) -> str:
        """Read a document."""
        raise AssertionError("backend executed a tool")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
            client = getattr(sdk, "Async" + sdk_class)(api_key="offline", http_client=http_client)
            backend = backend_class("test-model", async_client=client, timeout=12.5)
            history = [UserMessage("read intro")]
            turn = await backend.aturn(history, tools=[read])
            assert turn.tool_calls[0].arguments == {"key": "intro"}
            history.extend([turn, ToolResult("call_1", "Introduction")])
            final = await backend.aturn(history, tools=[read])
            assert final.text == "  answer\n" and final.stop_reason == "end_turn"
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
        assert calls[0]["arguments"] == '{"key":"intro"}'
        assert second["input"][1]["encrypted_content"] == "opaque"
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
        assert second["messages"][-2]["content"][0]["signature"] == "signed"

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

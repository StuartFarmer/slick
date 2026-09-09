"""Chat Completions conversion; no execution or stored state."""

from __future__ import annotations

import json
from copy import deepcopy

from ...tools._protocol import make_request, validate_response


def tool_definitions(prepared: dict) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.parameters),
            },
        }
        for tool in prepared.values()
    ]


def encode_request(context, prepared, results):
    messages = [{"role": "user", "content": context}] if context else []
    calls = []
    for result in results:
        call = result["request"]
        arguments = call["arguments"]
        calls.append(
            {
                "type": "function",
                "id": call["id"],
                "function": {
                    "name": call["name"],
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                },
            }
        )
    if results:
        messages.append({"role": "assistant", "tool_calls": calls})
        messages.extend(
            {"role": "tool", "tool_call_id": result["request"]["id"], "content": result["content"]}
            for result in results
        )
    payload = {"messages": messages}
    if prepared:
        payload["tools"] = tool_definitions(prepared)
    return payload


def _decode_call(item: dict) -> dict:
    if item.get("type") != "function":
        raise ValueError("Only local function tools are supported")
    function = item.get("function")
    if not isinstance(function, dict):
        raise ValueError("Chat Completions tool call requires a function object")
    if "arguments" not in function:
        raise ValueError(f"Chat Completions call {item.get('id')!r} is missing arguments")
    return make_request(item.get("id", ""), function.get("name", ""), function["arguments"])


def decode_response(response):
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Chat Completions must return exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Expected an assistant response")
    if message.get("refusal"):
        raise ValueError("Assistant response was refused")
    if message.get("function_call"):
        raise ValueError("Legacy function calls are unsupported")
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise ValueError("Chat Completions tool_calls must be a list")
    requests = [_decode_call(item) for item in raw_calls]
    expected = "tool_calls" if requests else "stop"
    if choice.get("finish_reason") != expected:
        raise ValueError("Chat Completions response is incomplete or inconsistent")
    if requests and any(
        message.get(key) for key in ("reasoning", "reasoning_content", "reasoning_details")
    ):
        raise ValueError(
            "Native reasoning with tools is unsupported by the portable tool interface"
        )
    text = message.get("content")
    if text is None and requests:
        text = ""
    return validate_response((text, requests))

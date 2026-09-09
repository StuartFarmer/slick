"""OpenAI Responses conversion; no execution or stored state."""

from __future__ import annotations

import json
from copy import deepcopy

from ...tools._protocol import make_request, validate_response


def tool_definitions(prepared: dict) -> list[dict]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": deepcopy(tool.parameters),
            "strict": False,
        }
        for tool in prepared.values()
    ]


def encode_request(context, prepared, results):
    items = [{"role": "user", "content": context}] if context else []
    for result in results:
        call = result["request"]
        arguments = call["arguments"]
        items.append(
            {
                "type": "function_call",
                "call_id": call["id"],
                "name": call["name"],
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            }
        )
    items.extend(
        {
            "type": "function_call_output",
            "call_id": result["request"]["id"],
            "output": result["content"],
        }
        for result in results
    )
    payload = {"input": items if results else context}
    if prepared:
        payload["tools"] = tool_definitions(prepared)
        payload["parallel_tool_calls"] = False
    return payload


def _message_text(item: dict) -> str:
    if item.get("status") != "completed":
        raise ValueError("OpenAI message did not complete")
    content = item.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("OpenAI message content must be a nonempty list")
    text = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "output_text":
            raise ValueError("unsupported OpenAI message content (including refusal)")
        if not isinstance(block.get("text"), str):
            raise ValueError("OpenAI output_text requires string text")
        text.append(block["text"])
    return "".join(text)


def _tool_call(item: dict) -> dict:
    if "arguments" not in item:
        raise ValueError(f"OpenAI call {item.get('call_id')!r} is missing arguments")
    return make_request(item.get("call_id", ""), item.get("name", ""), item["arguments"])


def _decode_items(items: list) -> tuple[str, list[dict]]:
    text, calls = [], []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("OpenAI output item must be an object")
        if item.get("status") not in (None, "completed"):
            raise ValueError(f"OpenAI output item did not complete: {item.get('status')}")
        kind = item.get("type")
        if kind == "message":
            text.append(_message_text(item))
        elif kind == "function_call":
            calls.append(_tool_call(item))
        elif kind != "reasoning":
            raise ValueError(f"unsupported OpenAI output type: {kind!r}")
    return "".join(text), calls


def decode_response(response):
    if response.get("status") != "completed":
        raise ValueError(f"OpenAI response did not complete: {response.get('status')}")
    items = response.get("output")
    if not isinstance(items, list):
        raise ValueError("OpenAI output must be a list")
    text, requests = _decode_items(items)
    if not requests and not any(item.get("type") == "message" for item in items):
        raise ValueError("Response contains no text blocks or tool requests")
    if requests and any(item.get("type") == "reasoning" for item in items):
        raise ValueError(
            "Native reasoning with tools is unsupported by the portable tool interface"
        )
    return validate_response((text, requests))

"""Anthropic Messages conversion; no execution or stored state."""

from __future__ import annotations

from copy import deepcopy

from ...tools._protocol import make_request, validate_response


def tool_definitions(prepared: dict) -> list[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": deepcopy(tool.parameters),
        }
        for tool in prepared.values()
    ]


def encode_request(context, prepared, results):
    messages = [{"role": "user", "content": context}] if context else []
    calls, outputs = [], []
    for result in results:
        call = result["request"]
        if not isinstance(call["arguments"], dict):
            raise ValueError("Anthropic tool input must be an object; cannot replay malformed JSON")
        calls.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": deepcopy(call["arguments"]),
            }
        )
        outputs.append(
            {
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": result["content"],
                "is_error": result["is_error"],
            }
        )
    if results:
        messages.extend(
            [{"role": "assistant", "content": calls}, {"role": "user", "content": outputs}]
        )
    payload = {"messages": messages}
    if prepared:
        payload["tools"] = tool_definitions(prepared)
        payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    return payload


def _tool_call(item: dict) -> dict:
    if "input" not in item:
        raise ValueError(f"Anthropic call {item.get('id')!r} is missing input")
    return make_request(item.get("id", ""), item.get("name", ""), item["input"])


def _decode_items(items: list) -> tuple[str, list[dict]]:
    text, calls = [], []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Anthropic content block must be an object")
        kind = item.get("type")
        if kind == "text":
            if not isinstance(item.get("text"), str):
                raise ValueError("Anthropic text block requires string text")
            text.append(item["text"])
        elif kind == "tool_use":
            calls.append(_tool_call(item))
        elif kind not in {"thinking", "redacted_thinking"}:
            raise ValueError(f"unsupported Anthropic content type: {kind!r}")
    return "".join(text), calls


def decode_response(response):
    stop = response.get("stop_reason")
    if stop not in {"end_turn", "stop_sequence", "tool_use"}:
        raise ValueError(f"Anthropic response did not complete: {stop!r}")
    items = response.get("content")
    if not isinstance(items, list):
        raise ValueError("Anthropic content must be a list")
    text, requests = _decode_items(items)
    if not requests and not any(item.get("type") == "text" for item in items):
        raise ValueError("Response contains no text blocks or tool requests")
    if bool(requests) != (stop == "tool_use"):
        raise ValueError(f"Anthropic stop_reason {stop!r} is inconsistent with tool requests")
    if requests and any(item.get("type") in {"thinking", "redacted_thinking"} for item in items):
        raise ValueError(
            "Native reasoning with tools is unsupported by the portable tool interface"
        )
    return validate_response((text, requests))

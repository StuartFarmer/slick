"""Pure Anthropic Messages conversion; preserve signed continuation blocks."""

from __future__ import annotations

from copy import deepcopy

from .turns import ModelTurn, ToolCall, ToolResult, UserMessage, _validate_turn, decode_arguments


def tool_definitions(prepared: dict) -> list[dict]:
    return [
        {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
        for tool in prepared.values()
    ]


def encode_history(history: list) -> list[dict]:
    messages: list[dict] = []
    results: list[dict] = []
    for record in history:
        if isinstance(record, ToolResult):
            if not results:
                messages.append({"role": "user", "content": results})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": record.call_id,
                    "content": record.content,
                    "is_error": record.is_error,
                }
            )
            continue
        results = []
        if isinstance(record, UserMessage):
            messages.append({"role": "user", "content": record.text})
        elif isinstance(record, ModelTurn):
            messages.append({"role": "assistant", "content": deepcopy(record.items)})
        else:
            raise ValueError(f"unsupported history record: {type(record).__name__}")
    return messages


def _tool_call(item: dict) -> ToolCall:
    if "input" not in item:
        raise ValueError(f"Anthropic call {item.get('id')!r} is missing input")
    arguments, error = decode_arguments(item["input"])
    return ToolCall(item.get("id", ""), item.get("name", ""), arguments, error)


def _decode_items(items: list) -> tuple[str, list[ToolCall]]:
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


def decode_turn(response: dict, *, model: str) -> ModelTurn:
    stop = response.get("stop_reason")
    if stop not in {"end_turn", "tool_use"}:
        raise ValueError(f"Anthropic response did not complete: {stop!r}")
    items = deepcopy(response.get("content"))
    if not isinstance(items, list):
        raise ValueError("Anthropic content must be a list")
    text, calls = _decode_items(items)
    if bool(calls) != (stop == "tool_use"):
        raise ValueError(f"Anthropic stop_reason {stop!r} is inconsistent with tool calls")
    usage = response.get("usage") or {}
    turn = ModelTurn(
        "anthropic",
        model,
        text,
        calls,
        items,
        "tool_calls" if calls else "end_turn",
        usage.get("input_tokens"),
        usage.get("output_tokens"),
    )
    _validate_turn(turn)
    return turn

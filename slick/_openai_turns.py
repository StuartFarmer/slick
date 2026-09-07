"""Pure OpenAI Responses conversion; no SDK imports or tool execution."""

from __future__ import annotations

from copy import deepcopy

from .turns import ModelTurn, ToolCall, ToolResult, UserMessage, _validate_turn, decode_arguments


def tool_definitions(prepared: dict) -> list[dict]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": False,
        }
        for tool in prepared.values()
    ]


def encode_history(history: list) -> list[dict]:
    items = []
    for record in history:
        if isinstance(record, UserMessage):
            items.append({"role": "user", "content": record.text})
        elif isinstance(record, ModelTurn):
            items.extend(deepcopy(record.items))
        elif isinstance(record, ToolResult):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": record.call_id,
                    "output": record.content,
                }
            )
        else:
            raise ValueError(f"unsupported history record: {type(record).__name__}")
    return items


def _message_text(item: dict) -> str:
    if item.get("status") != "completed":
        raise ValueError("OpenAI message did not complete")
    content = item.get("content")
    if not isinstance(content, list):
        raise ValueError("OpenAI message content must be a list")
    text = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "output_text":
            raise ValueError("unsupported OpenAI message content (including refusal)")
        if not isinstance(block.get("text"), str):
            raise ValueError("OpenAI output_text requires string text")
        text.append(block["text"])
    return "".join(text)


def _tool_call(item: dict) -> ToolCall:
    if "arguments" not in item:
        raise ValueError(f"OpenAI call {item.get('call_id')!r} is missing arguments")
    arguments, error = decode_arguments(item["arguments"])
    return ToolCall(item.get("call_id", ""), item.get("name", ""), arguments, error)


def _decode_items(items: list) -> tuple[str, list[ToolCall]]:
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


def decode_turn(response: dict, *, model: str) -> ModelTurn:
    if response.get("status") != "completed":
        raise ValueError(f"OpenAI response did not complete: {response.get('status')}")
    items = deepcopy(response.get("output"))
    if not isinstance(items, list):
        raise ValueError("OpenAI output must be a list")
    text, calls = _decode_items(items)
    usage = response.get("usage") or {}
    turn = ModelTurn(
        "openai",
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

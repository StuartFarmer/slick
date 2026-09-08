"""Common OpenAI-compatible Chat Completions conversion."""

from __future__ import annotations

from copy import deepcopy

from . import ModelTurn, ToolCall, ToolResult, UserMessage, _validate_turn, decode_arguments


def tool_definitions(prepared: dict) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in prepared.values()
    ]


def encode_history(history: list, *, instructions: str = "") -> list[dict]:
    messages = [{"role": "system", "content": instructions}] if instructions else []
    for record in history:
        if isinstance(record, UserMessage):
            messages.append({"role": "user", "content": record.text})
        elif isinstance(record, ModelTurn):
            messages.extend(deepcopy(record.items))
        elif isinstance(record, ToolResult):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": record.call_id,
                    "content": record.content,
                }
            )
        else:
            raise ValueError(f"unsupported history record: {type(record).__name__}")
    return messages


def _decode_call(item: dict) -> ToolCall:
    if item.get("type") != "function":
        raise ValueError("Only local function tools are supported")
    function = item.get("function")
    if not isinstance(function, dict):
        raise ValueError("Chat Completions tool call requires a function object")
    if "arguments" not in function:
        raise ValueError(f"Chat Completions call {item.get('id')!r} is missing arguments")
    arguments, error = decode_arguments(function["arguments"])
    return ToolCall(item.get("id", ""), function.get("name", ""), arguments, error)


def decode_turn(response: dict, *, provider: str, model: str) -> ModelTurn:
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("Chat Completions must return exactly one choice")
    choice = choices[0]
    message = deepcopy(choice.get("message"))
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Expected an assistant response")
    if message.get("refusal"):
        raise ValueError("Assistant response was refused")
    if message.get("function_call"):
        raise ValueError("Legacy function calls are unsupported")
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ValueError("Chat Completions tool_calls must be a list")
    calls = [_decode_call(item) for item in raw_calls]
    finish_reason = choice.get("finish_reason")
    expected = "tool_calls" if calls else "stop"
    if finish_reason != expected:
        raise ValueError("Chat Completions response is incomplete or inconsistent")
    usage = response.get("usage") or {}
    turn = ModelTurn(
        provider,
        model,
        message.get("content") or "",
        calls,
        [message],
        "tool_calls" if calls else "end_turn",
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )
    _validate_turn(turn)
    return turn

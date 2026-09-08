"""One native model turn and application-owned history, without tool execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from .tools import _check_value


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict | None
    argument_error: str | None = None


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ModelTurn:
    provider: str
    model: str
    text: str
    tool_calls: list[ToolCall]
    items: list[dict]
    stop_reason: Literal["tool_calls", "end_turn"]
    input_tokens: int | None = None
    output_tokens: int | None = None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate argument key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite number {value}")


def decode_arguments(raw: str | dict) -> tuple[dict | None, str | None]:
    """Decode a strict JSON object, retaining a recoverable error for bad calls."""
    try:
        if isinstance(raw, dict):
            _check_value(raw, "arguments")
            raw = json.dumps(raw, allow_nan=False)
        if not isinstance(raw, str):
            raise ValueError("arguments must be a JSON object")
        arguments = json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        _check_value(arguments, "arguments")
        return arguments, None
    except (ValueError, TypeError, RecursionError) as exc:
        return None, f"Invalid arguments: {exc}"


def _validate_turn(turn: ModelTurn) -> None:
    if not isinstance(turn.text, str):
        raise ValueError("turn text must be a string")
    if not turn.text and not turn.tool_calls:
        raise ValueError("turn must contain text or tool calls")
    expected = "tool_calls" if turn.tool_calls else "end_turn"
    if turn.stop_reason != expected:
        raise ValueError(f"turn stop_reason must be {expected!r}")
    seen = set()
    for call in turn.tool_calls:
        if not isinstance(call, ToolCall):
            raise ValueError("turn tool_calls must contain ToolCall records")
        if not isinstance(call.id, str) or not call.id.strip():
            raise ValueError("tool call id must be nonempty")
        if call.id in seen:
            raise ValueError(f"duplicate tool call id {call.id!r}")
        if not isinstance(call.name, str) or not call.name.strip():
            raise ValueError(f"tool call {call.id!r} name must be nonempty")
        seen.add(call.id)


def validate_history(history: list, *, provider: str, model: str) -> None:
    """Reject foreign turns and broken call/result groups before a new request."""
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a nonempty list (empty history is invalid)")
    pending: set[str] = set()
    seen: set[str] = set()
    for record in history:
        if isinstance(record, ToolResult):
            _validate_result(record, pending)
            pending.remove(record.call_id)
            continue
        if pending:
            raise ValueError(f"unresolved tool calls before next record: {sorted(pending)}")
        if isinstance(record, UserMessage):
            if not isinstance(record.text, str):
                raise ValueError("user text must be a string")
        elif isinstance(record, ModelTurn):
            _validate_history_turn(record, provider, model, seen)
            pending.update(call.id for call in record.tool_calls)
            seen.update(pending)
        else:
            raise ValueError(f"unsupported history record: {type(record).__name__}")
    if pending:
        raise ValueError(f"unresolved tool calls: {sorted(pending)}")


def _validate_result(result: ToolResult, pending: set[str]) -> None:
    if not isinstance(result.call_id, str) or result.call_id not in pending:
        raise ValueError(f"unknown or duplicate result call_id {result.call_id!r}")
    if not isinstance(result.content, str) or type(result.is_error) is not bool:
        raise ValueError(f"result {result.call_id!r} requires string content and boolean is_error")


def _validate_history_turn(turn: ModelTurn, provider: str, model: str, seen: set[str]) -> None:
    if turn.provider != provider:
        raise ValueError(f"turn provider {turn.provider!r} does not match {provider!r}")
    if turn.model != model:
        raise ValueError(f"turn model {turn.model!r} does not match {model!r}")
    _validate_turn(turn)
    for call in turn.tool_calls:
        if call.id in seen:
            raise ValueError(f"duplicate tool call id {call.id!r}")

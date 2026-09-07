"""Native history and strict JSON argument contracts."""

from dataclasses import FrozenInstanceError

import pytest

from slick.turns import (
    ModelTurn,
    ToolCall,
    ToolResult,
    UserMessage,
    decode_arguments,
    validate_history,
)


def turn(ids=("a",), **kwargs):
    return ModelTurn(
        kwargs.get("provider", "openai"),
        kwargs.get("model", "test-model"),
        "",
        [ToolCall(id, "read", {}) for id in ids],
        [],
        "tool_calls",
    )


def validate(history):
    validate_history(history, provider="openai", model="test-model")


def test_records_require_turn_payload_and_freeze_shells():
    with pytest.raises(TypeError):
        ModelTurn("openai", "test-model", "answer")
    with pytest.raises(FrozenInstanceError):
        UserMessage("hello").text = "other"
    first, second = turn(), turn()
    first.items.append({"type": "reasoning"})
    assert second.items == []


@pytest.mark.parametrize(
    "raw",
    [
        '{"x": 1, "x": 2}',
        '{"nested": {"x": 1, "x": 2}}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":[1e999]}',
        "```json\n{}\n```",
        "[]",
        "null",
        "1",
        "{",
        {"x": float("nan")},
        {"x": [float("inf")]},
        {1: "bad"},
        {"x": (1, 2)},
    ],
)
def test_arguments_reject_non_strict_objects(raw):
    arguments, error = decode_arguments(raw)
    assert arguments is None
    assert isinstance(error, str) and error


@pytest.mark.parametrize(
    "raw,expected", [("{}", {}), ('{"x": [2, null]}', {"x": [2, None]}), ({}, {})]
)
def test_arguments_accept_strict_objects(raw, expected):
    assert decode_arguments(raw) == (expected, None)


def test_arguments_copy_nested_data():
    original = {"x": [{"y": 1}]}
    copied, error = decode_arguments(original)
    copied["x"][0]["y"] = 2
    assert error is None
    assert original == {"x": [{"y": 1}]}


@pytest.mark.parametrize(
    "history,match",
    [
        ([], "empty"),
        ([object()], "record"),
        ([UserMessage("read"), turn()], "a"),
        ([turn(), UserMessage("continue"), ToolResult("a", "done")], "a"),
        ([turn(), turn(("b",))], "a"),
        ([ToolResult("unknown", "done")], "unknown"),
        ([turn(), ToolResult("a", "done"), ToolResult("a", "again")], "a"),
        ([turn(("a", "a")), ToolResult("a", "done")], "a"),
        ([turn(), ToolResult("a", "done"), turn()], "a"),
        ([turn(provider="anthropic"), ToolResult("a", "done")], "provider"),
        ([turn(model="other"), ToolResult("a", "done")], "model"),
        ([turn(("",)), ToolResult("", "done")], "id"),
        (
            [ModelTurn("openai", "test-model", "", [ToolCall("a", "", {})], [], "tool_calls")],
            "name",
        ),
        ([ModelTurn("openai", "test-model", "", [], [], "end_turn")], "text"),
        ([ModelTurn("openai", "test-model", "hi", [], [], "tool_calls")], "stop_reason"),
    ],
)
def test_history_rejects_invalid_groups(history, match):
    with pytest.raises(ValueError, match=match):
        validate(history)


def test_history_accepts_complete_groups_and_plain_conversation():
    validate(
        [
            UserMessage("read"),
            turn(("a", "b")),
            ToolResult("b", "missing", True),
            ToolResult("a", "contents"),
            ModelTurn("openai", "test-model", "answer", [], [], "end_turn"),
            UserMessage("thanks"),
        ]
    )

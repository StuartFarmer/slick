import json

import pytest
from pydantic import ValidationError

from examples.coding_harness.state import Check, HarnessConfig, Limits, SessionState, load_config


def test_defaults_and_independent_mutable_state():
    config = load_config(None)
    assert config == HarnessConfig()
    assert config.limits.model_dump() == {
        "max_turns": 30,
        "max_tool_calls": 60,
        "max_repairs": 3,
        "task_timeout": 900,
        "command_timeout": 120,
        "context_soft_chars": 80000,
        "context_hard_chars": 120000,
    }
    first, second = SessionState(), SessionState()
    first.history.append("a")
    assert second.history == []


@pytest.mark.parametrize(
    "values",
    [
        {"max_turns": 0},
        {"max_repairs": -1},
        {"max_turns": True},
        {"task_timeout": "2"},
        {"extra": 1},
        {"context_soft_chars": 120000},
    ],
)
def test_invalid_limits(values):
    with pytest.raises(ValidationError):
        Limits(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"checks": [{"name": "x", "argv": []}]},
        {"checks": [{"name": "", "argv": ["a"]}]},
        {"checks": [{"name": "x", "argv": [""]}]},
        {"checks": [{"name": "x", "argv": ["a"]}, {"name": "x", "argv": ["b"]}]},
        {"unknown": 1},
    ],
)
def test_invalid_config(values):
    with pytest.raises(ValidationError):
        HarnessConfig(**values)


def test_explicit_json_config(tmp_path):
    path = tmp_path / "checks.json"
    path.write_text(
        json.dumps({"checks": [{"name": "unit", "argv": ["python", "-m", "unittest"]}]})
    )
    assert load_config(path).checks == [Check(name="unit", argv=["python", "-m", "unittest"])]

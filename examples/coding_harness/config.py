"""Harness configuration, defaults, and JSON loading."""

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from .checks import Check

PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class Limits(BaseModel, extra="forbid"):
    max_turns: PositiveInt = 30
    max_tool_calls: PositiveInt = 60
    max_repairs: PositiveInt = 3
    task_timeout: PositiveInt = 900
    command_timeout: PositiveInt = 120
    context_soft_chars: PositiveInt = 80000
    context_hard_chars: PositiveInt = 120000

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if self.context_soft_chars >= self.context_hard_chars:
            raise ValueError("context_soft_chars must be below context_hard_chars")
        return self


class HarnessConfig(BaseModel, extra="forbid"):
    checks: list[Check] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)

    @field_validator("checks")
    @classmethod
    def unique_checks(cls, checks):
        if len({check.name for check in checks}) != len(checks):
            raise ValueError("check names must be unique")
        return checks


def load_config(path: Path | None) -> HarnessConfig:
    return (
        HarnessConfig()
        if path is None
        else HarnessConfig.model_validate_json(path.read_text(encoding="utf-8"))
    )

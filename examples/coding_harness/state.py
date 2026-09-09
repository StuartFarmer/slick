"""Application configuration and inert observations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Check(Record):
    name: str = Field(min_length=1)
    argv: list[str]
    timeout: PositiveInt = 120

    @field_validator("argv")
    @classmethod
    def valid_argv(cls, argv):
        if not argv or not argv[0] or any("\0" in arg for arg in argv):
            raise ValueError("argv must contain an executable and no NUL bytes")
        return argv


class Limits(Record):
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


class HarnessConfig(Record):
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


class FileList(Record):
    paths: list[str]
    truncated: bool


class SearchHit(Record):
    path: str
    line: int
    text: str


class SearchResult(Record):
    hits: list[SearchHit]
    truncated: bool


class FileSlice(Record):
    path: str
    text: str
    start_line: int
    end_line: int
    sha256: str
    truncated: bool


class EditResult(Record):
    path: str
    sha256: str
    diff: str


class CommandResult(Record):
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


class DiffResult(Record):
    unstaged: str
    staged: str
    untracked: list[str]
    truncated: bool


class CommandRequest(Record):
    argv: list[str]
    cwd: str
    timeout: PositiveInt


CommandDecision = Literal["once", "session", "deny"]


class CheckResult(Record):
    name: str
    command: CommandResult
    before_fingerprint: str
    after_fingerprint: str


class Verification(Record):
    results: list[CheckResult]
    fingerprint: str
    passed: bool
    stable: bool


class RunResult(Record):
    status: Literal["verified", "unverified", "blocked", "cancelled", "failed"]
    answer: str
    checks: list[CheckResult]
    turns: int
    tool_calls: int
    repairs: int
    changed_paths: list[str]


class ContextSummary(Record):
    facts: list[str]
    decisions: list[str]
    open_questions: list[str]
    modified_files: list[str]
    next_steps: list[str]


@dataclass
class HarnessEvent:
    kind: str
    data: dict = field(default_factory=dict)


@dataclass
class SessionState:
    context_notes: list[dict] = field(default_factory=list)
    context_start: int = 0
    archived_histories: list[list] = field(default_factory=list)
    task: str = ""
    turns: int = 0
    tool_calls: int = 0
    repairs: int = 0
    running: bool = False
    baseline: Verification | None = None
    last_result: RunResult | None = None
    edit_ledger: list[dict] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    root: str = ""
    head: str | None = None
    fingerprint: str = ""

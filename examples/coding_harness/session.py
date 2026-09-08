"""Explicit JSON snapshots of inert session data; never replay saved actions."""

import json
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter, model_validator

from slick.turns import (
    ModelTurn,
    ToolCall,
    ToolResult,
    UserMessage,
    decode_arguments,
    validate_history,
)

from .agent import CodingAgent
from .state import HarnessConfig, Record, RunResult, SessionState, Verification
from .workspace import Workspace

MAX_SESSION_BYTES = 20 * 1024 * 1024
KINDS = {UserMessage: "user", ModelTurn: "model", ToolResult: "result"}
RECORDS = {kind: cls for cls, kind in KINDS.items()}


def encode_history(history):
    return [{"kind": KINDS[type(item)], **asdict(item)} for item in history]


def decode_history(records):
    history = []
    for record in records:
        data = dict(record)
        kind = data.pop("kind", None)
        if type(kind) is not str or kind not in RECORDS:
            raise ValueError(f"Unknown history tag: {kind!r}")
        cls = RECORDS[kind]
        _check_fields(data, cls)
        if cls is ModelTurn:
            calls = data.get("tool_calls")
            if type(calls) is not list:
                raise ValueError("tool_calls must be a list")
            for call in calls:
                _check_fields(call, ToolCall)
        item = TypeAdapter(cls).validate_json(json.dumps(data, allow_nan=False), strict=True)
        history.append(item)
    return history


def _check_fields(data, cls):
    if type(data) is not dict or set(data) - {field.name for field in fields(cls)}:
        raise ValueError(f"Invalid fields for {cls.__name__}")


class SavedEdit(Record):
    path: str = Field(min_length=1)
    before_sha256: str | None
    after_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    diff: str


class SavedSession(Record):
    version: int = Field(strict=True, ge=1, le=1)
    provider: Literal["openai", "anthropic", "demo"]
    model: str
    root: str
    head: str | None
    fingerprint: str
    config: HarnessConfig
    history: list[dict]
    archived_histories: list[list[dict]]
    task: str
    turns: int = Field(strict=True, ge=0)
    tool_calls: int = Field(strict=True, ge=0)
    repairs: int = Field(strict=True, ge=0)
    edit_ledger: list[SavedEdit]
    last_result: RunResult | None
    baseline: Verification | None

    @model_validator(mode="after")
    def valid_history(self):
        for history in [self.history, *self.archived_histories]:
            if history:
                validate_history(decode_history(history), provider=self.provider, model=self.model)
        return self


def save_session(path: Path, agent: CodingAgent) -> None:
    if agent.state.running:
        raise ValueError("Save requires an idle agent")
    path = Path(path).expanduser()
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"Session destination already exists: {path}")
    path = path.resolve()
    if path.is_relative_to(agent.workspace.root):
        raise ValueError("Save the session outside the worktree")
    state = agent.state
    snapshot = SavedSession(
        version=1,
        provider=state.provider,
        model=state.model,
        root=str(agent.workspace.root),
        head=state.head,
        fingerprint=state.fingerprint,
        config=agent.config,
        history=encode_history(state.history),
        archived_histories=[encode_history(history) for history in state.archived_histories],
        task=state.task,
        turns=state.turns,
        tool_calls=state.tool_calls,
        repairs=state.repairs,
        edit_ledger=agent.workspace.edit_ledger,
        last_result=state.last_result,
        baseline=state.baseline,
    )
    payload = snapshot.model_dump_json(indent=2).encode("utf-8")
    if len(payload) > MAX_SESSION_BYTES:
        raise ValueError("Session exceeds the 20 MiB save limit")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".slick-session-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_session(path: Path) -> SavedSession:
    with Path(path).expanduser().open("rb") as handle:
        payload = handle.read(MAX_SESSION_BYTES + 1)
    if len(payload) > MAX_SESSION_BYTES:
        raise ValueError("Session exceeds the 20 MiB load limit")
    parsed, error = decode_arguments(payload.decode("utf-8"))
    if error:
        raise ValueError(error)
    return SavedSession.model_validate(parsed)


async def restore_session(saved: SavedSession, provider, *, decide, emit) -> CodingAgent:
    identity = provider.identity()
    if identity["provider"] != saved.provider or identity["model"] != saved.model:
        raise ValueError("Session provider/model does not match provider")
    root = Path(saved.root)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("Session workspace root must be an absolute resolved path")
    workspace = Workspace(
        root,
        checks=saved.config.checks,
        decide=decide,
        command_timeout=saved.config.limits.command_timeout,
    )
    await workspace.initialize()
    fingerprint = await workspace.fingerprint()
    agent = CodingAgent(provider, workspace, saved.config, emit=emit)
    workspace.edit_ledger = [entry.model_dump() for entry in saved.edit_ledger]
    agent.state = SessionState(
        provider=saved.provider,
        model=saved.model,
        root=saved.root,
        head=workspace.head,
        fingerprint=fingerprint,
        history=decode_history(saved.history),
        archived_histories=[decode_history(history) for history in saved.archived_histories],
        task=saved.task,
        turns=saved.turns,
        tool_calls=saved.tool_calls,
        repairs=saved.repairs,
        edit_ledger=list(workspace.edit_ledger),
        last_result=saved.last_result,
        baseline=saved.baseline,
    )
    if saved.head != workspace.head or saved.fingerprint != fingerprint:
        agent.state.history.append(
            UserMessage(
                "Workspace changed since the saved session. Previous verification is stale; "
                "read current files and recheck."
            )
        )
        if agent.state.last_result is not None:
            agent.state.last_result = agent.state.last_result.model_copy(
                update={"status": "unverified"}
            )
    return agent

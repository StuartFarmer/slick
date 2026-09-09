"""Explicit JSON snapshots of inert session data; never replay saved actions."""

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from slick import Session
from slick.tools._protocol import (
    decode_arguments,
    make_request,
    validate_requests,
    validate_response,
    validate_results,
)

from .agent import CodingAgent
from .state import HarnessConfig, Record, RunResult, SessionState, Verification
from .workspace import Workspace

MAX_SESSION_BYTES = 20 * 1024 * 1024


def decode_history(records):
    """Validate application transcript records without imposing an API replay order."""
    if not isinstance(records, list):
        raise ValueError("History must be a list")
    normalized = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("History records must be dictionaries")
        data = dict(record)
        kind = data.pop("kind", None)
        if kind == "user":
            if data.keys() != {"text"} or not isinstance(data["text"], str):
                raise ValueError("User record requires string text")
        elif kind == "model":
            if data.keys() != {"text", "tool_requests"}:
                raise ValueError("Model record requires text and tool_requests")
            validate_response((data["text"], data["tool_requests"]))
        elif kind == "result":
            data = validate_results([data])[0]
        else:
            raise ValueError(f"Unknown history tag: {kind!r}")
        normalized.append({"kind": kind, **deepcopy(data)})
    return normalized


def _legacy_request(call, items):
    if not isinstance(call, dict) or set(call) - {"id", "name", "arguments", "argument_error"}:
        raise ValueError("Invalid legacy tool call")
    if not {"id", "name", "arguments"} <= call.keys():
        raise ValueError("Incomplete legacy tool call")
    raw = call["arguments"]
    if raw is None:
        matches = [
            item
            for item in items
            if isinstance(item, dict)
            and (item.get("call_id") == call["id"] or item.get("id") == call["id"])
        ]
        for item in items:
            if isinstance(item, dict):
                matches.extend(
                    candidate
                    for candidate in item.get("tool_calls", [])
                    if candidate.get("id") == call["id"]
                )
        if not matches:
            raise ValueError("Cannot migrate legacy malformed arguments without original input")
        item = matches[0]
        raw = item.get("arguments", item.get("input", item.get("function", {}).get("arguments")))
    return make_request(call["id"], call["name"], raw)


def _migrate_history(records):
    if not isinstance(records, list):
        raise ValueError("Legacy history must be a list")
    converted, pending = [], {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Legacy history record must be an object")
        kind = record.get("kind")
        if pending and kind != "result":
            raise ValueError("Legacy snapshot contains unresolved tool requests")
        if kind == "user":
            converted.append(record)
        elif kind == "model":
            fields = {
                "kind",
                "provider",
                "model",
                "text",
                "tool_calls",
                "items",
                "stop_reason",
                "input_tokens",
                "output_tokens",
            }
            required = {"text", "tool_calls", "items", "stop_reason"}
            if (
                record.keys() - fields
                or not required <= record.keys()
                or not isinstance(record["items"], list)
                or not isinstance(record["tool_calls"], list)
            ):
                raise ValueError("Invalid legacy model record")
            calls = validate_requests(
                [_legacy_request(call, record["items"]) for call in record["tool_calls"]]
            )
            pending = {call["id"]: call for call in calls}
            converted.append({"kind": "model", "text": record["text"], "tool_requests": calls})
        elif kind == "result":
            if (
                record.keys() - {"kind", "call_id", "content", "is_error"}
                or not {"call_id", "content"} <= record.keys()
            ):
                raise ValueError("Invalid legacy tool result")
            call = pending.pop(record.get("call_id"), None)
            if call is None:
                raise ValueError("Cannot migrate result without matching legacy request")
            converted.append(
                {
                    "kind": "result",
                    "request": call,
                    "content": record["content"],
                    "is_error": record.get("is_error", False),
                }
            )
        else:
            raise ValueError(f"Unknown legacy history tag: {kind!r}")
    if pending:
        raise ValueError("Legacy snapshot contains unresolved tool requests")
    return decode_history(converted)


def _legacy_views(history: list, pending_results=()) -> list[dict]:
    """Render application observations, excluding the separately supplied pending batch."""
    pending = [result["request"] for result in pending_results]
    pending_start = next(
        (
            index
            for index in range(len(history) - 1, -1, -1)
            if pending and history[index].get("tool_requests") == pending
        ),
        None,
    )
    views = []
    for index, item in enumerate(history):
        kind = item["kind"]
        if kind == "user":
            views.append({"role": "user", "text": item["text"]})
        elif kind == "model":
            calls = [] if index == pending_start else item["tool_requests"]
            views.append({"role": "assistant", "text": item["text"], "calls": calls})
        elif kind == "result":
            if pending_start is not None and index > pending_start and item["request"] in pending:
                continue
            views.append(
                {
                    "role": "tool",
                    "id": item["request"]["id"],
                    "text": item["content"],
                    "error": item.get("is_error", False),
                }
            )
        else:
            raise ValueError(f"Unknown history record: {kind!r}")
    return views


class ContextView(Record):
    role: Literal["user", "assistant", "tool"]
    text: str
    calls: list[dict] = Field(default_factory=list)
    id: str | None = None
    error: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def valid_calls(self):
        validate_requests(self.calls)
        if self.role == "tool" and (not self.id or not self.id.strip()):
            raise ValueError("Tool context requires an ID")
        return self


class ContextNote(ContextView):
    after: int = Field(strict=True, ge=0)


def _context_views(records, cls=ContextView):
    return [cls.model_validate(record).model_dump(exclude_unset=True) for record in records]


class SavedEdit(Record):
    path: str = Field(min_length=1)
    before_sha256: str | None
    after_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    diff: str


class SavedSession(Record):
    version: int = Field(strict=True, ge=3, le=3)
    provider: Literal["openai", "anthropic", "litellm", "demo"]
    model: str
    root: str
    head: str | None
    fingerprint: str
    config: HarnessConfig
    session: dict
    context_notes: list[dict]
    context_start: int = Field(strict=True, ge=0)
    archived_histories: list[list[dict]]
    task: str
    turns: int = Field(strict=True, ge=0)
    tool_calls: int = Field(strict=True, ge=0)
    repairs: int = Field(strict=True, ge=0)
    edit_ledger: list[SavedEdit]
    last_result: RunResult | None
    baseline: Verification | None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy(cls, data):
        if not isinstance(data, dict) or type(data.get("version")) is not int:
            return data
        if data["version"] not in {1, 2}:
            return data
        if {"session", "context_notes", "context_start"} & data.keys():
            raise ValueError("Legacy snapshot contains new-format fields")
        data = deepcopy(data)
        if data["version"] == 1:
            data["history"] = _migrate_history(data.get("history", []))
            data["archived_histories"] = [
                _migrate_history(history) for history in data.get("archived_histories", [])
            ]
            data["pending_results"] = []
        history = decode_history(data.pop("history", []))
        results = validate_results(data.pop("pending_results", []))
        data["context_notes"] = [{"after": 0, **view} for view in _legacy_views(history, results)]
        data["context_start"] = 0
        data["archived_histories"] = [
            _legacy_views(decode_history(history)) for history in data.get("archived_histories", [])
        ]
        exchange = {
            "context": None,
            "text": "",
            "work": [
                {"request": item["request"], "result": item, "submitted": False} for item in results
            ],
        }
        data["session"] = {"version": 1, "history": [exchange] if results else []}
        data["version"] = 3
        return data

    @model_validator(mode="after")
    def valid_history(self):
        self.session = Session.from_dict(self.session).to_dict()
        length = len(self.session["history"])
        self.context_notes = _context_views(self.context_notes, ContextNote)
        if self.context_start > length or any(
            note["after"] > length for note in self.context_notes
        ):
            raise ValueError("Context position exceeds Session history")
        self.archived_histories = [_context_views(history) for history in self.archived_histories]
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
        version=3,
        provider=state.provider,
        model=state.model,
        root=str(agent.workspace.root),
        head=state.head,
        fingerprint=state.fingerprint,
        config=agent.config,
        session=agent.session.to_dict(),
        context_notes=deepcopy(state.context_notes),
        context_start=state.context_start,
        archived_histories=deepcopy(state.archived_histories),
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
    agent.session = Session.from_dict(saved.session, provider=provider, tools=agent.session.tools)
    workspace.edit_ledger = [entry.model_dump() for entry in saved.edit_ledger]
    agent.state = SessionState(
        provider=identity["provider"],
        model=identity["model"],
        root=saved.root,
        head=workspace.head,
        fingerprint=fingerprint,
        context_notes=deepcopy(saved.context_notes),
        context_start=saved.context_start,
        archived_histories=deepcopy(saved.archived_histories),
        task=saved.task,
        turns=saved.turns,
        tool_calls=saved.tool_calls,
        repairs=saved.repairs,
        edit_ledger=list(workspace.edit_ledger),
        last_result=saved.last_result,
        baseline=saved.baseline,
    )
    if saved.head != workspace.head or saved.fingerprint != fingerprint:
        agent._note(
            "Workspace changed since the saved session. Previous verification is stale; "
            "read current files and recheck."
        )
        if agent.state.last_result is not None:
            agent.state.last_result = agent.state.last_result.model_copy(
                update={"status": "unverified"}
            )
    return agent

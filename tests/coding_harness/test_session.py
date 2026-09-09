"""Persist interaction work separately from application context and workspace state."""

import asyncio
import json
import stat
from copy import deepcopy

import pytest

from examples.coding_harness.agent import CodingAgent
from examples.coding_harness.session import load_session, restore_session, save_session
from examples.coding_harness.state import HarnessConfig
from tests.test_session_calls import Script


class FakeProvider(Script):
    def identity(self):
        return {"provider": "demo", "model": "scripted"}


async def deny(request):
    return "deny"


def legacy(root, version=2):
    call = {"id": "a", "name": "read_file", "arguments": {"path": "sample.py"}}
    history = [
        {"kind": "user", "text": "Read only"},
        {"kind": "model", "text": "Reading", "tool_requests": [call]},
        {"kind": "result", "request": call, "content": "observed", "is_error": False},
    ]
    if version == 1:
        history[1] = {
            "kind": "model",
            "text": "Reading",
            "tool_calls": [call],
            "items": [],
            "stop_reason": "tool_calls",
            "provider": "demo",
            "model": "old",
        }
        history[2] = {"kind": "result", "call_id": "a", "content": "observed", "is_error": False}
    return {
        "version": version,
        "provider": "demo",
        "model": "old",
        "root": str(root),
        "head": None,
        "fingerprint": "old",
        "config": {},
        "history": history,
        "archived_histories": [deepcopy(history)],
        "task": "Read",
        "turns": 1,
        "tool_calls": 1,
        "repairs": 0,
        "edit_ledger": [],
        "last_result": None,
        "baseline": None,
        **({"pending_results": []} if version == 2 else {}),
    }


@pytest.mark.parametrize("completed", [0, 1, 2])
def test_v3_partial_work_restores_inertly_and_resumes_without_repeating(
    workspace, tmp_path, completed
):
    calls = [
        {"id": "a", "name": "create_file", "arguments": {"path": "a.py", "content": "1"}},
        {"id": "b", "name": "create_file", "arguments": {"path": "b.py", "content": "2"}},
    ]
    agent = CodingAgent(FakeProvider(("Writing", calls)), workspace, HarnessConfig())
    agent._note("Create files")
    asyncio.run(agent.session.acall("Create files"))
    for call in calls[:completed]:
        asyncio.run(agent.session.resolve(call))
    agent.state.fingerprint = asyncio.run(workspace.fingerprint())
    path = tmp_path.parent / f"{tmp_path.name}-partial.json"
    save_session(path, agent)
    saved = load_session(path)
    assert saved.version == 3
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    next_provider = FakeProvider(("Done", []))
    restored = asyncio.run(restore_session(saved, next_provider, decide=deny, emit=lambda e: None))
    assert restored.session.to_dict() == agent.session.to_dict()
    assert restored.state.context_notes == agent.state.context_notes
    assert [(workspace.root / name).exists() for name in ("a.py", "b.py")] == [
        completed >= 1,
        completed == 2,
    ]
    result = asyncio.run(restored.run("Continue"))
    assert result.status == "unverified"
    assert [entry["path"] for entry in restored.workspace.edit_ledger] == ["a.py", "b.py"]
    assert len(next_provider.inputs[0][1]) == 2
    assert all(not item["is_error"] for item in next_provider.inputs[0][1])


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_history_migrates_without_scheduling_actions(workspace, tmp_path, version):
    path = tmp_path / "legacy.json"
    original = json.dumps(legacy(workspace.root, version))
    path.write_text(original)
    saved = load_session(path)
    assert saved.version == 3
    assert saved.session == {"version": 1, "history": []}
    assert [note["role"] for note in saved.context_notes] == ["user", "assistant", "tool"]
    assert saved.context_notes[-1]["text"] == "observed"
    assert len(saved.archived_histories) == 1
    assert path.read_text() == original


def test_legacy_ready_results_are_submitted_but_not_executed(workspace, tmp_path):
    data = legacy(workspace.root)
    data["pending_results"] = [
        {key: value for key, value in data["history"][-1].items() if key != "kind"}
    ]
    path = tmp_path / "legacy-ready.json"
    path.write_text(json.dumps(data))
    provider = FakeProvider(("Done", []))
    saved = load_session(path)
    restored = asyncio.run(restore_session(saved, provider, decide=deny, emit=lambda e: None))
    assert restored.session.pending_requests == []
    assert not any(note["role"] == "tool" for note in saved.context_notes)
    assert saved.session["history"][0]["context"] is None
    asyncio.run(restored.run("Continue"))
    assert provider.inputs[0][1] == data["pending_results"]
    assert "observed" not in provider.inputs[0][0]
    assert restored.state.tool_calls == 0


@pytest.mark.parametrize(
    "record",
    [
        {"kind": "model", "items": []},
        {"kind": "result", "call_id": "a"},
        {
            "kind": "model",
            "text": "",
            "items": [],
            "stop_reason": "tool_calls",
            "tool_calls": [{"id": "a", "name": "read", "arguments": {}}],
        },
    ],
)
def test_incomplete_legacy_records_fail_clearly(record):
    from examples.coding_harness.session import _migrate_history

    with pytest.raises(ValueError):
        _migrate_history([record])


def test_save_restore_does_not_restore_allowances_or_overwrite(workspace, tmp_path):
    agent = CodingAgent(FakeProvider(), workspace, HarnessConfig())
    agent._note("Read only")
    agent.state.fingerprint = asyncio.run(workspace.fingerprint())
    workspace.allowed_commands.add((str(workspace.root), ("unapproved",)))
    path = tmp_path.parent / (tmp_path.name + "-session.json")
    save_session(path, agent)
    restored = asyncio.run(
        restore_session(load_session(path), FakeProvider(), decide=deny, emit=lambda e: None)
    )
    assert restored.state.context_notes == agent.state.context_notes
    assert restored.workspace.allowed_commands == restored.workspace.configured_commands
    with pytest.raises(FileExistsError):
        save_session(path, agent)


def test_changed_workspace_appends_context_warning(workspace, tmp_path):
    agent = CodingAgent(FakeProvider(), workspace, HarnessConfig())
    agent._note("Read only")
    agent.state.fingerprint = asyncio.run(workspace.fingerprint())
    path = tmp_path.parent / (tmp_path.name + "-changed.json")
    save_session(path, agent)
    (workspace.root / "new.py").write_text("changed")
    restored = asyncio.run(
        restore_session(load_session(path), FakeProvider(), decide=deny, emit=lambda e: None)
    )
    assert "changed" in restored.state.context_notes[-1]["text"].lower()
    assert (workspace.root / "new.py").read_text() == "changed"


def test_save_requires_idle_external_path_and_known_version(workspace, tmp_path):
    agent = CodingAgent(FakeProvider(), workspace, HarnessConfig())
    with pytest.raises(ValueError, match="outside"):
        save_session(workspace.root / "session.json", agent)
    agent.state.running = True
    with pytest.raises(ValueError, match="idle"):
        save_session(tmp_path.parent / "unused.json", agent)
    path = tmp_path / "invalid.json"
    path.write_text('{"version": 99}')
    with pytest.raises(ValueError):
        load_session(path)


@pytest.mark.parametrize("corruption", ["version", "context", "cursor", "session", "ledger"])
def test_loaded_snapshots_are_strictly_validated(workspace, tmp_path, corruption):
    agent = CodingAgent(FakeProvider(), workspace, HarnessConfig())
    agent._note("Hello")
    path = tmp_path.parent / (tmp_path.name + "-schema.json")
    save_session(path, agent)
    data = json.loads(path.read_text())
    if corruption == "version":
        data["version"] = True
    elif corruption == "context":
        data["context_notes"][0]["after"] = 99
    elif corruption == "cursor":
        data["context_start"] = 99
    elif corruption == "session":
        data["session"]["unexpected"] = True
    else:
        data["edit_ledger"] = [{"nonsense": True}]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_session(path)


def test_save_rejects_dangling_symlink(workspace, tmp_path):
    agent = CodingAgent(FakeProvider(), workspace, HarnessConfig())
    destination = tmp_path.parent / (tmp_path.name + "-link.json")
    target = tmp_path.parent / (tmp_path.name + "-target.json")
    destination.symlink_to(target)
    with pytest.raises((ValueError, FileExistsError)):
        save_session(destination, agent)
    assert not target.exists()


def test_archives_ledger_and_empty_conversation_round_trip(workspace, tmp_path):
    agent = CodingAgent(FakeProvider(), workspace, HarnessConfig())
    workspace.create_file("new.py", "value = 2\n")
    agent.state.fingerprint = asyncio.run(workspace.fingerprint())
    agent.state.archived_histories = [[{"role": "user", "text": "Old context"}]]
    path = tmp_path.parent / (tmp_path.name + "-empty.json")
    save_session(path, agent)
    restored = asyncio.run(
        restore_session(load_session(path), FakeProvider(), decide=deny, emit=lambda e: None)
    )
    assert restored.session.history == []
    assert restored.state.archived_histories == agent.state.archived_histories
    assert restored.workspace.edit_ledger == workspace.edit_ledger
    assert (workspace.root / "new.py").read_text() == "value = 2\n"

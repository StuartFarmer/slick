import asyncio
import json
import stat
from copy import deepcopy

import pytest

from examples.coding_harness.agent import CodingAgent, create_agent
from examples.coding_harness.config import HarnessConfig
from examples.coding_harness.session import compact, load, save
from tests.coding_harness.test_agent import Script, call


def test_save_resume_keeps_conversation_but_never_schedules_tools(workspace, tmp_path):
    provider = Script(
        ("", [call("create_file", {"path": "made.txt", "content": "once"})]), ("Done", [])
    )
    agent = CodingAgent(provider, workspace, HarnessConfig())
    asyncio.run(agent.run("Make a file"))
    path = tmp_path.parent / (tmp_path.name + ".json")
    save(path, agent)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    saved = load(path)
    provider = Script(("Continued", []))
    restored = asyncio.run(create_agent(provider, saved.root, saved.config, saved=saved))
    assert not provider.inputs
    assert restored.messages[: len(agent.messages)] == agent.messages
    assert not restored.workspace.allowed_commands
    assert (workspace.root / "made.txt").read_text() == "once"
    assert asyncio.run(restored.run("Continue"))["answer"] == "Continued"
    assert restored.workspace.edited_paths == set()


def test_save_rejects_active_internal_existing_and_symlink_paths(workspace, tmp_path):
    agent = CodingAgent(Script(), workspace, HarnessConfig())
    with pytest.raises(ValueError, match="outside"):
        save(workspace.root / "session.json", agent)
    path = tmp_path.parent / (tmp_path.name + ".json")
    agent.running = True
    with pytest.raises(ValueError, match="idle"):
        save(path, agent)
    agent.running = False
    path.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        save(path, agent)
    path.unlink()
    save(path, agent)
    with pytest.raises(FileExistsError):
        save(path, agent)


def test_old_format_is_rejected_explicitly(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"session": {"history": []}}))
    with pytest.raises(ValueError, match="Unsupported session format"):
        load(path)


def test_compaction_preserves_recent_exchange_and_failure_preserves_messages(workspace):
    messages = [
        {"role": "user", "text": "Fix the bug"},
        {"role": "assistant", "text": "Inspected", "calls": []},
        {"role": "assistant", "text": "Reading", "calls": [call("read_file")]},
        {"role": "tool", "text": "exact recent output", "id": "one", "error": False},
    ]
    original = deepcopy(messages)
    summary = asyncio.run(compact(Script(("The task is to fix the bug.", [])), messages, 10000))
    assert "The task is to fix the bug." in summary[0]["text"]
    assert summary[1:] == messages[2:]
    assert messages == original
    agent = CodingAgent(Script(("", [])), workspace, HarnessConfig())
    agent.messages = messages
    with pytest.raises(ValueError, match="nonempty summary"):
        asyncio.run(agent.compact())
    assert agent.messages == original
    assert not agent.running


def test_changed_workspace_is_not_restored_as_verified(workspace, tmp_path):
    agent = CodingAgent(Script(), workspace, HarnessConfig())
    agent.fingerprint = asyncio.run(workspace.fingerprint())
    path = tmp_path.parent / (tmp_path.name + ".json")
    save(path, agent)
    (workspace.root / "sample.py").write_text("external change")
    saved = load(path)
    restored = asyncio.run(create_agent(Script(), saved.root, saved.config, saved=saved))
    assert restored.last_result is None
    assert "Workspace changed" in restored.messages[-1]["text"]


def test_new_conversation_keeps_files(workspace):
    agent = CodingAgent(Script(("Done", [])), workspace, HarnessConfig())
    asyncio.run(agent.run("Hello"))
    before = (workspace.root / "sample.py").read_bytes()
    asyncio.run(agent.new_conversation())
    assert agent.messages == []
    assert agent.last_result is None
    assert (workspace.root / "sample.py").read_bytes() == before

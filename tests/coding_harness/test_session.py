import asyncio
import json

import pytest

from examples.coding_harness.agent import CodingAgent
from examples.coding_harness.session import load_session, restore_session, save_session
from examples.coding_harness.state import HarnessConfig
from slick.turns import UserMessage


class Backend:
    def identity(self):
        return {"provider": "demo", "model": "scripted"}


async def deny(request):
    return "deny"


def test_explicit_save_restore_is_inert_and_does_not_restore_allowances(workspace, tmp_path):
    agent = CodingAgent(Backend(), workspace, HarnessConfig())
    agent.state.history = [UserMessage("Read only")]
    agent.state.fingerprint = asyncio.run(workspace.fingerprint())
    path = tmp_path.parent / (tmp_path.name + "-session.json")
    workspace.allowed_commands.add((str(workspace.root), ("unapproved",)))
    save_session(path, agent)
    restored = asyncio.run(
        restore_session(load_session(path), Backend(), decide=deny, emit=lambda e: None)
    )
    assert restored.state.history == agent.state.history
    assert restored.workspace.allowed_commands == restored.workspace.configured_commands
    with pytest.raises(FileExistsError):
        save_session(path, agent)


def test_saved_changed_workspace_invalidates_prior_context(workspace, tmp_path):
    agent = CodingAgent(Backend(), workspace, HarnessConfig())
    agent.state.history = [UserMessage("Read only")]
    agent.state.fingerprint = asyncio.run(workspace.fingerprint())
    path = tmp_path.parent / (tmp_path.name + "-changed.json")
    save_session(path, agent)
    (workspace.root / "new.py").write_text("changed")
    restored = asyncio.run(
        restore_session(load_session(path), Backend(), decide=deny, emit=lambda e: None)
    )
    assert "changed" in restored.state.history[-1].text.lower()
    assert (workspace.root / "new.py").read_text() == "changed"


def test_save_requires_idle_external_path_and_known_json_version(workspace, tmp_path):
    agent = CodingAgent(Backend(), workspace, HarnessConfig())
    agent.state.history = [UserMessage("hello")]
    with pytest.raises(ValueError, match="outside"):
        save_session(workspace.root / "session.json", agent)
    agent.state.running = True
    with pytest.raises(ValueError, match="idle"):
        save_session(tmp_path.parent / "unused.json", agent)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"version": 99}))
    with pytest.raises(ValueError):
        load_session(path)


@pytest.mark.parametrize("corruption", ["version", "items", "arguments", "ledger"])
def test_loaded_records_are_strictly_validated(workspace, tmp_path, corruption):
    from slick.turns import ModelTurn, ToolCall, ToolResult

    agent = CodingAgent(Backend(), workspace, HarnessConfig())
    agent.state.history = [
        UserMessage("Read"),
        ModelTurn(
            "demo",
            "scripted",
            "",
            [ToolCall("a", "read_file", {"path": "sample.py"})],
            [],
            "tool_calls",
        ),
        ToolResult("a", "read"),
    ]
    path = tmp_path.parent / (tmp_path.name + "-schema.json")
    save_session(path, agent)
    data = json.loads(path.read_text())
    if corruption == "version":
        data["version"] = True
    elif corruption == "items":
        data["history"][1]["items"] = "not a list"
    elif corruption == "arguments":
        data["history"][1]["tool_calls"][0]["arguments"] = ["not", "an", "object"]
    else:
        data["edit_ledger"] = [{"nonsense": True}]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_session(path)


def test_save_rejects_dangling_symlink_destination(workspace, tmp_path):
    agent = CodingAgent(Backend(), workspace, HarnessConfig())
    agent.state.history = [UserMessage("Read")]
    destination = tmp_path.parent / (tmp_path.name + "-link.json")
    target = tmp_path.parent / (tmp_path.name + "-target.json")
    destination.symlink_to(target)
    with pytest.raises((ValueError, FileExistsError)):
        save_session(destination, agent)
    assert not target.exists()


def test_native_groups_archives_and_edit_ledger_round_trip(workspace, tmp_path):
    import stat

    from slick.turns import ModelTurn, ToolCall, ToolResult

    agent = CodingAgent(Backend(), workspace, HarnessConfig())
    workspace.create_file("new.py", "value = 2\n")
    agent.state.fingerprint = asyncio.run(workspace.fingerprint())
    group = [
        UserMessage("Inspect"),
        ModelTurn(
            "demo",
            "scripted",
            "Reading",
            [
                ToolCall("a", "read_file", {"path": "new.py"}),
                ToolCall("b", "git_diff", {}),
            ],
            [{"type": "reasoning", "opaque": "preserve me"}],
            "tool_calls",
        ),
        ToolResult("a", "file observation"),
        ToolResult("b", "diff observation"),
    ]
    agent.state.history = group
    agent.state.archived_histories = [group]
    path = tmp_path.parent / (tmp_path.name + "-roundtrip.json")
    before = (workspace.root / "new.py").read_bytes()
    save_session(path, agent)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    restored = asyncio.run(
        restore_session(load_session(path), Backend(), decide=deny, emit=lambda e: None)
    )
    assert restored.state.history == group
    assert restored.state.archived_histories == [group]
    assert restored.workspace.edit_ledger == workspace.edit_ledger
    assert (workspace.root / "new.py").read_bytes() == before


def test_fresh_idle_conversation_can_be_saved_and_resumed(workspace, tmp_path):
    agent = CodingAgent(Backend(), workspace, HarnessConfig())
    agent.state.fingerprint = asyncio.run(workspace.fingerprint())
    path = tmp_path.parent / (tmp_path.name + "-empty.json")
    save_session(path, agent)
    restored = asyncio.run(
        restore_session(load_session(path), Backend(), decide=deny, emit=lambda e: None)
    )
    assert restored.state.history == []

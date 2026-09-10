import asyncio
import json
import sys

import pytest

from examples.coding_harness.agent import CodingAgent, create_agent
from examples.coding_harness.checks import Check
from examples.coding_harness.config import HarnessConfig, Limits
from examples.coding_harness.demo import DemoProvider, create_demo
from examples.coding_harness.ui import ConsoleUI
from slick.tools import tool


class Script:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.inputs = []

    def identity(self):
        return {"provider": "demo", "model": "test"}

    async def acall(self, context, *, tools=None, tool_results=None):
        self.inputs.append((context, tool_results))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def call(name, arguments=None, id="one"):
    return {"id": id, "name": name, "arguments": arguments or {}}


def test_demo_repairs_actual_code(tmp_path):
    config = create_demo(tmp_path)

    async def run():
        agent = await create_agent(DemoProvider(tmp_path), tmp_path, config)
        result = await agent.run("Fix the total calculation")
        assert result["status"] == "verified"
        assert result["repairs"] == 1
        assert all(item["command"]["exit_code"] == 0 for item in result["checks"])
        assert (
            tmp_path / "pricing.py"
        ).read_text() == "def total(values):\n    return sum(values)\n"

    asyncio.run(run())


def test_followup_sees_conversation_and_real_tool_result(workspace):
    provider = Script(
        ("Looking", [call("read_file", {"path": "sample.py"})]), ("Done", []), ("Again", [])
    )
    agent = CodingAgent(provider, workspace, HarnessConfig())
    assert asyncio.run(agent.run("Read the file"))["status"] == "unverified"
    observation = json.loads(provider.inputs[1][1][0]["content"])
    assert observation["text"] == "value = 1\n"
    asyncio.run(agent.run("What did you find?"))
    assert "Read the file" in provider.inputs[2][0]
    assert "value = 1" in provider.inputs[2][0]
    assert "Looking" in provider.inputs[2][0]
    assert provider.inputs[2][0].count("You are a coding assistant") == 1


@pytest.mark.parametrize("tool_request", [call("unknown"), call("create_file", {"wrong": "x"})])
def test_invalid_calls_become_observations_without_effects(workspace, tool_request):
    before = asyncio.run(workspace.fingerprint())
    provider = Script(("", [tool_request]), ("Recovered", []))
    agent = CodingAgent(provider, workspace, HarnessConfig())
    result = asyncio.run(agent.run("Try a tool"))
    assert result["answer"] == "Recovered"
    assert provider.inputs[1][1][0]["is_error"]
    assert asyncio.run(workspace.fingerprint()) == before


def test_budget_stops_remaining_tools_without_replay(workspace):
    calls = [call("create_file", {"path": f"{i}.txt", "content": "x"}, str(i)) for i in range(2)]
    agent = CodingAgent(
        Script(("", calls), ("Followup", [])),
        workspace,
        HarnessConfig(limits=Limits(max_tool_calls=1)),
    )
    assert asyncio.run(agent.run("Create files"))["status"] == "blocked"
    assert (workspace.root / "0.txt").exists()
    assert not (workspace.root / "1.txt").exists()
    asyncio.run(agent.run("Continue"))
    assert not (workspace.root / "1.txt").exists()
    assert any(m.get("error") for m in agent.messages)


def test_effect_is_not_repeated_after_provider_failure(workspace):
    provider = Script(
        ("", [call("create_file", {"path": "new.txt", "content": "once"})]),
        RuntimeError("offline failure"),
        ("Followup", []),
    )
    agent = CodingAgent(provider, workspace, HarnessConfig())
    assert asyncio.run(agent.run("Create"))["status"] == "failed"
    assert (workspace.root / "new.txt").read_text() == "once"
    asyncio.run(agent.run("Continue"))
    assert len(workspace.edited_paths) == 1
    assert "new.txt" in provider.inputs[-1][0]


def test_cancelled_tool_is_recorded_and_followup_is_possible(workspace):
    started = asyncio.Event()
    cleaned = []

    async def wait() -> str:
        """Wait until cancelled."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    provider = Script(
        ("", [call("wait"), call("create_file", {"path": "late.txt", "content": "x"}, "two")]),
        ("Followup", []),
    )
    agent = CodingAgent(provider, workspace, HarnessConfig())
    agent.tools["wait"] = tool(wait)

    async def run():
        task = asyncio.create_task(agent.run("Wait"))
        await asyncio.wait_for(started.wait(), 5)
        with pytest.raises(RuntimeError, match="already active"):
            await agent.run("Overlap")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned == [True]
        assert agent.last_result["status"] == "cancelled"
        assert not agent.running
        assert not (workspace.root / "late.txt").exists()
        assert (await agent.run("Continue"))["status"] == "unverified"

    asyncio.run(run())


def test_failing_checks_stop_without_progress(workspace):
    checks = [Check(name="unit", argv=[sys.executable, "-c", "raise SystemExit(1)"])]
    workspace.allowed_commands.add((str(workspace.root), tuple(checks[0].argv)))
    agent = CodingAgent(Script(("Done", []), ("Done", [])), workspace, HarnessConfig(checks=checks))
    result = asyncio.run(agent.run("Fix"))
    assert result["status"] == "blocked"
    assert result["repairs"] == 1
    assert result["checks"][0]["command"]["exit_code"] == 1


def test_model_and_context_limits_stop_before_extra_requests(workspace):
    provider = Script(("", [call("list_files")]))
    agent = CodingAgent(provider, workspace, HarnessConfig(limits=Limits(max_turns=1)))
    assert asyncio.run(agent.run("Inspect"))["status"] == "blocked"
    assert len(provider.inputs) == 1
    agent = CodingAgent(
        Script(),
        workspace,
        HarnessConfig(limits=Limits(context_soft_chars=100, context_hard_chars=200)),
    )
    assert asyncio.run(agent.run("x" * 300))["status"] == "blocked"
    assert not agent.provider.inputs


def test_initial_soft_limit_does_not_spend_a_request_on_impossible_compaction(workspace):
    provider = Script(("Done", []))
    agent = CodingAgent(
        provider,
        workspace,
        HarnessConfig(limits=Limits(max_turns=1, context_soft_chars=100)),
    )
    result = asyncio.run(agent.run("Inspect"))
    assert result["status"] == "unverified"
    assert result["turns"] == len(provider.inputs) == 1


def test_files_changed_after_checks_cannot_be_reported_verified(workspace):
    checks = [Check(name="unit", argv=[sys.executable, "-c", "print('ok')"])]
    workspace.allowed_commands.add((str(workspace.root), tuple(checks[0].argv)))

    class ExternalWriter(ConsoleUI):
        def checks(self, results, baseline=False):
            if not baseline:
                source = workspace.root / "sample.py"
                source.write_text(source.read_text() + "# external change\n")

    agent = CodingAgent(
        Script(("Done", []), ("Done", [])),
        workspace,
        HarnessConfig(checks=checks, limits=Limits(max_repairs=1)),
        ui=ExternalWriter(),
    )
    assert asyncio.run(agent.run("Check"))["status"] == "blocked"
    assert "results are stale" in agent.provider.inputs[1][0]
    assert "results are stale" in agent.last_result["answer"]

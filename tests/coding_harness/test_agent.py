import asyncio
import sys
from collections import deque

import pytest

from examples.coding_harness.agent import CodingAgent
from examples.coding_harness.state import Check, HarnessConfig, Limits
from slick import Session


class Script:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.requests = []

    def identity(self):
        return {"provider": "demo", "model": "scripted"}

    async def acall(self, context, *, tools=None, tool_results=""):
        self.requests.append({"context": context, "tool_results": tool_results or []})
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def turn(text="", calls=()):
    return (text, list(calls))


def test_model_cannot_declare_unchecked_work_verified(workspace):
    agent = CodingAgent(Script(turn("Everything is verified!")), workspace, HarnessConfig())
    result = asyncio.run(agent.run("Explain the code"))
    assert result.status == "unverified"
    assert result.turns == 1
    assert not agent.state.running


def test_unknown_and_invalid_tools_do_not_execute(workspace):
    provider = Script(
        turn(
            calls=[
                {"id": "a", "name": "missing", "arguments": {}},
                {"id": "b", "name": "create_file", "arguments": {"path": "bad.py"}},
            ]
        ),
        turn("Done"),
    )
    agent = CodingAgent(provider, workspace, HarnessConfig())
    asyncio.run(agent.run("Read"))
    results = [item for item in provider.requests[1]["tool_results"] if "request" in item]
    assert len(results) == 2 and all(item["is_error"] for item in results)
    assert not (workspace.root / "bad.py").exists()


def test_budget_closes_call_group_without_extra_effects(workspace):
    provider = Script(
        turn(
            calls=[
                {"id": "a", "name": "create_file", "arguments": {"path": "a.py", "content": "1"}},
                {"id": "b", "name": "create_file", "arguments": {"path": "b.py", "content": "2"}},
            ]
        )
    )
    config = HarnessConfig(limits=Limits(max_tool_calls=1))
    agent = CodingAgent(provider, workspace, config)
    result = asyncio.run(agent.run("Create files"))
    assert result.status == "blocked"
    assert (workspace.root / "a.py").exists()
    assert not (workspace.root / "b.py").exists()
    results = [work["result"] for exchange in agent.session.history for work in exchange["work"]]
    assert [item["request"]["id"] for item in results] == ["a", "b"]
    assert results[-1]["is_error"]


def test_request_failure_does_not_fabricate_turn_and_can_continue(workspace):
    provider = Script(RuntimeError("offline transport failed"), turn("Recovered"))
    agent = CodingAgent(provider, workspace, HarnessConfig())
    first = asyncio.run(agent.run("Fix"))
    assert first.status == "failed"
    assert agent.session.history == []
    second = asyncio.run(agent.run("Try again"))
    assert second.answer == "Recovered"


def test_cancelled_api_records_status_and_rejects_overlapping_run(workspace):
    async def scenario():
        started = asyncio.Event()

        class Waiting(Script):
            async def acall(self, *args, **kwargs):
                started.set()
                await asyncio.Event().wait()

        agent = CodingAgent(Waiting(), workspace, HarnessConfig())
        task = asyncio.create_task(agent.run("Wait"))
        await asyncio.wait_for(started.wait(), 2)
        with pytest.raises(RuntimeError, match="active"):
            await agent.run("Overlap")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert agent.state.last_result.status == "cancelled"
        assert not agent.state.running

    asyncio.run(scenario())


def test_call_ids_are_scoped_to_independent_requests(workspace):
    provider = Script(
        turn(
            calls=[
                {"id": "same", "name": "create_file", "arguments": {"path": "a.py", "content": "1"}}
            ]
        ),
        turn(
            calls=[
                {"id": "same", "name": "create_file", "arguments": {"path": "b.py", "content": "2"}}
            ]
        ),
    )
    provider.responses.append(turn("Done"))
    agent = CodingAgent(provider, workspace, HarnessConfig())
    result = asyncio.run(agent.run("Create"))
    assert result.status == "unverified"
    assert (workspace.root / "a.py").exists()
    assert (workspace.root / "b.py").exists()


def test_request_budget_prevents_next_request(workspace):
    provider = Script(
        turn(calls=[{"id": "read", "name": "read_file", "arguments": {"path": "sample.py"}}])
    )
    agent = CodingAgent(provider, workspace, HarnessConfig(limits=Limits(max_turns=1)))
    result = asyncio.run(agent.run("Inspect"))
    assert result.status == "blocked" and result.turns == 1
    assert len(provider.requests) == 1


def test_task_deadline_stops_waiting_request(workspace):
    class Waiting(Script):
        async def acall(self, *args, **kwargs):
            await asyncio.Event().wait()

    agent = CodingAgent(Waiting(), workspace, HarnessConfig(limits=Limits(task_timeout=1)))
    result = asyncio.run(agent.run("Wait"))
    assert result.status == "blocked" and "deadline" in result.answer
    assert result.turns == 1 and (not agent.state.running)


def test_serialization_failure_after_effect_is_not_retried(workspace):
    def broken_return() -> int:
        """Write once then return an invalid result."""
        with (workspace.root / "effects.txt").open("a") as stream:
            stream.write("effect\n")
        return "invalid"

    provider = Script(
        turn(calls=[{"id": "write", "name": "broken_return", "arguments": {}}]), turn("Stopped")
    )
    agent = CodingAgent(provider, workspace, HarnessConfig())
    agent.session = Session(provider=provider, tools=[broken_return])
    asyncio.run(agent.run("Write once"))
    assert (workspace.root / "effects.txt").read_text() == "effect\n"
    result = next(item for item in provider.requests[1]["tool_results"] if "request" in item)
    assert result["is_error"] and "effects may have occurred" in result["content"]


def test_cancellation_completes_multicall_history_without_later_effect(workspace):
    async def scenario():
        started = asyncio.Event()

        async def slow_write() -> str:
            """Write then wait."""
            (workspace.root / "first.txt").write_text("effect")
            started.set()
            await asyncio.Event().wait()
            return "done"

        provider = Script(
            turn(
                calls=[
                    {"id": "first", "name": "slow_write", "arguments": {}},
                    {
                        "id": "second",
                        "name": "create_file",
                        "arguments": {"path": "second.txt", "content": "no"},
                    },
                ]
            )
        )
        agent = CodingAgent(provider, workspace, HarnessConfig())
        agent.session = Session(provider=provider, tools=[*agent.session.tools, slow_write])
        task = asyncio.create_task(agent.run("Write"))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        results = [
            work["result"] for exchange in agent.session.history for work in exchange["work"]
        ]
        assert [item["request"]["id"] for item in results] == ["first", "second"]
        assert "effects may have occurred" in results[0]["content"]
        assert "Not started" in results[1]["content"]
        assert (workspace.root / "first.txt").read_text() == "effect"
        assert not (workspace.root / "second.txt").exists()
        assert len(agent.session.ready_results) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("change", [False, True])
def test_failed_checks_stop_at_no_progress_or_repair_budget(workspace, change):
    check = Check(name="failure", argv=[sys.executable, "-c", "raise SystemExit(1)"])
    workspace.allowed_commands.add((str(workspace.root), tuple(check.argv)))
    responses = [turn("First attempt")]
    if change:
        responses.append(
            turn(
                calls=[
                    {
                        "id": "edit",
                        "name": "create_file",
                        "arguments": {"path": "new.py", "content": "1"},
                    }
                ]
            )
        )
    responses.append(turn("Second attempt"))
    provider = Script(*responses)
    agent = CodingAgent(
        provider, workspace, HarnessConfig(checks=[check], limits=Limits(max_repairs=1))
    )
    result = asyncio.run(agent.run("Fix"))
    assert result.status == "blocked" and result.repairs == 1
    assert ("Repair budget" if change else "without workspace progress") in result.answer
    assert not provider.responses


def test_changes_after_checks_cannot_be_published_as_verified(workspace):
    check = Check(name="pass", argv=[sys.executable, "-c", "pass"])
    workspace.allowed_commands.add((str(workspace.root), tuple(check.argv)))

    def external_edit(event):
        if event.kind == "verification" and (not event.data["baseline"]):
            with (workspace.root / "sample.py").open("a") as stream:
                stream.write("# external edit\n")

    agent = CodingAgent(
        Script(turn("Done"), turn("Done again")),
        workspace,
        HarnessConfig(checks=[check], limits=Limits(max_repairs=1)),
        emit=external_edit,
    )
    result = asyncio.run(agent.run("Check"))
    assert result.status == "blocked"


def test_modified_verification_files_are_prominent(workspace):
    provider = Script(
        turn(
            calls=[
                {
                    "id": "test",
                    "name": "create_file",
                    "arguments": {"path": "test_new.py", "content": "pass"},
                }
            ]
        ),
        turn("Done"),
    )
    result = asyncio.run(CodingAgent(provider, workspace, HarnessConfig()).run("Add test"))
    assert "Verification-related files changed: test_new.py" in result.answer


def test_configured_check_script_changes_are_prominent(workspace):
    script = workspace.root / "check_behavior.py"
    script.write_text("raise SystemExit(1)\n")
    check = Check(name="behavior", argv=[sys.executable, "check_behavior.py"])
    workspace.allowed_commands.add((str(workspace.root), tuple(check.argv)))
    source = workspace.read_file("check_behavior.py")
    provider = Script(
        turn(
            calls=[
                {
                    "id": "weaken",
                    "name": "edit_file",
                    "arguments": {
                        "path": "check_behavior.py",
                        "old": source.text,
                        "new": "pass\n",
                        "expected_sha256": source.sha256,
                    },
                }
            ]
        ),
        turn("Done"),
    )
    agent = CodingAgent(provider, workspace, HarnessConfig(checks=[check]))
    result = asyncio.run(agent.run("Change check"))
    assert result.status == "verified"
    assert "Verification-related files changed: check_behavior.py" in result.answer

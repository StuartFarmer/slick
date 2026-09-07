import asyncio

import pytest

from examples.coding_harness.agent import CodingAgent
from examples.coding_harness.context import context_size, render_instructions
from examples.coding_harness.state import HarnessConfig
from slick.turns import ModelTurn, ToolCall, UserMessage


class SummaryBackend:
    def __init__(self, answer):
        self.answer = answer

    def identity(self):
        return {"provider": "demo", "model": "scripted"}

    async def aturn(self, history, *, tools, instructions=""):
        assert tools == []
        return ModelTurn("demo", "scripted", self.answer, [], [], "end_turn")


def test_compaction_swaps_complete_history_only_after_validation(workspace):
    backend = SummaryBackend(
        '{"facts":["API uses integers"],"decisions":[],"open_questions":[],'
        '"modified_files":[],"next_steps":[]}'
    )
    agent = CodingAgent(backend, workspace, HarnessConfig(skills=["python"]))
    agent.state.history = [UserMessage("Keep the API stable")]
    asyncio.run(agent.compact())
    assert len(agent.state.archived_histories) == 1
    assert "API uses integers" in agent.state.history[0].text
    assert "Python" in render_instructions(workspace, agent.config)


def test_bad_summary_preserves_history(workspace):
    agent = CodingAgent(SummaryBackend("not JSON"), workspace, HarnessConfig())
    agent.state.history = [UserMessage("Keep me")]
    before = list(agent.state.history)
    with pytest.raises(ValueError):
        asyncio.run(agent.compact())
    assert agent.state.history == before
    assert not agent.state.archived_histories


def test_pending_calls_cannot_compact(workspace):
    agent = CodingAgent(SummaryBackend("unused"), workspace, HarnessConfig())
    agent.state.history = [
        UserMessage("Read"),
        ModelTurn(
            "demo", "scripted", "", [ToolCall("a", "read_file", {"path": "a.py"})], [], "tool_calls"
        ),
    ]
    with pytest.raises(ValueError):
        asyncio.run(agent.compact())


def test_context_size_counts_instructions_and_opaque_items():
    history = [
        UserMessage("hi"),
        ModelTurn("demo", "scripted", "ok", [], [{"encrypted_content": "x" * 1000}], "end_turn"),
    ]
    assert context_size(history, instructions="rule", tools=[]) > 1000


def test_manual_compaction_blocks_a_concurrent_run(workspace):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        class WaitingSummary(SummaryBackend):
            async def aturn(self, *args, **kwargs):
                started.set()
                await release.wait()
                return await super().aturn(*args, **kwargs)

        backend = WaitingSummary(
            '{"facts":[],"decisions":[],"open_questions":[],"modified_files":[],"next_steps":[]}'
        )
        agent = CodingAgent(backend, workspace, HarnessConfig())
        agent.state.history = [UserMessage("old context")]
        task = asyncio.create_task(agent.compact())
        try:
            await asyncio.wait_for(started.wait(), 2)
            with pytest.raises(RuntimeError, match="active"):
                await asyncio.wait_for(agent.run("new work"), timeout=0.2)
        finally:
            release.set()
            await task
        assert not agent.state.running

    asyncio.run(scenario())


def test_summary_transport_failure_below_hard_limit_keeps_working(workspace):
    from examples.coding_harness.state import Limits
    from slick.backends import BackendError

    class SummaryFailure(SummaryBackend):
        async def aturn(self, history, *, tools, instructions=""):
            if not tools:
                raise BackendError("summary request unavailable")
            return ModelTurn("demo", "scripted", "Completed normally", [], [], "end_turn")

    agent = CodingAgent(
        SummaryFailure(""),
        workspace,
        HarnessConfig(limits=Limits(context_soft_chars=100, context_hard_chars=30000)),
    )
    result = asyncio.run(agent.run("Describe the code"))
    assert result.status == "unverified"
    assert result.answer == "Completed normally"
    assert not agent.state.archived_histories


def test_oversized_pinned_context_stops_before_any_request(workspace):
    from examples.coding_harness.state import Limits

    class NoRequests(SummaryBackend):
        async def aturn(self, *args, **kwargs):
            pytest.fail("oversized pinned input must not reach the backend")

    agent = CodingAgent(
        NoRequests(""),
        workspace,
        HarnessConfig(
            limits=Limits(context_soft_chars=100, context_hard_chars=200),
        ),
    )
    result = asyncio.run(agent.run("A task with pinned context"))
    assert result.status == "blocked"
    assert result.turns == 0


def test_unknown_skill_fails_before_execution(workspace):
    with pytest.raises(ValueError, match="Unknown skills"):
        CodingAgent(SummaryBackend(""), workspace, HarnessConfig(skills=["missing"]))


def test_bounded_summary_retains_complete_recent_exchange():
    from examples.coding_harness.context import summary_prompt
    from examples.coding_harness.state import Limits, SessionState
    from slick.turns import ToolResult

    state = SessionState(
        task="Keep recent",
        history=[
            UserMessage("old user " * 1000),
            ModelTurn(
                "demo", "scripted", "OLD_ASSISTANT", [ToolCall("old", "read", {})], [], "tool_calls"
            ),
            ToolResult("old", "OLD_RESULT"),
            UserMessage("RECENT_USER"),
            ModelTurn("demo", "scripted", "RECENT_ASSISTANT", [], [], "end_turn"),
        ],
    )
    text = summary_prompt(
        state, HarnessConfig(limits=Limits(context_soft_chars=4000, context_hard_chars=5000))
    )
    assert "RECENT_USER" in text and "RECENT_ASSISTANT" in text
    assert "OLD_ASSISTANT" not in text and "OLD_RESULT" not in text

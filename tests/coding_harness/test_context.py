import asyncio

import pytest

from examples.coding_harness.agent import CodingAgent
from examples.coding_harness.context import context_size, render_instructions
from examples.coding_harness.state import HarnessConfig
from slick import Session


class SummaryProvider:
    def __init__(self, answer):
        self.answer = answer

    def identity(self):
        return {"provider": "demo", "model": "scripted"}

    async def acall(self, context, *, tools=None, tool_results=""):
        assert tools == []
        return (self.answer, [])


def test_compaction_swaps_complete_history_only_after_validation(workspace):
    provider = SummaryProvider(
        '{"facts":["API uses integers"],"decisions":[],"open_questions":[],'
        '"modified_files":[],"next_steps":[]}'
    )
    agent = CodingAgent(provider, workspace, HarnessConfig(skills=["python"]))
    agent.state.context_notes = [{"after": 0, "role": "user", "text": "Keep the API stable"}]
    asyncio.run(agent.compact())
    assert len(agent.state.archived_histories) == 1
    assert "API uses integers" in agent.state.context_notes[0]["text"]
    assert "Python" in render_instructions(workspace, agent.config)


def test_bad_summary_preserves_history(workspace):
    agent = CodingAgent(SummaryProvider("not JSON"), workspace, HarnessConfig())
    agent.state.context_notes = [{"after": 0, "role": "user", "text": "Keep me"}]
    before = list(agent.state.context_notes)
    with pytest.raises(ValueError):
        asyncio.run(agent.compact())
    assert agent.state.context_notes == before
    assert not agent.state.archived_histories


def test_context_size_counts_supplied_context_and_results():
    results = [{"request": {"id": "a", "name": "read", "arguments": {}}, "content": "x" * 1000}]
    assert context_size("rule", tools=[], tool_results=results) > 1000


def test_manual_compaction_blocks_a_concurrent_run(workspace):
    async def scenario():
        started, release = (asyncio.Event(), asyncio.Event())

        class WaitingSummary(SummaryProvider):
            async def acall(self, *args, **kwargs):
                started.set()
                await release.wait()
                return await super().acall(*args, **kwargs)

        provider = WaitingSummary(
            '{"facts":[],"decisions":[],"open_questions":[],"modified_files":[],"next_steps":[]}'
        )
        agent = CodingAgent(provider, workspace, HarnessConfig())
        agent.state.context_notes = [{"after": 0, "role": "user", "text": "old context"}]
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
    from slick.providers import ProviderError

    class SummaryFailure(SummaryProvider):
        async def acall(self, context, *, tools=None, tool_results=""):
            if not tools:
                raise ProviderError("summary request unavailable")
            return ("Completed normally", [])

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

    class NoRequests(SummaryProvider):
        async def acall(self, *args, **kwargs):
            pytest.fail("oversized pinned input must not reach the provider")

    agent = CodingAgent(
        NoRequests(""),
        workspace,
        HarnessConfig(limits=Limits(context_soft_chars=100, context_hard_chars=200)),
    )
    result = asyncio.run(agent.run("A task with pinned context"))
    assert result.status == "blocked"
    assert result.turns == 0


def test_unknown_skill_fails_before_execution(workspace):
    with pytest.raises(ValueError, match="Unknown skills"):
        CodingAgent(SummaryProvider(""), workspace, HarnessConfig(skills=["missing"]))


def test_bounded_summary_retains_complete_recent_exchange():
    from examples.coding_harness.context import summary_prompt
    from examples.coding_harness.state import Limits, SessionState

    state = SessionState(
        task="Keep recent",
        context_notes=[
            {"after": 0, "role": "user", "text": "old user " * 1000},
            {
                "after": 0,
                "role": "assistant",
                "text": "OLD_ASSISTANT",
                "calls": [{"id": "old", "name": "read", "arguments": {}}],
            },
            {
                "after": 0,
                "role": "tool",
                "id": "old",
                "text": "OLD_RESULT",
                "error": False,
            },
            {"after": 0, "role": "user", "text": "RECENT_USER"},
            {"after": 0, "role": "assistant", "text": "RECENT_ASSISTANT", "calls": []},
        ],
    )
    text = summary_prompt(
        state,
        HarnessConfig(limits=Limits(context_soft_chars=4000, context_hard_chars=5000)),
        Session(),
    )
    assert "RECENT_USER" in text and "RECENT_ASSISTANT" in text
    assert "OLD_ASSISTANT" not in text and "OLD_RESULT" not in text

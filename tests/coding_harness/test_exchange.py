"""Harness state is independent of the provider's tool wire protocol."""

import asyncio
import json
from copy import deepcopy

import pytest

from examples.coding_harness.agent import CodingAgent
from examples.coding_harness.state import HarnessConfig


class Script:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []

    def identity(self):
        return {"provider": "demo", "model": "scripted"}

    async def acall(self, context, *, tools=None, tool_results=None):
        self.inputs.append((context, deepcopy(tool_results)))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_harness_executes_then_supplies_results_with_rendered_context(workspace):
    request = {"id": "a", "name": "read_file", "arguments": {"path": "sample.py"}}
    provider = Script([("Reading", [request]), ("Done", [])])
    agent = CodingAgent(provider, workspace, HarnessConfig())
    result = asyncio.run(agent.run("Inspect sample.py"))
    assert result.status == "unverified"
    context, results = provider.inputs[1]
    assert "Inspect sample.py" in context
    assert results[0]["request"] == request
    assert json.loads(results[0]["content"])["path"] == "sample.py"
    assert agent.session.ready_results == []
    assert len(agent.session.history) == 2


def test_session_history_does_not_recursively_replay_rendered_prompts(workspace):
    request = {"id": "a", "name": "read_file", "arguments": {"path": "sample.py"}}
    provider = Script([("FIRST_ANSWER", [request]), ("SECOND_ANSWER", [request]), ("Done", [])])
    agent = CodingAgent(provider, workspace, HarnessConfig())
    asyncio.run(agent.run("UNIQUE_TASK_MARKER"))
    assert len(agent.session.history) == 3
    for context, _ in provider.inputs:
        assert context.count("UNIQUE_TASK_MARKER") == 1
    assert provider.inputs[2][0].count("FIRST_ANSWER") == 1
    assert provider.inputs[2][0].count("SECOND_ANSWER") == 1


def test_failed_provider_call_keeps_pending_results(workspace):
    request = {"id": "a", "name": "read_file", "arguments": {"path": "sample.py"}}
    provider = Script([("", [request]), RuntimeError("offline")])
    agent = CodingAgent(provider, workspace, HarnessConfig())
    asyncio.run(agent.session.acall("Read"))
    results = asyncio.run(agent.session.resolve_pending())
    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(agent.step())
    assert agent.session.ready_results == results


def test_compaction_preserves_pending_results(workspace):
    summary = {
        "facts": ["Keep this"],
        "decisions": [],
        "open_questions": [],
        "modified_files": [],
        "next_steps": [],
    }
    request = {"id": "a", "name": "read_file", "arguments": {"path": "sample.py"}}
    provider = Script([("", [request]), (json.dumps(summary), [])])
    agent = CodingAgent(provider, workspace, HarnessConfig())
    agent._note("Old context")
    asyncio.run(agent.session.acall("Read"))
    results = asyncio.run(agent.session.resolve_pending())
    before = agent.session.history
    asyncio.run(agent.compact())
    assert agent.session.ready_results == results
    assert agent.session.history == before
    assert "Keep this" in agent.state.context_notes[0]["text"]


def test_repeated_compaction_can_still_observe_unsubmitted_tool_results(workspace):
    summary = json.dumps(
        {
            "facts": ["A brief checkpoint"],
            "decisions": [],
            "open_questions": [],
            "modified_files": [],
            "next_steps": [],
        }
    )
    request = {"id": "a", "name": "read_file", "arguments": {"path": "sample.py"}}
    provider = Script([("", [request]), (summary, []), (summary, [])])
    agent = CodingAgent(provider, workspace, HarnessConfig())
    agent._note("Read")
    asyncio.run(agent.session.acall("Read"))
    results = asyncio.run(agent.session.resolve_pending())
    asyncio.run(agent.compact())
    asyncio.run(agent.compact())
    assert "value = 1" in provider.inputs[2][0]
    assert agent.session.ready_results == results

"""Automatic tool conversations parse only final answers and preserve pending work."""

import asyncio
from copy import deepcopy

import pytest
from pydantic import BaseModel, ValidationError

from slick import Session


class Answer(BaseModel):
    value: int


class Script:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.inputs = []

    def call(self, context, **kwargs):
        self.inputs.append((context, deepcopy(kwargs)))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response

    async def acall(self, context, **kwargs):
        return self.call(context, **kwargs)


def request(id="one", value=3):
    return {"id": id, "name": "double", "arguments": {"value": value}}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_run_executes_multiple_turns_before_parsing(asynchronous):
    calls = []

    def double(value: int) -> int:
        """Double a number."""
        calls.append(value)
        return value * 2

    provider = Script(
        ("I'll calculate first.", [request()]),
        ("One more calculation.", [request("two", 6)]),
        ('{"value": 12}', []),
    )
    session = Session(provider=provider, tools=[double])
    result = (
        asyncio.run(session.arun("Calculate", output_type=Answer, max_turns=3))
        if asynchronous
        else session.run("Calculate", output_type=Answer, max_turns=3)
    )
    assert result == Answer(value=12)
    assert calls == [3, 6]
    assert provider.inputs[1][1]["tool_results"][0]["content"] == "6"
    assert provider.inputs[2][1]["tool_results"][0]["content"] == "12"
    assert "I'll calculate first." in provider.inputs[2][0]
    assert len(session.history) == 3
    assert session.pending_requests == session.ready_results == []


def test_limit_can_resume_from_a_snapshot_without_reexecuting_tools():
    calls = []

    def double(value: int) -> int:
        """Double a number."""
        calls.append(value)
        return value * 2

    async def scenario():
        session = Session(provider=Script(("working", [request()])), tools=[double])
        with pytest.raises(RuntimeError, match="turn"):
            await session.arun("Calculate", max_turns=1)
        assert calls == []
        provider = Script(("done", []))
        resumed = Session.from_dict(session.to_dict(), provider=provider, tools=[double])
        assert await resumed.arun() == "done"
        assert calls == [3]
        assert provider.inputs[0][1]["tool_results"][0]["content"] == "6"

    asyncio.run(scenario())


def test_bad_final_json_is_kept_and_body_is_not_retried():
    async def scenario():
        provider = Script(("bad JSON", []))
        session = Session(provider=provider)
        with pytest.raises(ValidationError):
            await session.arun("answer", output_type=Answer)
        assert session.history[-1]["text"] == "bad JSON"
        assert await session.arun() == "bad JSON"
        assert len(provider.inputs) == 1

    asyncio.run(scenario())


def test_async_tools_and_errors_are_returned_to_the_model():
    async def double(value: int) -> int:
        """Double a number."""
        raise ValueError("missing input")

    provider = Script(("working", [request()]), ("I could not complete it.", []))
    session = Session(provider=provider, tools=[double])
    assert asyncio.run(session.arun("calculate")) == "I could not complete it."
    assert provider.inputs[1][1]["tool_results"][0]["is_error"] is True


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_turn_budget_fails_before_execution(limit):
    session = Session(provider=Script())
    with pytest.raises(ValueError, match="positive integer"):
        session.run("answer", max_turns=limit)
    assert session.history == []


def test_active_run_rejects_competing_operations():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class WaitingProvider:
            async def acall(self, context, **kwargs):
                entered.set()
                await release.wait()
                return "done", []

        session = Session(provider=WaitingProvider())
        running = asyncio.create_task(session.arun("first"))
        await entered.wait()
        try:
            with pytest.raises(RuntimeError, match="already active"):
                await session.arun("second")
            with pytest.raises(RuntimeError, match="already active"):
                await session.acall("second")
        finally:
            release.set()
            assert await running == "done"

    asyncio.run(scenario())

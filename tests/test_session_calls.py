"""Automatic interaction bookkeeping without implicit context replay."""

import asyncio
from copy import deepcopy

import pytest

from slick import Session


class Script:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.inputs = []

    async def acall(self, context, *, tools=None, tool_results=None):
        self.inputs.append((context, deepcopy(tool_results)))
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def search(query: str) -> str:
    """Search documents."""
    return query


def test_context_and_provider_override_are_explicit():
    async def scenario():
        first = Script(("one", []), ("three", []))
        second = Script(("two", []))
        session = Session(provider=first)
        assert await session.acall("FIRST") == ("one", [])
        assert await session.acall("REPLACEMENT", provider=second) == ("two", [])
        await session.acall("THIRD")
        assert second.inputs == [("REPLACEMENT", [])]
        assert [item["context"] for item in session.history] == ["FIRST", "REPLACEMENT", "THIRD"]

    asyncio.run(scenario())


@pytest.mark.parametrize("context", [None, 4, [], ""])
def test_invalid_context_does_not_call_provider(context):
    provider = Script()
    session = Session(provider=provider)
    with pytest.raises(ValueError):
        asyncio.run(session.acall(context))
    assert provider.inputs == [] and session.history == []


def test_missing_provider_and_invalid_definitions_fail_early():
    with pytest.raises(ValueError, match="provider"):
        asyncio.run(Session().acall("hello"))
    with pytest.raises(ValueError):
        Session(tools=[lambda x: x])


@pytest.mark.parametrize(
    "answer",
    [
        "text",
        (3, []),
        ("", [{}]),
        (
            "",
            [
                {"id": "same", "name": "search", "arguments": {}},
                {"id": "same", "name": "search", "arguments": {}},
            ],
        ),
    ],
)
def test_invalid_response_does_not_record_exchange(answer):
    session = Session(provider=Script(answer), tools=[search])
    with pytest.raises(ValueError):
        asyncio.run(session.acall("hello"))
    assert session.history == []


def test_unsolicited_tools_are_rejected():
    session = Session(
        provider=Script(("", [{"id": "a", "name": "search", "arguments": {"query": "x"}}]))
    )
    with pytest.raises(ValueError, match="tools"):
        asyncio.run(session.acall("hello"))
    assert session.history == []


def test_data_views_do_not_mutate_history():
    request = {"id": "a", "name": "search", "arguments": {"query": "x"}}
    session = Session(provider=Script(("", [request])), tools=[search])
    _, returned = asyncio.run(session.acall("hello"))
    returned[0]["arguments"]["query"] = "changed"
    request["name"] = "also changed"
    session.history.clear()
    session.pending_requests[0]["arguments"].clear()
    session.tools.clear()
    assert session.pending_requests[0]["arguments"] == {"query": "x"}
    assert session.pending_requests[0]["name"] == "search"
    assert len(session.tools) == 1


def test_overlapping_calls_rejected_but_independent_sessions_work():
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        class Waiting:
            async def acall(self, context, **kwargs):
                started.set()
                await release.wait()
                return context, []

        provider = Waiting()
        session = Session(provider=provider)
        task = asyncio.create_task(session.acall("first"))
        await started.wait()
        with pytest.raises(RuntimeError, match="active"):
            await session.acall("overlap")
        other = asyncio.create_task(Session(provider=provider).acall("independent"))
        release.set()
        assert await task == ("first", [])
        assert await other == ("independent", [])
        assert len(session.history) == 1

    asyncio.run(scenario())

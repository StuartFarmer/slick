"""Execute real tools once and retain their results across failed model calls."""

import asyncio
import json
from copy import deepcopy

import pytest
from pydantic import BaseModel

from slick import Session
from tests.test_session_calls import Script, search


def request(name="search", **arguments):
    return {"id": "a", "name": name, "arguments": arguments}


def test_batch_resolution_and_repeated_individual_resolution_execute_once():
    async def scenario():
        effects = []

        async def lookup(query: str) -> str:
            """Find a document."""
            effects.append(query)
            return "pricing.md"

        call = request("lookup", query="pricing")
        provider = Script(("Searching", [call]), ("Found", []))
        session = Session(provider=provider, tools=[lookup])
        await session.acall("Find pricing")
        results = await session.resolve_pending()
        assert results == [{"request": call, "content": "pricing.md", "is_error": False}]
        assert await session.resolve_pending() == []
        assert await session.resolve(call) == results[0]
        await session.acall("")
        assert provider.inputs[1] == ("", results)
        assert effects == ["pricing"]
        assert session.ready_results == []
        assert session.history[0]["work"][0]["submitted"] is True

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [RuntimeError("offline"), (42, []), asyncio.CancelledError()])
def test_failed_call_preserves_results_and_allows_provider_switch(failure):
    async def scenario():
        provider = Script(("", [request(query="pricing")]), failure)
        session = Session(provider=provider, tools=[search])
        await session.acall("Find")
        results = await session.resolve_pending()
        before = session.history
        with pytest.raises((RuntimeError, ValueError, asyncio.CancelledError)):
            await session.acall("Continue")
        assert session.history == before
        assert session.ready_results == results
        second = Script(("Recovered", []))
        await session.acall("New context", provider=second)
        assert second.inputs == [("New context", results)]
        assert session.ready_results == []

    asyncio.run(scenario())


def test_mutating_provider_and_returned_results_cannot_corrupt_record():
    async def scenario():
        session = Session(provider=Script(("", [request(query="x")])), tools=[search])
        await session.acall("Find")
        returned = await session.resolve_pending()
        expected = deepcopy(returned)
        returned[0]["request"]["arguments"].clear()
        session.ready_results[0]["content"] = "corrupted"

        class Mutating:
            async def acall(self, context, *, tools, tool_results):
                tool_results[0]["content"] = "corrupted"
                raise RuntimeError("offline")

        with pytest.raises(RuntimeError):
            await session.acall("Continue", provider=Mutating())
        assert session.ready_results == expected

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "call",
    [
        request("missing"),
        request(),
        {"id": "a", "name": "search", "arguments": "{", "argument_error": "Invalid JSON"},
    ],
)
def test_invalid_tools_become_error_results_without_execution(call):
    session = Session(provider=Script(("", [call])), tools=[search])
    asyncio.run(session.acall("Find"))
    result = asyncio.run(session.resolve(call))
    assert result["is_error"] and result["content"]
    assert session.pending_requests == []


@pytest.mark.parametrize("bad_return", [False, True])
def test_effect_then_execution_or_result_failure_is_not_retried(bad_return):
    effects = []

    def write() -> int:
        """Write once."""
        effects.append("effect")
        if not bad_return:
            raise RuntimeError("failed after write")
        return "wrong type"

    call = request("write")
    session = Session(provider=Script(("", [call])), tools=[write])
    asyncio.run(session.acall("Write"))
    result = asyncio.run(session.resolve(call))
    assert result["is_error"] and "effects may have occurred" in result["content"]
    assert asyncio.run(session.resolve(call)) == result
    assert effects == ["effect"]


class Document(BaseModel):
    name: str


def test_bound_method_pydantic_return_and_partial_batch_order():
    class Store:
        def __init__(self):
            self.queries = []

        def find(self, query: str) -> Document:
            """Find a document."""
            self.queries.append(query)
            return Document(name=query)

    store = Store()
    calls = [request("find", query="first"), {**request("find", query="second"), "id": "b"}]
    session = Session(provider=Script(("", calls)), tools=[store.find])
    asyncio.run(session.acall("Find"))
    asyncio.run(session.resolve(calls[1]))
    assert session.pending_requests == [calls[0]]
    results = asyncio.run(session.resolve_pending())
    assert len(results) == 1 and json.loads(results[0]["content"]) == {"name": "first"}
    assert store.queries == ["second", "first"]
    assert [result["request"] for result in session.ready_results] == calls


def test_reused_ids_are_scoped_to_current_exchange():
    first, second = request(query="first"), request(query="second")
    session = Session(provider=Script(("", [first]), ("", [second])), tools=[search])
    asyncio.run(session.acall("First"))
    with pytest.raises(ValueError, match="pending"):
        asyncio.run(session.acall("Too soon"))
    with pytest.raises(ValueError):
        asyncio.run(session.resolve(second))
    asyncio.run(session.resolve(first))
    asyncio.run(session.acall("Second"))
    with pytest.raises(ValueError):
        asyncio.run(session.resolve(first))
    assert asyncio.run(session.resolve(second))["content"] == "second"
    assert session.history[0]["work"][0]["result"]["content"] == "first"


def test_cancelled_batch_records_interruption_and_leaves_later_work_unstarted():
    async def scenario():
        started = asyncio.Event()
        effects = []

        async def wait_tool() -> str:
            """Write then wait."""
            effects.append("write")
            started.set()
            await asyncio.Event().wait()
            return "unreachable"

        calls = [request("wait_tool"), {**request(query="later"), "id": "b"}]
        session = Session(provider=Script(("", calls)), tools=[wait_tool, search])
        await session.acall("Go")
        task = asyncio.create_task(session.resolve_pending())
        await started.wait()
        for operation in (session.resolve(calls[0]), session.resolve_pending(), session.acall("x")):
            with pytest.raises(RuntimeError, match="active"):
                await operation
        with pytest.raises(RuntimeError, match="active"):
            session.cancel_pending("stop")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.pending_requests == [calls[1]]
        assert "effects may have occurred" in session.ready_results[0]["content"]
        assert (await session.resolve(calls[0]))["is_error"]
        assert effects == ["write"]
        cancelled = session.cancel_pending("run stopped")
        assert len(cancelled) == 1 and "Not started" in cancelled[0]["content"]
        assert await session.resolve_pending() == []
        assert len(session.ready_results) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", [None, "", "  ", 1])
def test_invalid_cancel_reason_does_not_change_work(reason):
    session = Session(provider=Script(("", [request(query="x")])), tools=[search])
    asyncio.run(session.acall("Go"))
    before = session.history
    with pytest.raises(ValueError):
        session.cancel_pending(reason)
    assert session.history == before


def test_ordinary_tool_error_does_not_stop_batch():
    calls = [request("missing"), {**request(query="second"), "id": "b"}]
    session = Session(provider=Script(("", calls)), tools=[search])
    asyncio.run(session.acall("Go"))
    results = asyncio.run(session.resolve_pending())
    assert results[0]["is_error"]
    assert results[1]["content"] == "second" and not results[1]["is_error"]


def test_incompatible_provider_payload_preserves_ready_work():
    from slick.providers import AnthropicAPI, ProviderError

    call = {"id": "a", "name": "search", "arguments": "{", "argument_error": "Invalid JSON"}
    session = Session(provider=Script(("", [call])), tools=[search])
    asyncio.run(session.acall("Go"))
    results = asyncio.run(session.resolve_pending())
    before = session.history
    # Conversion fails before the unused client can perform any I/O.
    provider = AnthropicAPI("offline", async_client=object())
    with pytest.raises(ProviderError) as raised:
        asyncio.run(session.acall("Continue", provider=provider))
    assert "malformed JSON" in str(raised.value.__cause__)
    assert session.ready_results == results
    assert session.history == before

"""Snapshots are inert data, with relationships checked before restoration."""

import asyncio
import json
from copy import deepcopy

import pytest

from slick import Session
from tests.test_session_calls import Script
from tests.test_session_tools import request


@pytest.mark.parametrize("completed", [0, 1, 2])
def test_partial_work_round_trip_does_not_repeat_completed_effects(completed):
    async def scenario():
        effects = []

        def write(query: str) -> str:
            """Record a value."""
            effects.append(query)
            return query

        calls = [request("write", query="one"), {**request("write", query="two"), "id": "b"}]
        session = Session(provider=Script(("", calls)), tools=[write])
        await session.acall("Write")
        for call in calls[:completed]:
            await session.resolve(call)
        data = json.loads(json.dumps(session.to_dict(), allow_nan=False))
        provider = Script(("Done", []))
        restored = Session.from_dict(data, provider=provider, tools=[write])
        assert effects == ["one", "two"][:completed]
        assert restored.to_dict() == data
        data["history"].clear()
        assert len(restored.history) == 1
        await restored.resolve_pending()
        await restored.acall("Continue")
        assert effects == ["one", "two"]
        assert len(provider.inputs[0][1]) == 2
        assert restored.ready_results == []
        assert Session.from_dict(restored.to_dict()).to_dict() == restored.to_dict()

    asyncio.run(scenario())


def snapshot():
    call = request(query="x")
    return {
        "version": 1,
        "history": [
            {
                "context": "Go",
                "text": "",
                "work": [{"request": call, "result": None, "submitted": False}],
            }
        ],
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(version=True),
        lambda d: d.update(version=2),
        lambda d: d.update(extra="unknown"),
        lambda d: d.update(history={}),
        lambda d: d["history"][0].update(extra=1),
        lambda d: d["history"][0].update(context=3),
        lambda d: d["history"][0].update(text=None),
        lambda d: d["history"][0].update(work={}),
        lambda d: d["history"][0]["work"][0].update(submitted=True),
        lambda d: d["history"][0]["work"][0].update(submitted=1),
        lambda d: d["history"][0]["work"][0].update(extra=1),
        lambda d: d["history"][0]["work"].append(deepcopy(d["history"][0]["work"][0])),
        lambda d: d["history"].append({"context": "next", "text": "Done", "work": []}),
        lambda d: d["history"][0]["work"][0].update(
            result={"request": request(query="different"), "content": "x", "is_error": False}
        ),
        lambda d: d["history"][0]["work"][0]["request"]["arguments"].update(query=float("nan")),
    ],
)
def test_invalid_snapshots_reject_without_mutating_input(change):
    data = snapshot()
    change(data)
    before = deepcopy(data)
    with pytest.raises(ValueError):
        Session.from_dict(data)
    assert repr(data) == repr(before)


def test_empty_legacy_and_submitted_snapshots_round_trip():
    assert Session.from_dict(Session().to_dict()).history == []
    data = snapshot()
    work = data["history"][0]["work"][0]
    work["result"] = {"request": deepcopy(work["request"]), "content": "x", "is_error": False}
    data["history"][0]["context"] = None
    assert Session.from_dict(data).ready_results == [work["result"]]
    work["submitted"] = True
    data["history"].append(
        {
            "context": "next",
            "text": "",
            "work": [{"request": request(query="new"), "result": None, "submitted": False}],
        }
    )
    assert Session.from_dict(data).pending_requests == [request(query="new")]


def test_snapshot_rejected_during_execution_and_resources_are_not_saved():
    async def scenario():
        started = asyncio.Event()

        async def wait() -> str:
            """Wait."""
            started.set()
            await asyncio.Event().wait()
            return ""

        provider = Script(("", [request("wait")]))
        provider.api_key = "secret-not-in-snapshot"
        session = Session(provider=provider, tools=[wait])
        await session.acall("Go")
        task = asyncio.create_task(session.resolve_pending())
        await started.wait()
        with pytest.raises(RuntimeError, match="active"):
            session.to_dict()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        data = session.to_dict()
        assert "secret-not-in-snapshot" not in json.dumps(data)
        restored = Session.from_dict(data)
        assert restored.tools == []
        assert restored.ready_results[0]["is_error"]

    asyncio.run(scenario())

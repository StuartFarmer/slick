import asyncio

import pytest
from test_session_run import Answer, Script, request

from slick import Session, Workflow, prompt, workflow


@prompt(output_type=Answer)
async def calculate(*, generated: Answer) -> Answer:
    """Calculate with the available tools. {{ output_format }}"""
    return generated


@workflow
async def application(session):
    return await calculate(session=session)


def test_recovery_restores_tool_progress_and_completed_session(tmp_path):
    calls = []
    fail = True

    async def double(value: int) -> int:
        """Double a number."""
        calls.append(value)
        if value == 4 and fail:
            raise asyncio.CancelledError()
        return value * 2

    async def scenario():
        nonlocal fail
        path = tmp_path / "workflow.db"
        first = Session(
            provider=Script(("working", [request(), request("two", 4)])), tools=[double]
        )
        with pytest.raises(asyncio.CancelledError):
            async with Workflow(path, run_id="one"):
                await application(first)
        assert calls == [3, 4]
        fail = False
        provider = Script(('{"value": 14}', []))
        resumed = Session(provider=provider, tools=[double])
        async with Workflow(path, run_id="one"):
            assert await application(resumed) == Answer(value=14)
        assert calls == [3, 4, 4]
        assert len(provider.inputs) == 1
        assert [item["content"] for item in provider.inputs[0][1]["tool_results"]] == ["6", "8"]
        assert len(resumed.history) == 2

        finished = Session(provider=Script(), tools=[double])
        async with Workflow(path, run_id="one"):
            assert await application(finished) == Answer(value=14)
        assert finished.to_dict() == resumed.to_dict()
        assert calls == [3, 4, 4]

    asyncio.run(scenario())

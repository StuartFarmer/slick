"""Execute real Python loops twice and verify only unfinished work repeats."""

import asyncio
import inspect
from functools import wraps

import pytest
from pydantic import BaseModel

from slick import Inbox, Workflow, WorkflowError, workflow


class Item(BaseModel):
    value: int


class Decision(BaseModel):
    accepted: bool


@workflow
async def loop(worker, inbox):
    items = []
    for number in range(3):
        item = await worker(number)
        items.append(item.value)
    decision = await inbox.request(items, output_type=Decision)
    if decision.accepted:
        return items
    return []


def test_loop_replays_typed_results_and_reconnects_to_request(tmp_path):
    async def scenario():
        path = tmp_path / "workflow.db"
        inbox = Inbox(path, poll_interval=0.001)
        calls = []
        fail = True

        async def worker(number) -> Item:
            calls.append(number)
            if number == 2 and fail:
                raise RuntimeError("interrupted")
            return Item(value=number)

        async def execute():
            async with Workflow(path, run_id="one", inputs={"paper": "original"}):
                return await loop(worker, inbox)

        with pytest.raises(RuntimeError, match="interrupted"):
            await execute()
        fail = False
        waiting = asyncio.create_task(execute())
        while not inbox.pending():
            await asyncio.sleep(0.001)
        channel, message = inbox.pending()[0]
        assert message == [0, 1, 2]
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert calls == [0, 1, 2, 2]
        inbox.send(channel, {"accepted": True})
        assert await execute() == [0, 1, 2]
        assert await execute() == [0, 1, 2]
        assert calls == [0, 1, 2, 2]
        assert inbox.pending() == []

    asyncio.run(asyncio.wait_for(scenario(), 3))


def test_closures_defaults_and_unmanaged_execution(tmp_path):
    defaults = []

    def default():
        defaults.append(7)
        return 7

    async def operation(value) -> int:
        return value

    @workflow
    async def closed(value=default()):  # noqa: B008 - ensure the compiler preserves evaluated defaults.
        result = await operation(value)
        return result

    assert inspect.signature(closed).parameters["value"].default == 7
    assert defaults == [7]  # Compilation must not evaluate defaults again.
    assert asyncio.run(closed()) == 7
    assert asyncio.run(closed(9)) == 9

    async def scenario():
        async with Workflow(tmp_path / "workflow.db", run_id="closure"):
            assert await closed() == 7

    asyncio.run(scenario())
    assert defaults == [7]


def test_run_identity_and_single_owner(tmp_path):
    async def scenario():
        path = tmp_path / "workflow.db"
        async with Workflow(path, run_id="one", inputs={"paper": "a"}):
            with pytest.raises(WorkflowError):
                async with Workflow(path, run_id="one"):
                    pass
        for options in ({"inputs": {"paper": "b"}}, {"version": "2", "inputs": {"paper": "a"}}):
            with pytest.raises(WorkflowError, match="different"):
                async with Workflow(path, run_id="one", **options):
                    pass

    asyncio.run(scenario())


def test_unsupported_syntax_fails_at_decoration():
    async def unsupported():
        try:
            return await asyncio.sleep(0)
        finally:
            pass

    with pytest.raises(WorkflowError, match="Try"):
        workflow(unsupported)


def test_concurrent_instrumented_calls_are_rejected(tmp_path):
    @workflow
    async def child():
        return await asyncio.sleep(0)

    async def scenario():
        async with Workflow(tmp_path / "workflow.db", run_id="one"):
            with pytest.raises(WorkflowError, match="concurrent"):
                await asyncio.create_task(child())

    asyncio.run(scenario())


def test_changed_workflow_code_is_rejected(tmp_path):
    async def operation(value) -> int:
        return value

    async def original():
        return await operation(1)

    async def changed():
        return await operation(2)

    changed.__qualname__ = original.__qualname__
    first, second = workflow(original), workflow(changed)

    async def scenario():
        path = tmp_path / "workflow.db"
        async with Workflow(path, run_id="one"):
            assert await first() == 1
        with pytest.raises(WorkflowError, match="code changed"):
            async with Workflow(path, run_id="one"):
                await second()

    asyncio.run(scenario())


def test_await_arguments_keep_their_python_scope(tmp_path):
    async def operation(value) -> int:
        return value

    @workflow
    async def example(number):
        return await operation(locals()["number"])

    async def scenario():
        async with Workflow(tmp_path / "workflow.db", run_id="one"):
            assert await example(42) == 42

    asyncio.run(scenario())


def test_existing_decorators_are_not_silently_bypassed():
    async def original():
        return 1

    @wraps(original)
    async def wrapped():
        return 2

    with pytest.raises(WorkflowError, match="directly"):
        workflow(wrapped)

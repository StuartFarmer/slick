import asyncio

import pytest
from pydantic import validate_call
from test_session_run import Answer, Script, request

from slick import Session, prompt


@pytest.mark.parametrize("asynchronous", [False, True])
def test_session_generation_finishes_before_postprocessing(asynchronous):
    events = []

    def double(value: int) -> int:
        """Double a number."""
        events.append("tool")
        return value * 2

    def declaration(question: str, *, generated: Answer) -> str:
        """{{ question }} {{ output_format }}"""
        events.append("body")
        return str(generated.value)

    async def async_declaration(question: str, *, generated: Answer) -> str:
        """{{ question }} {{ output_format }}"""
        return declaration(question, generated=generated)

    decorated = validate_call(
        prompt(output_type=Answer, max_turns=2)(async_declaration if asynchronous else declaration)
    )
    session = Session(
        provider=Script(("working", [request()]), ('{"value": 6}', [])), tools=[double]
    )
    result = decorated("Calculate", session=session)
    assert (asyncio.run(result) if asynchronous else result) == "6"
    assert events == ["tool", "body"]
    for kwargs in ({}, {"provider": Script(), "session": session}):
        with pytest.raises(TypeError, match="exactly one"):
            result = decorated("Calculate", **kwargs)
            if asynchronous:
                asyncio.run(result)


def test_session_turn_limit_prevents_postprocessing():
    @prompt(max_turns=1)
    async def operation(*, generated):
        """Calculate."""
        raise AssertionError("No final answer yet")

    def double(value: int) -> int:
        """Double a number."""
        raise AssertionError("Budget exhausted before tool execution")

    session = Session(provider=Script(("working", [request()])), tools=[double])
    with pytest.raises(RuntimeError, match="turn"):
        asyncio.run(operation(session=session))

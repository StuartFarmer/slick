"""The decorator generates first, then runs ordinary Python postprocessing."""

import asyncio
import inspect

import pytest
from pydantic import BaseModel, ValidationError

from slick import prompt


class Answer(BaseModel):
    value: int


def invoke(fn, asynchronous, *args, **kwargs):
    return asyncio.run(fn(*args, **kwargs)) if asynchronous else fn(*args, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_generation_precedes_body_and_output_type_is_separate_from_return_type(asynchronous):
    events = []

    class Provider:
        def call(self, context):
            events.append("generation")
            assert "input" in context and '"integer"' in context
            return '{"value": 4}', [{"id": "unused", "name": "unused", "arguments": {}}]

        async def acall(self, context):
            return self.call(context)

    def declaration(text, *, generated: Answer) -> str:
        """{{ text }} {{ output_format }}"""
        events.append("body")
        return f"{text}: {generated.value * 2}"

    async def async_declaration(text, *, generated: Answer) -> str:
        """{{ text }} {{ output_format }}"""
        await asyncio.sleep(0)
        return declaration(text, generated=generated)

    fn = prompt(output_type=Answer)(async_declaration if asynchronous else declaration)
    invoke(fn.render, asynchronous, "input")
    assert events == []
    assert invoke(fn, asynchronous, "input", provider=Provider()) == "input: 8"
    assert events == ["generation", "body"]
    assert "generated" not in inspect.signature(fn).parameters
    assert inspect.signature(fn).parameters["provider"].default is inspect.Parameter.empty


def test_instance_binding_provider_switching_and_reserved_arguments():
    class Provider:
        def __init__(self, response):
            self.response = response

        def call(self, context):
            assert context == "Ada: hello"
            return self.response, []

    class Greeter:
        def __init__(self, name):
            self.name = name

        @prompt
        def greet(self, message="hello", *, generated):
            """{{ instance.name }}: {{ message }}"""
            return generated.upper()

    greeter = Greeter("Ada")
    assert greeter.greet(provider=Provider("first")) == "FIRST"
    assert greeter.greet(provider=Provider("second")) == "SECOND"
    with pytest.raises(TypeError, match="provider"):
        greeter.greet()
    with pytest.raises(TypeError, match="generated"):
        greeter.greet(provider=Provider("unused"), generated="override")


@pytest.mark.parametrize("result", [None, Ellipsis, "", False, 0])
def test_body_passthrough_and_explicit_falsy_results(result):
    class Provider:
        def call(self, context):
            return " raw text ", []

    @prompt
    def transform() -> Answer:
        """No output_type means raw text, regardless of the return annotation."""
        return result

    expected = " raw text " if result is None or result is Ellipsis else result
    assert transform(provider=Provider()) == expected


@pytest.mark.parametrize("asynchronous", [False, True])
def test_failed_generation_or_parsing_never_runs_the_body(asynchronous):
    class Provider:
        def call(self, context):
            return "invalid JSON", []

        async def acall(self, context):
            raise RuntimeError("offline")

    def declaration(*, generated):
        """Generate an answer."""
        raise AssertionError("Body must not run")

    async def async_declaration(*, generated):
        """Generate an answer."""
        raise AssertionError("Body must not run")

    fn = prompt(output_type=Answer)(async_declaration if asynchronous else declaration)
    with pytest.raises(RuntimeError if asynchronous else ValidationError):
        invoke(fn, asynchronous, provider=Provider())

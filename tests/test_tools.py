"""Callable schemas, invocation, serialization, and cancellation."""

import asyncio
import importlib
import inspect
import json
import threading

import pytest
from pydantic import BaseModel, field_serializer


@pytest.fixture(scope="session")
def api():
    return importlib.import_module("slick.tools")


class Document(BaseModel):
    id: str
    text: str


class BrokenSerialization(BaseModel):
    text: str

    @field_serializer("text")
    def fail(self, value):
        raise RuntimeError("serializer failed")


class Connection:
    pass


def returns(value, annotation=inspect.Signature.empty, calls=None):
    def result():
        """Return a fixture value."""
        if calls is not None:
            calls.append("called")
        return value

    if annotation is not inspect.Signature.empty:
        result.__annotations__["return"] = annotation
    return result


def assert_error(api, caught, phase, name=None):
    assert isinstance(caught.value, api.ToolError)
    assert isinstance(caught.value, ValueError)
    assert caught.value.phase == phase
    if name is not None:
        assert caught.value.name == name
        assert name in str(caught.value)


def test_public_exports(api):
    import slick

    assert slick.Tool is api.Tool
    assert slick.ToolError is api.ToolError
    assert slick.tool is api.tool


@pytest.mark.parametrize("parentheses", [False, True])
def test_tool_decorator_discovers_metadata_and_prepares_without_execution(api, parentheses):
    calls = []
    decorate = api.tool() if parentheses else api.tool

    @decorate
    def lookup(key: str) -> str:
        """Read a document.

        Return its contents.
        """
        calls.append(key)
        return key

    assert isinstance(lookup, api.Tool)
    assert lookup.name == "lookup"
    assert lookup.description == "Read a document.\n\nReturn its contents."
    assert api.prepare_tools([lookup])["lookup"] is lookup
    assert calls == []
    assert lookup.invoke({"key": "intro"}) == "intro"
    assert calls == ["intro"]


@pytest.mark.parametrize(
    "overrides, expected_name, expected_description",
    [
        ({"name": "read"}, "read", "Read a document."),
        ({"description": "Fetch content."}, "lookup", "Fetch content."),
        ({"name": None, "description": None}, "lookup", "Read a document."),
        ({"name": "read", "description": "Fetch content."}, "read", "Fetch content."),
    ],
)
def test_tool_decorator_overrides_metadata_independently(
    api, overrides, expected_name, expected_description
):
    @api.tool(**overrides)
    async def lookup(key: str) -> str:
        """Read a document."""
        return key

    assert lookup.name == expected_name
    assert lookup.description == expected_description
    assert asyncio.run(lookup.ainvoke({"key": "intro"})) == "intro"


def test_direct_tool_requires_explicit_metadata_and_does_not_read_docstrings(api, monkeypatch):
    def lookup(key: str) -> str:
        return key

    def unexpected_doc_lookup(*args, **kwargs):
        raise AssertionError("Tool must receive its description explicitly")

    monkeypatch.setattr(inspect, "getdoc", unexpected_doc_lookup)
    explicit = api.Tool(lookup, name="read", description="Read a document.")
    assert explicit.name == "read"
    assert explicit.description == "Read a document."
    assert explicit.invoke({"key": "intro"}) == "intro"
    decorated = api.tool(lookup, name="read", description="Read a document.")
    assert decorated.invoke({"key": "intro"}) == "intro"
    for metadata in ({}, {"name": "read"}, {"description": "Read a document."}):
        with pytest.raises(TypeError):
            api.Tool(lookup, **metadata)


def test_metadata_and_original_callable_are_preserved(api):
    def search(query: str, *, limit: int = 5) -> list[str]:
        """Find documents.

        Returns identifiers in relevance order.
        """
        return [query] * limit

    original_signature = inspect.signature(search)
    tool = api.tool(search)
    assert tool.name == "search"
    assert tool.description == inspect.getdoc(search)
    schema = tool.parameters
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"query"}
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["limit"]["type"] == "integer"
    assert inspect.signature(search) == original_signature
    assert search("direct", limit=1) == ["direct"]


def test_overrides_and_schema_copies(api):
    def lookup(key: str) -> str:
        return key

    tool = api.Tool(lookup, name="read_document", description="  Read a document.  ")
    assert tool.name == "read_document"
    assert tool.description == "Read a document."
    schema = tool.parameters
    schema["properties"]["key"]["type"] = "integer"
    assert tool.parameters["properties"]["key"]["type"] == "string"
    assert tool.invoke({"key": "intro"}) == "intro"


@pytest.mark.parametrize("text", ["", "  answer\n", '"already quoted"'])
def test_string_results_are_exact(api, text):
    assert api.tool(returns(text, str)).invoke({}) == text


@pytest.mark.parametrize("value", [None, True, 4, 1.5, [1, {"ok": True}], {"x": [None]}])
def test_missing_return_annotation_allows_json_values(api, value):
    assert json.loads(api.tool(returns(value)).invoke({})) == value


@pytest.mark.parametrize("value", [Connection(), object()])
def test_missing_return_annotation_rejects_unserializable_results(api, value):
    calls = []
    with pytest.raises(api.ToolError) as caught:
        api.tool(returns(value, calls=calls)).invoke({})
    assert_error(api, caught, "result", "result")
    assert calls == ["called"]


def test_model_mapping_return_and_serializer_failure(api):
    result = api.tool(returns({"id": "x", "text": "y"}, Document)).invoke({})
    assert json.loads(result) == {"id": "x", "text": "y"}
    with pytest.raises(api.ToolError) as caught:
        api.tool(returns(BrokenSerialization(text="x"), BrokenSerialization)).invoke({})
    assert_error(api, caught, "result")
    assert caught.value.__cause__ is not None


def test_execution_error_preserves_cause_and_runs_once(api):
    calls = []
    failure = RuntimeError("database failed")

    def fail() -> str:
        """Fail after starting work."""
        calls.append("started")
        raise failure

    with pytest.raises(api.ToolError) as caught:
        api.tool(fail).invoke({})
    assert_error(api, caught, "execution", "fail")
    assert caught.value.__cause__ is failure
    assert calls == ["started"]


def test_sync_function_in_async_mode_runs_inline(api):
    caller = threading.get_ident()

    def current_thread() -> int:
        """Return the execution thread identifier."""
        return threading.get_ident()

    assert json.loads(asyncio.run(api.tool(current_thread).ainvoke({}))) == caller


def test_async_cancellation_propagates_without_retry(api):
    calls = []

    async def scenario():
        started = asyncio.Event()
        wait_forever = asyncio.Event()

        async def blocked() -> str:
            """Wait until the caller cancels."""
            calls.append("started")
            started.set()
            await wait_forever.wait()
            return "unreachable"

        task = asyncio.create_task(api.tool(blocked).ainvoke({}))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert calls == ["started"]


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(api, failure_type):
    def stop() -> str:
        """Stop execution."""
        raise failure_type()

    with pytest.raises(failure_type):
        api.tool(stop).invoke({})


def test_parameter_names_can_overlap_pydantic_members(api):
    def read(model_dump: str, _key: int, argument_0: bool) -> list[str]:
        """Use ordinary Python parameter names."""
        return [model_dump, str(_key), str(argument_0)]

    tool = api.tool(read)
    assert set(tool.parameters["properties"]) == {"model_dump", "_key", "argument_0"}
    assert json.loads(tool.invoke({"model_dump": "x", "_key": 2, "argument_0": True})) == [
        "x",
        "2",
        "True",
    ]

"""Acceptance tests for docs/superpowers/specs/2026-09-07-callable-tools.md."""

import asyncio
import importlib
import inspect
import json
import threading
from collections import deque
from collections.abc import Callable, Iterator
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from functools import partial, wraps
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, field_serializer, field_validator
from typing_extensions import NotRequired, TypedDict


@pytest.fixture(scope="session")
def api():
    return importlib.import_module("slick.tools")


class Mode(str, Enum):
    draft = "draft"
    published = "published"


class Level(Enum):
    low = 1
    high = 2


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    limit: int = Field(default=5, ge=1, le=50)


class Document(BaseModel):
    id: str
    text: str


class Options(TypedDict):
    query: str
    limit: NotRequired[int]


class AliasedDocument(BaseModel):
    document_id: str = Field(alias="id")


class BrokenSerialization(BaseModel):
    text: str

    @field_serializer("text")
    def fail(self, value):
        raise RuntimeError("serializer failed")


class RecursiveNode(BaseModel):
    child: "RecursiveNode | None" = None


class Connection:
    pass


class LooseModel(BaseModel):
    payload: Any


class Documents:
    Key = str

    def __init__(self, text):
        self.text = text
        self.calls = []

    def read(self, key: str) -> str:
        """Retrieve a document by its key."""
        self.calls.append(key)
        return self.text

    def hidden(self) -> str:
        """A method the application has not supplied."""
        return "hidden"

    @classmethod
    def kind(cls) -> str:
        """Return the class name."""
        return cls.__name__

    @staticmethod
    def label(key: str) -> str:
        """Return a document label."""
        return key

    @staticmethod
    def typed_label(key: "Key") -> str:
        """Resolve a type declared on the owning class."""
        return key


def echo_for(annotation, seen=None):
    """Construct a real function with a parameterized, resolved annotation."""

    def echo(value):
        """Return the supplied value."""
        if seen is not None:
            seen.append(value)
        return value

    echo.__annotations__ = {"value": annotation, "return": annotation}
    return echo


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


def test_metadata_and_original_callable_are_preserved(api):
    def search(query: str, *, limit: int = 5) -> list[str]:
        """Find documents.

        Returns identifiers in relevance order.
        """
        return [query] * limit

    original_signature = inspect.signature(search)
    tool = api.Tool(search)
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


@pytest.mark.parametrize("name", ["", "bad name", "1lookup", "a" * 65, "lookup.read"])
def test_invalid_names(api, name):
    with pytest.raises(api.ToolError) as caught:
        api.Tool(echo_for(str), name=name)
    assert_error(api, caught, "definition")


@pytest.mark.parametrize("description", [None, "", " \n\t"])
def test_missing_description(api, description):
    def lookup(key: str) -> str:
        return key

    with pytest.raises(api.ToolError) as caught:
        api.Tool(lookup, description=description)
    assert_error(api, caught, "definition", "lookup")


def test_annotated_constraints_are_in_schema_and_enforced(api):
    annotation = Annotated[int, Field(ge=1, le=3, description="Number of documents")]
    seen = []
    tool = api.Tool(echo_for(annotation, seen))
    schema = tool.parameters["properties"]["value"]
    assert schema["minimum"] == 1 and schema["maximum"] == 3
    assert schema["description"] == "Number of documents"
    assert tool.invoke({"value": 2}) == "2"
    with pytest.raises(api.ToolError) as caught:
        tool.invoke({"value": 4})
    assert_error(api, caught, "arguments", "echo")
    assert seen == [2]


@pytest.mark.parametrize(
    "annotation,argument,expected",
    [
        (str, " hello ", " hello "),
        (int, 3, "3"),
        (float, 1.5, "1.5"),
        (float, 2, "2.0"),
        (bool, True, "true"),
        (type(None), None, "null"),
        (list[str], ["a", "b"], '["a", "b"]'),
        (dict[str, int], {"a": 2}, '{"a": 2}'),
        (list[dict[str, list[int]]], [{"a": [1, 2]}], '[{"a": [1, 2]}]'),
        (Literal["draft", "published"], "draft", "draft"),
        (Literal[1, 2], 2, "2"),
        (int | str, "three", "three"),
        (int | None, None, "null"),
        (int | None, 2, "2"),
        (Mode, "draft", "draft"),
        (Level, 1, "1"),
    ],
)
def test_supported_values_round_trip(api, annotation, argument, expected):
    seen = []
    output = api.Tool(echo_for(annotation, seen)).invoke({"value": argument})
    assert len(seen) == 1
    if isinstance(argument, str):
        assert output == expected
    else:
        assert json.loads(output) == json.loads(expected)
    if annotation in (Mode, Level):
        assert isinstance(seen[0], annotation)


@pytest.mark.parametrize(
    "annotation,value",
    [
        (int, "3"),
        (int, 3.0),
        (int, True),
        (float, "3.0"),
        (float, True),
        (bool, "true"),
        (bool, 1),
        (str, 123),
        (list[int], [1, "2"]),
        (dict[str, int], {"a": "2"}),
        (Literal["draft", "published"], "other"),
        (Mode, "other"),
        (Level, "1"),
    ],
)
def test_invalid_values_do_not_execute(api, annotation, value):
    seen = []
    with pytest.raises(api.ToolError) as caught:
        api.Tool(echo_for(annotation, seen)).invoke({"value": value})
    assert_error(api, caught, "arguments", "echo")
    assert "value" in str(caught.value)
    assert caught.value.__cause__ is not None
    assert seen == []


UNSUPPORTED = [
    Any,
    object,
    list,
    dict,
    dict[int, str],
    tuple[int, str],
    set[str],
    frozenset[str],
    deque[str],
    bytes,
    bytearray,
    memoryview,
    datetime,
    date,
    time,
    timedelta,
    UUID,
    Path,
    Decimal,
    complex,
    Connection,
    Callable[[str], str],
    type,
    range,
    Iterator[str],
    LooseModel,
    RecursiveNode,
    RootModel[list[int]],
    TypeVar("T"),
]


@pytest.mark.parametrize("annotation", UNSUPPORTED, ids=str)
def test_unsupported_input_types_fail_during_preparation(api, annotation):
    seen = []
    with pytest.raises(api.ToolError) as caught:
        api.Tool(echo_for(annotation, seen))
    assert_error(api, caught, "definition", "echo")
    assert seen == []


@pytest.mark.parametrize("annotation", [Any, bytes, Connection, tuple[int, str]])
def test_unsupported_return_annotations_fail_before_invocation(api, annotation):
    calls = []
    with pytest.raises(api.ToolError) as caught:
        api.Tool(returns(None, annotation, calls))
    assert_error(api, caught, "definition", "result")
    assert calls == []


def test_missing_and_unresolved_input_annotations(api):
    def missing(key):
        """Missing type."""
        return key

    def unresolved(key: "NotDefinedHere") -> str:  # noqa: F821
        """Unresolved type."""
        return key

    for function in (missing, unresolved):
        with pytest.raises(api.ToolError) as caught:
            api.Tool(function)
        assert_error(api, caught, "definition", function.__name__)
        assert "key" in str(caught.value)


def test_resolved_string_annotations(api):
    def search(request: "Query") -> "list[Document]":
        """Search using a typed request."""
        assert isinstance(request, Query)
        return [Document(id="one", text=request.query)]

    result = api.Tool(search).invoke({"request": {"query": "hello"}})
    assert json.loads(result) == [{"id": "one", "text": "hello"}]


def test_nested_model_constraints_and_typed_dict(api):
    tool = api.Tool(echo_for(Query))
    assert json.loads(tool.invoke({"value": {"query": "hello"}})) == {
        "query": "hello",
        "limit": 5,
    }
    for request in ({"query": "hello", "limit": 0}, {"query": "hello", "extra": 1}):
        with pytest.raises(api.ToolError) as caught:
            tool.invoke({"value": request})
        assert_error(api, caught, "arguments")
    seen = []
    result = api.Tool(echo_for(Options, seen)).invoke({"value": {"query": "hello"}})
    assert type(seen[0]) is dict
    assert json.loads(result) == {"query": "hello"}


def test_model_aliases_round_trip(api):
    result = api.Tool(echo_for(AliasedDocument)).invoke({"value": {"id": "intro"}})
    assert json.loads(result) == {"id": "intro"}


def test_parameter_alias_is_rejected(api):
    with pytest.raises(api.ToolError) as caught:
        api.Tool(echo_for(Annotated[str, Field(alias="other")]))
    assert_error(api, caught, "definition")


def test_unsupported_signatures(api):
    def positional(value: str, /) -> str:
        """Positional only."""
        return value

    def varargs(*values: str) -> str:
        """Variadic."""
        return "".join(values)

    def kwargs(**values: str) -> str:
        """Keyword variadic."""
        return str(values)

    def generator(value: str) -> Iterator[str]:
        """Generator."""
        yield value

    async def async_generator(value: str):
        """Async generator."""
        yield value

    class CallableObject:
        def __call__(self, value: str) -> str:
            """Callable object."""
            return value

    for function in (
        positional,
        varargs,
        kwargs,
        generator,
        async_generator,
        Documents.read,
        CallableObject(),
        partial(echo_for(str), value="x"),
    ):
        with pytest.raises(api.ToolError) as caught:
            api.Tool(function)
        assert_error(api, caught, "definition")


def test_defaults_nullable_required_and_keyword_only(api):
    def search(query: str, nullable: str | None, *, limit: int = 5) -> list[str]:
        """Search with independent nullable and default arguments."""
        return [query, str(nullable), str(limit)]

    tool = api.Tool(search)
    assert set(tool.parameters["required"]) == {"query", "nullable"}
    assert json.loads(tool.invoke({"query": "x", "nullable": None})) == ["x", "None", "5"]
    assert json.loads(tool.invoke({"query": "x", "nullable": "y", "limit": 2})) == [
        "x",
        "y",
        "2",
    ]
    for arguments in ({"query": "x"}, {"query": "x", "nullable": None, "limit": None}):
        with pytest.raises(api.ToolError) as caught:
            tool.invoke(arguments)
        assert_error(api, caught, "arguments")


def test_invalid_default_and_original_default_identity(api):
    def wrong(value: int = "5") -> int:
        """Incorrect default."""
        return value

    with pytest.raises(api.ToolError) as caught:
        api.Tool(wrong)
    assert_error(api, caught, "definition", "wrong")

    default = [1]

    def same(value: list[int] = default) -> bool:
        """Preserve Python's default identity."""
        return value is default

    tool = api.Tool(same)
    assert tool.invoke({}) == "true"
    assert tool.invoke({"value": [1]}) == "false"


@pytest.mark.parametrize("arguments", [None, [], "{}", {"extra": 1}])
def test_zero_argument_tool_rejects_invalid_objects(api, arguments):
    calls = []
    tool = api.Tool(returns("ok", str, calls))
    assert tool.parameters["properties"] == {}
    with pytest.raises(api.ToolError) as caught:
        tool.invoke(arguments)
    assert_error(api, caught, "arguments")
    assert calls == []


def test_root_extra_or_missing_arguments_do_not_execute(api):
    seen = []
    tool = api.Tool(echo_for(str, seen))
    for arguments in ({}, {"value": "x", "extra": 2}):
        with pytest.raises(api.ToolError) as caught:
            tool.invoke(arguments)
        assert_error(api, caught, "arguments")
    assert seen == []


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), {1: "x"}, (1, 2), Connection()]
)
def test_non_json_arguments_fail_without_execution(api, value):
    seen = []
    with pytest.raises(api.ToolError) as caught:
        api.Tool(echo_for(float, seen)).invoke({"value": value})
    assert_error(api, caught, "arguments")
    assert seen == []


def test_cyclic_input_fails_and_valid_input_is_not_mutated(api):
    def change(values: list[int]) -> list[int]:
        """Change the tool's own argument copy."""
        values.append(2)
        return values

    tool = api.Tool(change)
    original = {"values": [1]}
    assert json.loads(tool.invoke(original)) == [1, 2]
    assert original == {"values": [1]}
    cycle = []
    cycle.append(cycle)
    with pytest.raises(api.ToolError) as caught:
        tool.invoke({"values": cycle})
    assert_error(api, caught, "arguments")


def test_bound_instances_class_and_static_methods(api):
    first, second = Documents("first"), Documents("second")
    a, b = api.Tool(first.read), api.Tool(second.read)
    assert set(a.parameters["properties"]) == {"key"}
    assert a.invoke({"key": "intro"}) == "first"
    assert b.invoke({"key": "other"}) == "second"
    assert first.calls == ["intro"] and second.calls == ["other"]
    assert first.read("direct") == "first"
    assert api.Tool(Documents.kind).invoke({}) == "Documents"
    assert api.Tool(Documents.label).invoke({"key": "intro"}) == "intro"


def test_wrapped_signature_executes_wrapper(api):
    calls = []

    def original(key: str) -> str:
        """Read a value."""
        return key

    @wraps(original)
    def wrapped(*args, **kwargs):
        calls.append("wrapper")
        return original(*args, **kwargs)

    assert api.Tool(wrapped).invoke({"key": "x"}) == "x"
    assert calls == ["wrapper"]


def test_preparation_is_explicit_ordered_and_has_no_execution(api):
    docs = Documents("text")
    explicit = api.Tool(docs.read, name="read_again")
    prepared = api.prepare_tools([docs.read, explicit])
    assert list(prepared) == ["read", "read_again"]
    assert prepared["read_again"] is explicit
    assert "hidden" not in prepared
    assert docs.calls == []
    assert api.prepare_tools([]) == {}
    for supplied in ([docs.read, docs.read], [docs.read, Documents("other").read], [docs.read, 1]):
        with pytest.raises(api.ToolError) as caught:
            api.prepare_tools(supplied)
        assert_error(api, caught, "definition")
    assert docs.calls == []


@pytest.mark.parametrize("text", ["", "  answer\n", '"already quoted"'])
def test_string_results_are_exact(api, text):
    assert api.Tool(returns(text, str)).invoke({}) == text


@pytest.mark.parametrize("value", [None, True, 4, 1.5, [1, {"ok": True}], {"x": [None]}])
def test_missing_return_annotation_allows_json_values(api, value):
    assert json.loads(api.Tool(returns(value)).invoke({})) == value


@pytest.mark.parametrize(
    "value", [Connection(), b"abc", (1, 2), {1, 2}, Document(id="x", text="y"), {1: "x"}]
)
def test_missing_return_annotation_rejects_non_json_results(api, value):
    calls = []
    with pytest.raises(api.ToolError) as caught:
        api.Tool(returns(value, calls=calls)).invoke({})
    assert_error(api, caught, "result", "result")
    assert calls == ["called"]


@pytest.mark.parametrize(
    "annotation,value",
    [(int, "3"), (int, True), (type(None), "done"), (list[int], [1, "2"]), (Mode, "draft")],
)
def test_invalid_annotated_result_is_not_retried(api, annotation, value):
    calls = []
    with pytest.raises(api.ToolError) as caught:
        api.Tool(returns(value, annotation, calls)).invoke({})
    assert_error(api, caught, "result", "result")
    assert caught.value.__cause__ is not None
    assert calls == ["called"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_results_never_become_null(api, value):
    for annotation, result in (
        (float, value),
        (list[float], [value]),
        (inspect.Signature.empty, value),
    ):
        with pytest.raises(api.ToolError) as caught:
            api.Tool(returns(result, annotation)).invoke({})
        assert_error(api, caught, "result")


def test_model_mapping_return_and_serializer_failure(api):
    result = api.Tool(returns({"id": "x", "text": "y"}, Document)).invoke({})
    assert json.loads(result) == {"id": "x", "text": "y"}
    with pytest.raises(api.ToolError) as caught:
        api.Tool(returns(BrokenSerialization(text="x"), BrokenSerialization)).invoke({})
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
        api.Tool(fail).invoke({})
    assert_error(api, caught, "execution", "fail")
    assert caught.value.__cause__ is failure
    assert calls == ["started"]


def test_async_function_and_sync_mode_rejection(api):
    calls = []

    async def lookup(key: str) -> str:
        """Read asynchronously."""
        calls.append(key)
        await asyncio.sleep(0)
        return key

    tool = api.Tool(lookup)
    with pytest.raises(api.ToolError) as caught:
        tool.invoke({"key": "sync"})
    assert_error(api, caught, "execution")
    assert calls == []
    assert asyncio.run(tool.ainvoke({"key": "async"})) == "async"
    assert calls == ["async"]
    with pytest.raises(api.ToolError) as caught:
        asyncio.run(tool.ainvoke({"key": 1}))
    assert_error(api, caught, "arguments")
    assert calls == ["async"]


def test_sync_function_in_async_mode_runs_inline(api):
    caller = threading.get_ident()

    def current_thread() -> int:
        """Return the execution thread identifier."""
        return threading.get_ident()

    assert json.loads(asyncio.run(api.Tool(current_thread).ainvoke({}))) == caller


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

        task = asyncio.create_task(api.Tool(blocked).ainvoke({}))
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
        api.Tool(stop).invoke({})


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unexpected_coroutine_is_closed_not_executed(api, asynchronous):
    created = []
    executed = []

    async def inner():
        executed.append("executed")
        return "value"

    def unexpected() -> str:
        """Incorrectly return an awaitable from a synchronous function."""
        coroutine = inner()
        created.append(coroutine)
        return coroutine

    tool = api.Tool(unexpected)
    try:
        with pytest.raises(api.ToolError) as caught:
            if asynchronous:
                asyncio.run(tool.ainvoke({}))
            else:
                tool.invoke({})
        assert_error(api, caught, "result")
        assert executed == []
        assert inspect.getcoroutinestate(created[0]) == inspect.CORO_CLOSED
    finally:
        for coroutine in created:
            coroutine.close()


def test_parameter_names_can_overlap_pydantic_members(api):
    def read(model_dump: str, _key: int, argument_0: bool) -> list[str]:
        """Use ordinary Python parameter names."""
        return [model_dump, str(_key), str(argument_0)]

    tool = api.Tool(read)
    assert set(tool.parameters["properties"]) == {"model_dump", "_key", "argument_0"}
    assert json.loads(tool.invoke({"model_dump": "x", "_key": 2, "argument_0": True})) == [
        "x",
        "2",
        "True",
    ]


def test_inherited_method_uses_defining_class_annotation_namespace(api):
    class Parent:
        Value = str

        def lookup(self, value: "Value") -> str:
            """Look up a value declared in the defining class."""
            return value

    class Child(Parent):
        Value = int

    tool = api.Tool(Child().lookup)
    assert tool.invoke({"value": "hello"}) == "hello"


def test_static_method_uses_defining_class_annotation_namespace(api):
    assert api.Tool(Documents.typed_label).invoke({"key": "intro"}) == "intro"


def test_validator_cannot_introduce_nonfinite_arguments(api):
    class InvalidFloat(BaseModel):
        value: float

        @field_validator("value")
        @classmethod
        def invalid(cls, value):
            return float("inf")

    seen = []
    with pytest.raises(api.ToolError) as caught:
        api.Tool(echo_for(InvalidFloat, seen)).invoke({"value": {"value": 1.0}})
    assert_error(api, caught, "arguments")
    assert seen == []


def test_python_signature_controls_required_fields_despite_field_metadata(api):
    annotation = Annotated[int, Field(default_factory=lambda: 7)]
    seen = []
    tool = api.Tool(echo_for(annotation, seen))
    assert tool.parameters["required"] == ["value"]
    with pytest.raises(api.ToolError) as caught:
        tool.invoke({})
    assert_error(api, caught, "arguments")
    assert seen == []


def test_unexpected_future_is_not_cancelled(api):
    async def scenario():
        future = asyncio.get_running_loop().create_future()
        try:
            with pytest.raises(api.ToolError) as caught:
                await api.Tool(returns(future)).ainvoke({})
            assert_error(api, caught, "result")
            assert not future.cancelled()
        finally:
            future.cancel()

    asyncio.run(scenario())

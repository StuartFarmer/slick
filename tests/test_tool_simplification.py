import asyncio
import json

import pytest

from slick.tools import prepare_tools, tool


def test_annotations_describe_tools_without_validating_calls():
    seen = []

    @tool
    def echo(value: int = "default") -> int:
        seen.append(value)
        return value

    assert echo.description == ""
    assert echo.parameters["properties"]["value"]["type"] == "integer"
    assert echo.invoke({}) == "default"
    assert echo.invoke({"value": "unchanged"}) == "unchanged"
    assert seen == ["default", "unchanged"]


def test_preparation_uses_normal_dictionary_replacement():
    first = tool(lambda: "first", name="any name")
    second = tool(lambda: "second", name="any name")
    assert prepare_tools((first, second))["any name"].invoke({}) == "second"


def test_arguments_are_passed_through_without_conversion():
    @tool
    def append(values: list[int]):
        values.append("two")
        return values

    values = [1]
    assert json.loads(append.invoke({"values": values})) == [1, "two"]
    assert values == [1, "two"]


def test_async_tool_and_bound_method_use_python_arguments():
    class Store:
        def __init__(self):
            self.calls = []

        async def read(self, key: str, *, limit: int = 5):
            self.calls.append((key, limit))
            return key

    store = Store()
    read = tool(store.read)
    assert set(read.parameters["properties"]) == {"key", "limit"}
    assert asyncio.run(read.ainvoke({"key": "intro"})) == "intro"
    assert store.calls == [("intro", 5)]


def test_python_reports_missing_and_unexpected_arguments():
    from slick.tools import ToolError

    @tool
    def echo(value):
        return value

    for arguments in ({}, {"value": 1, "extra": True}):
        with pytest.raises(ToolError) as raised:
            echo.invoke(arguments)
        assert raised.value.phase == "execution"
        assert isinstance(raised.value.__cause__, TypeError)

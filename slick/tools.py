"""Expose ordinary Python functions as tools, without a provider loop."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python
from typing_extensions import NotRequired, TypedDict


class ToolError(ValueError):
    """A tool definition, execution, or serialization failure, with its cause."""

    def __init__(self, name: str, phase: str, detail: str):
        self.name = name
        self.phase = phase
        super().__init__(f"Tool {name!r} ({phase}): {detail}")


class ToolRequest(TypedDict):
    id: str
    name: str
    arguments: dict | str
    argument_error: NotRequired[str | None]


class ToolResult(TypedDict):
    request: ToolRequest
    content: str
    is_error: NotRequired[bool]


def make_request(id: str, name: str, raw: str | dict) -> ToolRequest:
    request = {"id": id, "name": name, "arguments": raw}
    if isinstance(raw, str):
        try:
            request["arguments"] = json.loads(raw)
        except json.JSONDecodeError as exc:
            request["argument_error"] = str(exc)
    return request


@contextmanager
def _tool_phase(name: str, phase: str):
    """Attach tool context at a boundary; cancellation and process exits propagate."""
    try:
        yield
    except Exception as exc:
        raise ToolError(name, phase, str(exc)) from exc


class Tool:
    """A function or bound method with an explicit model-facing name and description.

    invoke/ainvoke return serialized tool results. Direct calls to the original
    function remain unchanged. Synchronous handlers run inline in either mode.
    """

    def __init__(
        self,
        function: Callable,
        *,
        name: str,
        description: str,
    ):
        self.name = name
        with _tool_phase(self.name, "definition"):
            self.description = inspect.cleandoc(description).strip()
            self._parameters = TypeAdapter(function).json_schema(by_alias=False)
        self._function = function
        self._async = inspect.iscoroutinefunction(function)

    @property
    def parameters(self) -> dict:
        """A fresh local input schema; provider conversion may safely modify it."""
        return deepcopy(self._parameters)

    def _result(self, result) -> str:
        with _tool_phase(self.name, "result"):
            return result if isinstance(result, str) else json.dumps(to_jsonable_python(result))

    def invoke(self, arguments: dict) -> str:
        """Pass arguments to a synchronous function once and serialize its result."""
        if self._async:
            raise ToolError(self.name, "execution", "async functions require ainvoke")
        with _tool_phase(self.name, "execution"):
            result = self._function(**arguments)
        return self._result(result)

    async def ainvoke(self, arguments: dict) -> str:
        """Await an async handler or execute a sync handler inline, without retries."""
        with _tool_phase(self.name, "execution"):
            result = (
                await self._function(**arguments) if self._async else self._function(**arguments)
            )
        return self._result(result)


def tool(
    function: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable], Tool]:
    """Decorate a function as a Tool, defaulting to its name and inspected docstring.

    Supports @tool, @tool(), and @tool(name=..., description=...). Bound methods
    can be supplied directly as tool(instance.method). No function is executed.
    """
    if function is None:
        return lambda fn: tool(fn, name=name, description=description)
    with _tool_phase(name if name is not None else "tool", "definition"):
        name = function.__name__ if name is None else name
        description = (inspect.getdoc(function) or "") if description is None else description
    return Tool(function, name=name, description=description)


def prepare_tools(functions: list) -> dict[str, Tool]:
    """Prepare exactly the supplied functions, without invoking or registering them globally."""
    prepared = [
        function if isinstance(function, Tool) else tool(function) for function in functions
    ]
    return {item.name: item for item in prepared}


__all__ = [
    "Tool",
    "ToolError",
    "ToolRequest",
    "ToolResult",
    "make_request",
    "prepare_tools",
    "tool",
]

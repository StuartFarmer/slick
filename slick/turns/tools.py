"""Validate and invoke ordinary Python functions as tools, without a provider loop."""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model
from pydantic.fields import FieldInfo
from typing_extensions import NotRequired, Required, is_typeddict


class ToolError(ValueError):
    """A tool definition, argument, execution, or result failure, with its cause."""

    def __init__(self, name: str, phase: str, detail: str):
        self.name = name
        self.phase = phase
        super().__init__(f"Tool {name!r} ({phase}): {detail}")


@contextmanager
def _tool_phase(name: str, phase: str):
    """Attach tool context at a boundary; cancellation and process exits propagate."""
    try:
        yield
    except Exception as exc:
        raise ToolError(name, phase, str(exc)) from exc


def _check_annotation(annotation, path: str, parents: tuple = ()) -> None:
    """Enforce the intentionally small, recursive type boundary before schema creation."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation in (str, int, float, bool, None, type(None)):
        return
    if origin in (Annotated, Required, NotRequired):
        _check_annotation(args[0], path, parents)
        return
    if origin in (Union, UnionType):
        for branch in args:
            _check_annotation(branch, path, parents)
        return
    if origin is Literal and args and all(type(x) in (str, int, bool, type(None)) for x in args):
        return
    if origin is list and len(args) == 1:
        _check_annotation(args[0], f"{path}[]", parents)
        return
    if origin is dict and len(args) == 2 and args[0] is str:
        _check_annotation(args[1], f"{path}.*", parents)
        return
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        value_types = {type(member.value) for member in annotation}
        if value_types in ({str}, {int}):
            return
    elif is_typeddict(annotation):
        if annotation in parents:
            raise TypeError(f"{path}: recursive types are unsupported")
        for name, field_type in get_type_hints(annotation, include_extras=True).items():
            _check_annotation(field_type, f"{path}.{name}", (*parents, annotation))
        return
    elif inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        if annotation in parents or annotation.__pydantic_root_model__:
            raise TypeError(f"{path}: recursive models and RootModel are unsupported")
        annotation.model_rebuild()
        for name, field in annotation.model_fields.items():
            _check_annotation(field.annotation, f"{path}.{name}", (*parents, annotation))
        return
    raise TypeError(f"{path}: unsupported annotation {annotation!r}")


def _value_items(value, path: str, structured: bool):
    """Yield child paths and values for the supported containers."""
    if structured and isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield f"{path}.{name}", getattr(value, name)
        for name, extra in (value.model_extra or {}).items():
            yield f"{path}.{name}", extra
        return
    if type(value) is list:
        for index, item in enumerate(value):
            yield f"{path}[{index}]", item
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path}: object keys must be strings")
            yield f"{path}.{key}", item
        return
    raise TypeError(f"{path}: unsupported value of type {type(value).__name__}")


def _check_value(value, path: str, *, structured: bool = False, parents: tuple = ()) -> None:
    """Reject non-JSON values and nonfinite numbers before serialization can coerce them."""
    if structured and isinstance(value, Enum):
        _check_value(value.value, path)
        return
    if type(value) in (str, int, bool, type(None)):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: numbers must be finite")
        return
    if id(value) in parents:
        raise ValueError(f"{path}: cyclic values are unsupported")
    for child_path, child in _value_items(value, path, structured):
        _check_value(child, child_path, structured=structured, parents=(*parents, id(value)))


def _defining_class(function, target):
    """Find the owner of a bound or static method, including inherited methods."""
    owner = getattr(function, "__self__", None)
    if owner is None:
        # Static methods have no bound owner; locate their defining class by name.
        scope = target.__globals__
        for part in target.__qualname__.split(".")[:-1]:
            candidate = scope.get(part)
            if not inspect.isclass(candidate):
                return None
            owner = candidate
            scope = vars(candidate)
    if owner is None:
        return None
    cls = owner if inspect.isclass(owner) else type(owner)
    for base in cls.__mro__:
        members = (getattr(member, "__func__", member) for member in vars(base).values())
        if any(
            inspect.isfunction(member) and inspect.unwrap(member) is target for member in members
        ):
            return base
    return cls


def _hints(function) -> dict:
    """Resolve in the defining module/class, never in the caller's stack frame."""
    target = inspect.unwrap(function)
    if inspect.ismethod(target):
        target = target.__func__
    cls = _defining_class(function, target)
    localns = None if cls is None else {**vars(cls), cls.__name__: cls}
    try:
        return get_type_hints(
            target, globalns=target.__globals__, localns=localns, include_extras=True
        )
    except Exception as exc:
        # Include parameter names as well as the unresolved annotation in diagnostics.
        raise TypeError(f"could not resolve annotations {target.__annotations__}: {exc}") from exc


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
        raise ValueError("name must match [A-Za-z_][A-Za-z0-9_]{0,63}")


def _description(function: Callable, description: str | None) -> str:
    doc = inspect.getdoc(function) if description is None else description
    if not isinstance(doc, str) or not doc.strip():
        raise ValueError("a nonempty docstring or description is required")
    return inspect.cleandoc(doc).strip()


def _signature(function: Callable) -> inspect.Signature:
    """Accept only callables whose arguments can be supplied as named JSON fields."""
    if not (inspect.isfunction(function) or inspect.ismethod(function)):
        raise TypeError("supply a Python function or bound method")
    target = inspect.unwrap(function)
    if inspect.isgeneratorfunction(target) or inspect.isasyncgenfunction(target):
        raise TypeError("generator functions are unsupported")
    signature = inspect.signature(function)
    for index, parameter in enumerate(signature.parameters.values()):
        if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            raise TypeError(f"{parameter.name}: unsupported parameter kind {parameter.kind}")
        if index == 0 and parameter.name in {"self", "cls"} and not inspect.ismethod(function):
            raise TypeError(f"{parameter.name}: supply a bound method")
    return signature


def _parameter_default(parameter: inspect.Parameter, annotation):
    if parameter.default is inspect.Parameter.empty:
        return ...
    try:
        validated = TypeAdapter(annotation).validate_python(parameter.default, strict=True)
        _check_value(validated, parameter.name, structured=True)
    except Exception as exc:
        raise ValueError(f"{parameter.name}: invalid default: {exc}") from exc
    return parameter.default


def _parameter_field(parameter: inspect.Parameter, annotation) -> FieldInfo:
    """Copy annotation metadata and apply the Python signature's defaults and name."""
    _check_annotation(annotation, parameter.name)
    info = deepcopy(FieldInfo.from_annotation(annotation))
    if any(x is not None for x in (info.alias, info.validation_alias, info.serialization_alias)):
        raise TypeError(f"{parameter.name}: parameter aliases are unsupported")
    # A single FieldInfo works on Pydantic 2.0 as well as current versions.
    # Defaults belong to the Python signature, not Annotated metadata.
    info.default = Field(default=_parameter_default(parameter, annotation)).default
    info.default_factory = None
    info.alias = info.validation_alias = info.serialization_alias = parameter.name
    return info


def _argument_model(name: str, signature: inspect.Signature, hints: dict) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for index, parameter in enumerate(signature.parameters.values()):
        if parameter.name not in hints:
            raise TypeError(f"{parameter.name}: missing input annotation")
        info = _parameter_field(parameter, hints[parameter.name])
        # Internal names avoid collisions with Pydantic methods/private fields.
        fields[f"argument_{index}"] = (info.annotation, info)
    return create_model(f"{name}Arguments", __config__=ConfigDict(extra="forbid"), **fields)


def _return_adapter(hints: dict) -> TypeAdapter | None:
    if "return" not in hints:
        return None
    _check_annotation(hints["return"], "return")
    return TypeAdapter(hints["return"])


def _validate_arguments(arguments: dict, model: type[BaseModel]) -> dict:
    """Convert JSON arguments to Python values; omitted keywords keep Python defaults."""
    if type(arguments) is not dict:
        raise TypeError("arguments must be a JSON object")
    _check_value(arguments, "arguments")
    validated = model.model_validate_json(json.dumps(arguments, allow_nan=False), strict=True)
    kwargs = {
        info.alias: getattr(validated, field)
        for field, info in model.model_fields.items()
        if field in validated.model_fields_set
    }
    _check_value(kwargs, "arguments", structured=True)
    return kwargs


def _serialize_result(result, adapter: TypeAdapter | None = None) -> str:
    """Validate the declared return type and serialize only supported JSON values."""
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("unexpected awaitable result; use an async function")
    if adapter is not None:
        result = adapter.validate_python(result, strict=True)
        _check_value(result, "return", structured=True)
        result = adapter.dump_python(result, mode="json", by_alias=True)
    _check_value(result, "return")
    return result if type(result) is str else json.dumps(result, allow_nan=False)


class Tool:
    """Optional wrapper for a function or bound method and its model-facing contract.

    invoke/ainvoke return serialized tool results. Direct calls to the original
    function remain unchanged. Synchronous handlers run inline in either mode.
    """

    def __init__(
        self,
        function: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
    ):
        self.name = getattr(function, "__name__", type(function).__name__) if name is None else name
        with _tool_phase(self.name, "definition"):
            _check_name(self.name)
            signature = _signature(function)
            self.description = _description(function, description)
            hints = _hints(function)
            self._arguments = _argument_model(self.name, signature, hints)
            self._schema = self._arguments.model_json_schema()
            self._returns = _return_adapter(hints)
        self._function = function
        self._async = inspect.iscoroutinefunction(function)

    @property
    def parameters(self) -> dict:
        """A fresh local input schema; provider conversion may safely modify it."""
        return deepcopy(self._schema)

    def _kwargs(self, arguments: dict) -> dict:
        with _tool_phase(self.name, "arguments"):
            return _validate_arguments(arguments, self._arguments)

    def _result(self, result) -> str:
        with _tool_phase(self.name, "result"):
            return _serialize_result(result, self._returns)

    def invoke(self, arguments: dict) -> str:
        """Validate arguments, invoke a synchronous function once, and serialize its result."""
        if self._async:
            raise ToolError(self.name, "execution", "async functions require ainvoke")
        kwargs = self._kwargs(arguments)
        with _tool_phase(self.name, "execution"):
            result = self._function(**kwargs)
        return self._result(result)

    async def ainvoke(self, arguments: dict) -> str:
        """Await an async handler or execute a sync handler inline, without retries."""
        kwargs = self._kwargs(arguments)
        with _tool_phase(self.name, "execution"):
            result = await self._function(**kwargs) if self._async else self._function(**kwargs)
        return self._result(result)


def prepare_tools(functions: list) -> dict[str, Tool]:
    """Prepare exactly the supplied functions, without invoking or registering them globally."""
    if not isinstance(functions, list):
        raise ToolError("tools", "definition", "supply a list of functions or Tool instances")
    prepared = {}
    for function in functions:
        tool = function if isinstance(function, Tool) else Tool(function)
        if tool.name in prepared:
            raise ToolError(tool.name, "definition", "duplicate tool name; supply an explicit name")
        prepared[tool.name] = tool
    return prepared

"""Local JSON contracts; no history, provider state, or tool execution."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TypedDict

from ._functions import _check_value, prepare_tools


class _RequestFields(TypedDict):
    id: str
    name: str
    arguments: dict | str


class ToolRequest(_RequestFields, total=False):
    argument_error: str


class _ResultFields(TypedDict):
    request: ToolRequest
    content: str


class ToolResult(_ResultFields, total=False):
    is_error: bool


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate argument key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite number {value}")


def decode_arguments(raw: str | dict) -> tuple[dict | str, str | None]:
    if isinstance(raw, dict):
        _check_value(raw, "arguments")
        return deepcopy(raw), None
    if not isinstance(raw, str):
        raise ValueError("arguments must be a JSON object or JSON string")
    try:
        arguments = json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        _check_value(arguments, "arguments")
        return arguments, None
    except (ValueError, RecursionError) as exc:
        return raw, f"Invalid arguments: {exc}"


def make_request(id: str, name: str, raw: str | dict) -> ToolRequest:
    arguments, error = decode_arguments(raw)
    request = {"id": id, "name": name, "arguments": arguments}
    if error:
        request["argument_error"] = error
    return validate_requests([request])[0]


def _fields(value, required, optional, label):
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError(f"{label} requires fields {sorted(required)}")
    if value.keys() - required - optional:
        raise ValueError(f"unknown {label} fields: {sorted(value.keys() - required - optional)}")


def validate_requests(requests) -> list[ToolRequest]:
    if not isinstance(requests, list):
        raise ValueError("tool requests must be a list")
    seen = set()
    for request in requests:
        _fields(request, {"id", "name", "arguments"}, {"argument_error"}, "tool request")
        for key in ("id", "name"):
            if not isinstance(request[key], str) or not request[key].strip():
                raise ValueError(f"tool request {key} must be nonempty")
        if request["id"] in seen:
            raise ValueError(f"duplicate tool request id {request['id']!r}")
        seen.add(request["id"])
        arguments = request["arguments"]
        error = request.get("argument_error")
        if isinstance(arguments, dict):
            _check_value(arguments, "arguments")
            if error is not None:
                raise ValueError("decoded arguments must not have argument_error")
        elif isinstance(arguments, str):
            if not isinstance(error, str) or not error.strip():
                raise ValueError("raw arguments require argument_error")
            if decode_arguments(arguments)[1] is None:
                raise ValueError("valid JSON arguments must be decoded")
        else:
            raise ValueError("arguments must be a JSON object or malformed JSON string")
    return deepcopy(requests)


def validate_results(results) -> list[ToolResult]:
    if results is None:
        return []
    if not isinstance(results, list):
        raise ValueError("tool_results must be a list")
    for result in results:
        _fields(result, {"request", "content"}, {"is_error"}, "tool result")
        if (
            not isinstance(result["content"], str)
            or type(result.get("is_error", False)) is not bool
        ):
            raise ValueError("tool result requires string content and boolean is_error")
    requests = validate_requests([result["request"] for result in results])
    normalized = []
    for request, result in zip(requests, results, strict=False):
        error = result.get("is_error", False)
        if request.get("argument_error") and not error:
            raise ValueError("malformed tool request requires an error result")
        normalized.append({"request": request, "content": result["content"], "is_error": error})
    return normalized


def prepare_call(context, tools, tool_results):
    results = validate_results(tool_results)
    if not isinstance(context, str) or (not context and not results):
        raise ValueError("context must be a string and cannot be empty without tool results")
    return prepare_tools([] if tools is None else tools), results


def validate_response(response) -> tuple[str, list[ToolRequest]]:
    if not isinstance(response, tuple) or len(response) != 2:
        raise ValueError("provider must return a (text, tool_requests) tuple")
    text, requests = response
    if not isinstance(text, str):
        raise ValueError("response text must be a string")
    requests = validate_requests(requests)
    return text, requests

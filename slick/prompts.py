"""Jinja rendering, text execution, and typed Python results.

`Prompt(template)(**variables)` and `render(template, **variables)` only
render text. `parse(text, returns)`
only validates a result. `@prompt(provider=...)` composes rendering,
provider.call/acall, and parsing. Applications own persistence and retries.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, get_type_hints

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import TypeAdapter

TEMPLATE_ROOT = Path("prompts")


class PromptError(Exception):
    """A text-only prompt received tool requests."""


def render(template: str, /, **variables: Any) -> str:
    """Render a file under TEMPLATE_ROOT, without model calls or output instructions."""
    return _render(template, None, variables)


class Prompt:
    """A template file bound to a callable renderer, independent of execution.

    Files resolve under TEMPLATE_ROOT at call time, including includes and
    imports. Construction stores only the filename; each call renders afresh.
    """

    def __init__(self, template: str):
        self.template = template

    def __call__(self, /, **variables: Any) -> str:
        return render(self.template, **variables)


def parse(text: str, returns: Any = str) -> Any:
    """Return text unchanged, or validate JSON against a Pydantic-compatible type."""
    return text if returns is str else TypeAdapter(returns).validate_json(text)


def prompt(
    fn: Callable | None = None,
    *,
    template: str | None = None,
    provider: Any = None,
) -> Callable:
    """Render, call the supplied provider, and parse the annotated return type.

    Parsing failures propagate to the caller.
    .render() runs the function body in its sync/async mode without provider calls.
    """
    if fn is None:
        return lambda inner: prompt(inner, template=template, provider=provider)
    signature = inspect.signature(fn)
    docstring = inspect.getdoc(fn)
    returns = get_type_hints(fn).get("return", str)
    adapter = None if returns is str else TypeAdapter(returns)
    format_heading = "# Output Format"
    format_block = (
        f"{format_heading}\n\n"
        "Respond with ONLY a JSON value matching this schema — no preamble, no "
        "commentary, nothing outside the JSON.\n\n"
        f"{json.dumps(adapter.json_schema(), indent=2)}\n"
        if adapter is not None
        else ""
    )

    def render_context(arguments, extra):
        context = dict(arguments)
        if extra is not None:
            context.update(extra)
        context.setdefault("output_format", format_block)
        text = _render(template, docstring, context)
        if format_block and format_heading not in text:
            text = f"{text}\n\n---\n\n{format_block}"
        return text

    def parse_response(response):
        text, requests = response
        if requests:
            raise PromptError(
                "This prompt expects final text; handle tool requests in application code."
            )
        return text if adapter is None else adapter.validate_json(text)

    if inspect.iscoroutinefunction(fn):

        async def render_call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            extra = await fn(*bound.args, **bound.kwargs)
            return render_context(bound.arguments, extra)

        @wraps(fn)
        async def call(*args, **kwargs):
            text = await render_call(*args, **kwargs)
            return parse_response(await provider.acall(text))
    else:

        def render_call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            extra = fn(*bound.args, **bound.kwargs)
            return render_context(bound.arguments, extra)

        @wraps(fn)
        def call(*args, **kwargs):
            text = render_call(*args, **kwargs)
            return parse_response(provider.call(text))

    call.render = render_call
    return call


def _render(name: str | None, docstring: str | None, variables: Mapping) -> str:
    """A file under TEMPLATE_ROOT or an inline docstring, loaded on each render."""
    environment = _environment()
    template = (
        environment.from_string(docstring) if name is None else environment.get_template(name)
    )
    return template.render(**variables).strip()


def _environment() -> Environment:
    """Built per use so TEMPLATE_ROOT stays assignable at runtime."""
    return Environment(
        loader=FileSystemLoader(TEMPLATE_ROOT),
        undefined=StrictUndefined,  # a typo'd variable must fail, not render as ""
        trim_blocks=True,
        lstrip_blocks=True,
    )

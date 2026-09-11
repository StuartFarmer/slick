"""Jinja rendering, text execution, and typed Python results.

`Prompt(template)(**variables)` and `render(template, **variables)` only
render text. `parse(text, returns)`
only validates a result. `@prompt` renders, calls a runtime provider, parses,
and runs the function body as postprocessing. Applications own persistence and retries.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any

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


def _render_prompt(prompt: Prompt, output_type: Any, variables: dict) -> str:
    """Supply the output schema for provider and session prompt calls."""
    if output_type is not None:
        variables["schema"] = TypeAdapter(output_type).json_schema()
    return prompt(**variables)


def prompt(
    fn: Callable | None = None,
    *,
    template: str | None = None,
    output_type: Any = None,
    max_turns: int = 20,
) -> Callable:
    """Generate first, then run the body with optional keyword-only generated input.

    Supply exactly one of provider= (one call) or session= (a bounded tool conversation).
    output_type controls final parsing, not the return annotation.
    None or Ellipsis from the body passes the generated value through.
    .render() renders inputs only; it never calls the provider or the body.
    """
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    if fn is None:
        return lambda inner: prompt(
            inner, template=template, output_type=output_type, max_turns=max_turns
        )
    signature = inspect.signature(fn)
    for name in ("provider", "session"):
        if name in signature.parameters:
            raise TypeError(f"{name} is reserved for the call's execution resource")
    generated_parameter = signature.parameters.get("generated")
    if generated_parameter and generated_parameter.kind != inspect.Parameter.KEYWORD_ONLY:
        raise TypeError("generated must be a keyword-only parameter")
    inputs = signature.replace(
        parameters=[p for name, p in signature.parameters.items() if name != "generated"]
    )
    docstring = inspect.getdoc(fn)
    schema = None if output_type is None else TypeAdapter(output_type).json_schema()
    format_heading = "# Output Format"
    format_block = (
        f"{format_heading}\n\n"
        "Respond with ONLY a JSON value matching this schema — no preamble, no "
        "commentary, nothing outside the JSON.\n\n"
        f"{json.dumps(schema, indent=2)}\n"
        if schema is not None
        else ""
    )

    def bind_inputs(args, kwargs):
        for name in ("generated", "provider", "session"):
            if name in kwargs:
                raise TypeError(f"{name} is reserved and cannot be supplied as template input")
        bound = inputs.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound

    def render_context(bound):
        context = dict(bound.arguments)
        if "self" in context:
            context["instance"] = context.pop("self")  # Jinja reserves self for its template.
        if schema is not None:
            context["schema"] = schema
        context.setdefault("output_format", format_block)
        text = _render(template, docstring, context)
        if format_block and format_heading not in text:
            text = f"{text}\n\n---\n\n{format_block}"
        return text

    if inspect.iscoroutinefunction(fn):

        async def render_call(*args, **kwargs):
            return render_context(bind_inputs(args, kwargs))

        @wraps(fn)
        async def call(*args, provider=None, session=None, **kwargs):
            if (provider is None) == (session is None):
                raise TypeError("Supply exactly one of provider= or session=")
            bound = bind_inputs(args, kwargs)
            if session is not None:
                generated = await session.arun(
                    render_context(bound), output_type=output_type, max_turns=max_turns
                )
            else:
                text, _ = await provider.acall(render_context(bound))
                generated = parse(text, str if output_type is None else output_type)
            injected = {"generated": generated} if generated_parameter else {}
            result = await fn(*bound.args, **bound.kwargs, **injected)
            return generated if result is None or result is Ellipsis else result
    else:

        def render_call(*args, **kwargs):
            return render_context(bind_inputs(args, kwargs))

        @wraps(fn)
        def call(*args, provider=None, session=None, **kwargs):
            if (provider is None) == (session is None):
                raise TypeError("Supply exactly one of provider= or session=")
            bound = bind_inputs(args, kwargs)
            if session is not None:
                generated = session.run(
                    render_context(bound), output_type=output_type, max_turns=max_turns
                )
            else:
                text, _ = provider.call(render_context(bound))
                generated = parse(text, str if output_type is None else output_type)
            injected = {"generated": generated} if generated_parameter else {}
            result = fn(*bound.args, **bound.kwargs, **injected)
            return generated if result is None or result is Ellipsis else result

    parameters = list(inputs.parameters.values())
    index = next(
        (
            i
            for i, parameter in enumerate(parameters)
            if parameter.kind == inspect.Parameter.VAR_KEYWORD
        ),
        len(parameters),
    )
    parameters[index:index] = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Any)
        for name in ("provider", "session")
    ]
    call.__signature__ = inputs.replace(parameters=parameters)
    call.__annotations__ = {
        name: annotation for name, annotation in fn.__annotations__.items() if name != "generated"
    }
    call.__annotations__["provider"] = Any
    call.__annotations__["session"] = Any
    call._prompt_output_type = output_type
    render_call.__signature__ = inputs.replace(return_annotation=str)
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

"""Jinja rendering, text execution, and typed Python results.

`Prompt(template)(**variables)` and `render(template, **variables)` only
render text. `parse(text, returns)`
only validates a result. `@prompt(provider=...)` composes rendering,
provider.call/acall, and parsing without implicit persistence or repairs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, get_type_hints, overload

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import TypeAdapter

TEMPLATE_ROOT = Path("prompts")
LOG_DIR = Path("logs/prompts")

FORMAT_HEADING = "# Output Format"

FORMAT_BLOCK = f"""\
{FORMAT_HEADING}

Respond with ONLY a JSON value matching this schema — no preamble, no \
commentary, nothing outside the JSON.

{{schema}}
"""


class PromptError(Exception):
    """A text-only prompt received tool requests."""


T = TypeVar("T")


def render(template: str, /, **variables: Any) -> str:
    """Render a file under TEMPLATE_ROOT, without model calls or output instructions."""
    return _Template(template).render(variables)


class Prompt:
    """A template file bound to a callable renderer, independent of execution.

    Files resolve under TEMPLATE_ROOT at call time, including includes and
    imports. Construction stores only the filename; each call renders afresh.
    """

    def __init__(self, template: str):
        self.template = template

    def __call__(self, /, **variables: Any) -> str:
        return render(self.template, **variables)


@overload
def parse(text: str) -> str: ...


@overload
def parse(text: str, returns: type[T]) -> T: ...


def parse(text: str, returns: Any = str) -> Any:
    """Return text unchanged, or validate JSON against a Pydantic-compatible type."""
    return text if returns is str else TypeAdapter(returns).validate_json(text)


def prompt(
    fn: Callable | None = None,
    *,
    template: str | None = None,
    provider: Any = None,
    cache: bool = False,
    log_dir: Path | str | None = None,
) -> Callable:
    """Render, call the supplied provider, and parse the annotated return type.

    Logging and caching are opt-in. Parsing failures propagate to the caller.
    .render() runs the function body in its sync/async mode without provider calls.
    """
    if fn is None:
        return lambda inner: prompt(
            inner,
            template=template,
            provider=provider,
            cache=cache,
            log_dir=log_dir,
        )
    signature = inspect.signature(fn)
    source = _Template(template, inspect.getdoc(fn))
    returns = get_type_hints(fn).get("return", str)
    adapter = None if returns is str else TypeAdapter(returns)
    format_block = (
        FORMAT_BLOCK.format(schema=json.dumps(adapter.json_schema(), indent=2))
        if adapter is not None
        else ""
    )

    def render_context(arguments, extra):
        context = dict(arguments)
        if extra is not None:
            context.update(extra)
        context.setdefault("output_format", format_block)
        text = source.render(context)
        if format_block and FORMAT_HEADING not in text:
            text = f"{text}\n\n---\n\n{format_block}"
        return text

    def exchange(text, output):
        directory = Path(log_dir) if log_dir is not None else (LOG_DIR if cache else None)
        return _Exchange(provider, text, adapter, cache, directory, fn.__name__, output)

    if inspect.iscoroutinefunction(fn):

        async def render_call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            extra = await fn(*bound.args, **bound.kwargs)
            return render_context(bound.arguments, extra)

        @wraps(fn)
        async def call(*args, output=None, **kwargs):
            text = await render_call(*args, **kwargs)
            run = exchange(text, output)
            cached = run.cached()
            if cached is not _MISSING:
                return cached
            return run.accept(_final_text(await provider.acall(text)))
    else:

        def render_call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            extra = fn(*bound.args, **bound.kwargs)
            return render_context(bound.arguments, extra)

        @wraps(fn)
        def call(*args, output=None, **kwargs):
            text = render_call(*args, **kwargs)
            run = exchange(text, output)
            cached = run.cached()
            if cached is not _MISSING:
                return cached
            return run.accept(_final_text(provider.call(text)))

    call.render = render_call
    call.source = source.read
    call.template_name = source.name
    call.returns = returns
    return call


class _Template:
    """A file under TEMPLATE_ROOT or an inline docstring, loaded on each render."""

    def __init__(self, name: str | None, docstring: str | None = None):
        self.name = name
        self.docstring = docstring

    def read(self) -> str:
        if self.name is None:
            return self.docstring
        environment = _environment()
        return environment.loader.get_source(environment, self.name)[0]

    def render(self, variables: Mapping) -> str:
        environment = _environment()
        template = (
            environment.from_string(self.docstring)
            if self.name is None
            else environment.get_template(self.name)
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


_MISSING = object()


class _Exchange:
    """Parsing and optional disk persistence shared by sync and async execution."""

    def __init__(self, provider, text, adapter, cache, directory, name, output):
        self.text = text
        self.adapter = adapter
        self.cache = cache
        self.output = Path(output) if output is not None else None
        self.directory = None
        if directory is not None:
            config = json.dumps(provider.identity(), sort_keys=True)
            digest = hashlib.sha256(f"{text}\n{config}".encode()).hexdigest()[:12]
            self.directory = directory / f"{name}-{digest}"

    def write(self, filename, text):
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / filename).write_text(text, encoding="utf-8")

    def cached(self):
        if self.cache and self.directory is not None:
            accepted = self.directory / "response.txt"
            if accepted.exists():
                return self.value(accepted.read_text(encoding="utf-8"))
        self.write("prompt.md", self.text)
        return _MISSING

    def value(self, text):
        value = text if self.adapter is None else self.adapter.validate_json(text)
        if self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_text(text, encoding="utf-8")
        return value

    def accept(self, text):
        value = self.value(text)
        self.write("response.txt", text)
        return value


def _final_text(response):
    text, requests = response
    if requests:
        raise PromptError(
            "This prompt expects final text; handle tool requests in application code."
        )
    return text

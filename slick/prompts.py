"""Jinja rendering, text execution, and typed Python results.

`Prompt(template)(**variables)` and `render(template, **variables)` only
render text. `parse(text, returns)`
only validates a result. `@prompt(provider=...)` composes rendering,
provider.call/acall, and parsing without implicit persistence or repairs.
Legacy model= declarations retain their original logging/cache defaults.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, get_type_hints, overload

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound
from jinja2 import Template as JinjaTemplate
from pydantic import TypeAdapter, ValidationError

from .providers import Provider, get_command
from .tools._protocol import validate_response

TEMPLATE_ROOT = Path("prompts")
LOG_DIR = Path("logs/prompts")

# Kwargs the wrapper claims for itself; a template variable cannot use them.
RESERVED = ("model", "output")

FORMAT_HEADING = "# Output Format"

FORMAT_BLOCK = f"""\
{FORMAT_HEADING}

Respond with ONLY a JSON value matching this schema — no preamble, no \
commentary, nothing outside the JSON.

{{schema}}
"""

REPAIR = """\
# Repair

Your previous response could not be parsed.

Previous response:
{response}

Parse error:
{error}

Respond again with ONLY the corrected JSON.
"""


class PromptError(Exception):
    """Rendering or execution contract failure, optionally with rejected output."""

    def __init__(self, message: str, *, response: str | None = None):
        super().__init__(message)
        self.response = response


F = TypeVar("F", bound=Callable)
T = TypeVar("T")


def render(template: str, /, **variables: Any) -> str:
    """Render a file under TEMPLATE_ROOT, without model calls or output instructions."""
    return _Template(template, template, None).render(variables)


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
    """Return text unchanged, or validate it against a Pydantic-compatible type.

    Invalid structured results raise pydantic.ValidationError. Fenced JSON
    and the existing scalar/prose compatibility formats are accepted.
    """
    return text if returns is str else _Parser(returns).parse(text)


@overload
def prompt(
    fn: F,
    *,
    template: str | None = None,
    model: Provider | str | None = None,
    provider: Any = None,
    max_repairs: int | None = None,
    cache: bool | None = None,
    log_dir: Path | str | None = None,
) -> F: ...


@overload
def prompt(
    fn: None = None,
    *,
    template: str | None = None,
    model: Provider | str | None = None,
    provider: Any = None,
    max_repairs: int | None = None,
    cache: bool | None = None,
    log_dir: Path | str | None = None,
) -> Callable[[F], F]: ...


def prompt(
    fn: Callable | None = None,
    *,
    template: str | None = None,
    model: Provider | str | None = None,
    provider: Any = None,
    max_repairs: int | None = None,
    cache: bool | None = None,
    log_dir: Path | str | None = None,
) -> Callable:
    """Declare a template-backed function using call (def) or acall (async def).

    provider= accepts a configured text provider. Modern calls have no logging,
    cache or repair unless explicitly requested. Legacy synchronous model=
    declarations keep their original defaults. .render() follows the declared
    sync/async mode and never calls the provider; a computed-context body runs.
    """
    if provider is not None and model is not None:
        raise PromptError("Choose either model= or provider=, not both.")
    if max_repairs is not None and (type(max_repairs) is not int or max_repairs < 0):
        raise PromptError("max_repairs must be a non-negative integer.")
    if cache is not None and not isinstance(cache, bool):
        raise PromptError("cache must be a boolean.")
    if fn is None:
        return lambda inner: _build(inner, template, model, provider, max_repairs, cache, log_dir)
    return _build(fn, template, model, provider, max_repairs, cache, log_dir)


def _build(fn, template, model, provider, max_repairs, cache, log_dir):
    signature = inspect.signature(fn)
    clash = [name for name in RESERVED if name in signature.parameters]
    if clash:
        raise PromptError(
            f"{fn.__name__} declares {', '.join(clash)}; @prompt reserves "
            f"{' and '.join(RESERVED)}. Rename the parameter."
        )
    if template is None and not fn.__doc__:
        raise PromptError(
            f"{fn.__name__} has no template: name one with template=, or write the "
            f"template as the docstring."
        )
    source = _Template(
        fn.__name__,
        template,
        None if template is not None else inspect.cleandoc(fn.__doc__),
    )
    returns = get_type_hints(fn).get("return", str)
    parser = None if returns is str else _Parser(returns)
    asynchronous = inspect.iscoroutinefunction(fn)
    modern = provider is not None or asynchronous
    use_cache = not modern if cache is None else cache
    repairs = (0 if modern else 1) if max_repairs is None else max_repairs

    def prepare(kwargs):
        override = kwargs.pop("model", None)
        if provider is not None and override is not None:
            raise PromptError("A provider= declaration does not accept a model= override.")
        output = kwargs.pop("output", None)
        if output is not None and parser is not None:
            raise PromptError(
                f"{fn.__name__} returns {_name(returns)}, not str; "
                "output= only saves text responses."
            )
        selected = provider if provider is not None else _resolve(override or model)
        method = "acall" if asynchronous else "call"
        execute = getattr(selected, method, None)
        if not callable(execute):
            raise PromptError(f"{fn.__name__}: provider must implement {method}(text).")
        directory = (
            Path(log_dir) if log_dir is not None else (LOG_DIR if not modern or use_cache else None)
        )
        return selected, execute, directory, None if output is None else Path(output)

    def exchange(selected, text, directory, output):
        return _Exchange(
            selected,
            text,
            parser,
            repairs,
            use_cache,
            directory,
            fn.__name__,
            output,
            announce=not modern,
        )

    if asynchronous:

        async def arender_call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            extra = await fn(*bound.args, **bound.kwargs)
            return _render_context(fn, source, parser, bound.arguments, extra)

        @wraps(fn)
        async def call(*args, **kwargs):
            selected, execute, directory, output = prepare(kwargs)
            text = await arender_call(*args, **kwargs)
            run = exchange(selected, text, directory, output)
            cached = run.cached()
            if cached is not _MISSING:
                return cached
            for attempt in range(repairs + 1):
                response = _final_text(await execute(text))
                try:
                    return run.accept(response)
                except ValidationError as exc:
                    text = run.repair(response, exc, attempt)
    else:

        def render_call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            extra = fn(*bound.args, **bound.kwargs)
            return _render_context(fn, source, parser, bound.arguments, extra)

        @wraps(fn)
        def call(*args, **kwargs):
            selected, execute, directory, output = prepare(kwargs)
            text = render_call(*args, **kwargs)
            run = exchange(selected, text, directory, output)
            cached = run.cached()
            if cached is not _MISSING:
                return cached
            for attempt in range(repairs + 1):
                response = _final_text(execute(text))
                try:
                    return run.accept(response)
                except ValidationError as exc:
                    text = run.repair(response, exc, attempt)

    call.render = arender_call if asynchronous else render_call
    call.source = source.read
    call.template_name = source.name
    call.returns = returns
    return call


class _Template:
    """Where a prompt's Jinja source comes from: a file under TEMPLATE_ROOT,
    or the function's own docstring. Files are loaded per use, so an edit
    shows up on the next call."""

    def __init__(self, owner: str, name: str | None, docstring: str | None) -> None:
        self.owner = owner
        self.name = name
        self.docstring = docstring

    def read(self) -> str:
        """The template source."""
        if self.name is None:
            assert self.docstring is not None
            return self.docstring
        environment = _environment()
        try:
            assert environment.loader is not None
            return environment.loader.get_source(environment, self.name)[0]
        except TemplateNotFound as exc:
            raise self._missing() from exc

    def compile(self) -> JinjaTemplate:
        environment = _environment()
        if self.name is None:
            assert self.docstring is not None
            return environment.from_string(self.docstring)
        try:
            return environment.get_template(self.name)
        except TemplateNotFound as exc:
            raise self._missing() from exc

    def render(self, variables: Mapping) -> str:
        return self.compile().render(**variables).strip()

    def _missing(self) -> PromptError:
        return PromptError(
            f"{self.owner}: no template {self.name!r} under {TEMPLATE_ROOT}/ "
            f"(point slick.prompts.TEMPLATE_ROOT elsewhere to change that)."
        )


def _render_context(fn, source, parser, arguments, extra) -> str:
    context = dict(arguments)
    if extra is not None and extra is not Ellipsis:
        if not isinstance(extra, Mapping):
            raise PromptError(
                f"{fn.__name__} returned {type(extra).__name__}; a prompt body must return a "
                "mapping of extra template variables, or nothing at all."
            )
        context.update(extra)
    context.setdefault("output_format", "" if parser is None else parser.format_block)
    text = source.render(context)
    if parser is not None and FORMAT_HEADING not in text:
        text = f"{text}\n\n---\n\n{parser.format_block}"
    return text


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

    def __init__(self, provider, text, parser, repairs, cache, directory, name, output, announce):
        self.text = text
        self.parser = parser
        self.repairs = repairs
        self.cache = cache
        self.name = name
        self.output = output
        self.announce = announce
        self.directory = (
            directory / f"{name}-{_digest(text, provider)}" if directory is not None else None
        )

    def write(self, filename, text):
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / filename).write_text(text, encoding="utf-8")

    def cached(self):
        if self.cache and self.directory is not None:
            accepted = self.directory / "response.txt"
            if accepted.exists():
                if self.announce:
                    print(f"{self.name}: cached ({self.directory})", file=sys.stderr)
                text = accepted.read_text(encoding="utf-8")
                return self.value(text)
        self.write("prompt.md", self.text)
        return _MISSING

    def value(self, text):
        if not isinstance(text, str):
            raise PromptError(f"{self.name}: provider must return str, got {type(text).__name__}.")
        if self.parser is not None:
            return parse(text, self.parser.annotation)
        return _save(text, self.output, self.name)

    def accept(self, text):
        value = self.value(text)
        self.write("response.txt", text)
        return value

    def repair(self, text, error, attempt):
        self.write(f"rejected.{attempt + 1}.txt", text)
        if attempt >= self.repairs:
            detail = f" See {self.directory}." if self.directory is not None else ""
            raise PromptError(
                f"{self.name}: no {_name(self.parser.annotation)} could be parsed out of "
                f"the response after {self.repairs} repair(s).{detail}",
                response=text,
            ) from error
        repair = f"{self.text}\n\n---\n\n{REPAIR.format(response=text, error=error)}"
        self.write(f"repair.{attempt + 1}.md", repair)
        return repair


def _save(text: str, output: Path | None, name: str) -> str:
    """Return the response text, saving it to `output` when asked."""
    if output is None:
        return text
    if not text.strip():
        raise PromptError(f"{name} returned empty output; nothing written to {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


class _Parser:
    """Structured output for any return type pydantic can validate."""

    def __init__(self, annotation: Any) -> None:
        self.annotation = annotation
        self.adapter = TypeAdapter(annotation)
        schema = json.dumps(self.adapter.json_schema(), indent=2)
        self.format_block = FORMAT_BLOCK.format(schema=schema)

    def parse(self, text: str) -> Any:
        try:
            return self.adapter.validate_json(_json_slice(text))
        except ValidationError as exc:
            try:
                # A bare scalar answer (approved, 42, true) is right but is not JSON.
                return self.adapter.validate_python(text.strip())
            except ValidationError:
                raise exc from None


def _json_slice(text: str) -> str:
    """The JSON payload in a response: a fenced block if there is one, else
    the outermost braces or brackets, else the whole response."""
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not starts:
        return text.strip()
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start : end + 1] if end > start else text.strip()


def _resolve(model: Provider | str | None) -> Provider:
    if model is None:
        return get_command()
    return get_command(model) if isinstance(model, str) else model


def _digest(prompt_text: str, model) -> str:
    """Persistent calls need a stable identity; ordinary calls need only call/acall."""
    identity = getattr(model, "identity", None)
    if callable(identity):
        try:
            config = json.dumps(identity(), sort_keys=True, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise PromptError(
                "provider.identity() must return JSON-serializable configuration."
            ) from exc
        fingerprint = f"{prompt_text}\n{config}"
    elif hasattr(model, "provider") and hasattr(model, "model"):
        fingerprint = "\n".join([prompt_text, model.provider, model.model or ""])
    else:
        raise PromptError(
            "Logging/cache requires a provider identity() or legacy provider/model metadata."
        )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]


def _name(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation))


def _final_text(response):
    try:
        text, requests = validate_response(response)
    except ValueError as exc:
        raise PromptError(str(exc)) from exc
    if requests:
        raise PromptError(
            "This prompt expects final text; handle tool requests in application code."
        )
    return text

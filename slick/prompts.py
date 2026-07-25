"""Prompt-as-function: a Jinja template with a Python signature.

Keep the prompt in a template file under TEMPLATE_ROOT and name it on the
decorator. The function supplies the signature; the file supplies the words:

    @prompt(template="summarize.md.j2")
    def summarize(document: str, audience: str = "an engineer") -> Summary:
        '''Summarize a document for one audience.'''

    # prompts/summarize.md.j2
    #     Summarize the document below for {{ audience }}.
    #     {% include "house_style.md" %}
    #
    #     # Document
    #     {{ document }}
    #
    #     {{ output_format }}

For a prompt small enough to read beside the code, omit `template=` and the
docstring becomes the template instead:

    @prompt(model="claude")
    def headline(document: str) -> str:
        '''
        Write one headline for the document below.

        # Document
        {{ document }}
        '''

    summary = summarize(text)               # -> Summary, validated
    headline(text, output="headline.txt")   # -> str, also saved to file
    print(summarize.render(text))           # the prompt, no model call
    print(summarize.source())               # the template source

The decorator reads four things and nothing else: parameters are the
template variables, the template is the named file (or the docstring), the
return annotation is the output contract, and the body — if it returns a
mapping — adds computed variables to the render context.

A return annotation other than `str` turns on structured output: the
type's JSON schema is rendered wherever the template says
`{{ output_format }}` (appended if the template never asks for it), the
response is parsed against that type, and a validation failure is handed
back to the model once as a repair. CLI-backed models have no
constrained decoding, so parse-and-repair is the only enforcement we get.

Every call writes its rendered prompt and accepted response under
LOG_DIR, keyed by content hash. That doubles as a cache: an unchanged
prompt is never paid for twice.

Two module-level settings, both read at call time so an application can
assign to them at startup: LOG_DIR (where runs are logged) and
TEMPLATE_ROOT (where template files and {% include %} resolve). Templates
are loaded per call, so editing one takes effect without a restart.
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
from typing import Any, get_type_hints

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound
from jinja2 import Template as JinjaTemplate
from pydantic import TypeAdapter, ValidationError

from .models import Model, get_model

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
    pass


def prompt(
    fn: Callable | None = None,
    *,
    template: str | None = None,
    model: Model | str | None = None,
    max_repairs: int = 1,
    cache: bool = True,
    log_dir: Path | str | None = None,
) -> Callable:
    """Turn a function into a model call. Usable bare or with arguments.

    template:    template file under TEMPLATE_ROOT; the docstring is used
                 as the template when this is omitted.
    model:       a Model, a backend name, or None to resolve the default lazily.
    max_repairs: retries granted to a response that will not parse.
    cache:       reuse a logged response for an identical prompt.
    """
    if fn is None:
        return lambda inner: _build(inner, template, model, max_repairs, cache, log_dir)
    return _build(fn, template, model, max_repairs, cache, log_dir)


def _build(
    fn: Callable,
    template: str | None,
    model: Model | str | None,
    max_repairs: int,
    cache: bool,
    log_dir: Path | str | None,
) -> Callable:
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

    # With a template file the docstring is free to be documentation.
    source = _Template(
        fn.__name__,
        template,
        None if template is not None else inspect.cleandoc(fn.__doc__),
    )
    returns = get_type_hints(fn).get("return", str)
    parser = None if returns is str else _Parser(returns)
    directory = Path(log_dir) if log_dir is not None else None

    @wraps(fn)
    def call(*args, **kwargs) -> Any:
        override = kwargs.pop("model", None)
        output = kwargs.pop("output", None)
        if output is not None and parser is not None:
            raise PromptError(
                f"{fn.__name__} returns {_name(returns)}, not str; "
                f"output= only saves text responses."
            )
        return _run(
            _resolve(override or model),
            _render(fn, signature, source, parser, args, kwargs),
            parser=parser,
            max_repairs=max_repairs,
            cache=cache,
            log_dir=directory if directory is not None else LOG_DIR,
            name=fn.__name__,
            output=None if output is None else Path(output),
        )

    call.render = lambda *args, **kwargs: _render(fn, signature, source, parser, args, kwargs)
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
            return self.docstring
        environment = _environment()
        try:
            return environment.loader.get_source(environment, self.name)[0]
        except TemplateNotFound as exc:
            raise self._missing() from exc

    def compile(self) -> JinjaTemplate:
        environment = _environment()
        if self.name is None:
            return environment.from_string(self.docstring)
        try:
            return environment.get_template(self.name)
        except TemplateNotFound as exc:
            raise self._missing() from exc

    def _missing(self) -> PromptError:
        return PromptError(
            f"{self.owner}: no template {self.name!r} under {TEMPLATE_ROOT}/ "
            f"(point slick.prompts.TEMPLATE_ROOT elsewhere to change that)."
        )


def _render(
    fn: Callable,
    signature: inspect.Signature,
    source: _Template,
    parser: _Parser | None,
    args: tuple,
    kwargs: dict,
) -> str:
    """Bind arguments, run the body for computed variables, render the template."""
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    context = dict(bound.arguments)

    extra = fn(*bound.args, **bound.kwargs)
    if extra is not None and extra is not Ellipsis:  # `...` is a valid empty body
        if not isinstance(extra, Mapping):
            raise PromptError(
                f"{fn.__name__} returned {type(extra).__name__}; a prompt body must return a "
                f"mapping of extra template variables, or nothing at all."
            )
        context.update(extra)

    context.setdefault("output_format", "" if parser is None else parser.format_block)
    text = source.compile().render(**context).strip()

    # The template never asked for the schema, but the contract still has to
    # reach the model or the response cannot be parsed.
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


def _run(
    model: Model,
    prompt_text: str,
    *,
    parser: _Parser | None,
    max_repairs: int,
    cache: bool,
    log_dir: Path,
    name: str,
    output: Path | None,
) -> Any:
    """Call the model, parse it, log everything, repair once if it will not parse."""
    directory = log_dir / f"{name}-{_digest(prompt_text, model)}"
    accepted = directory / "response.txt"

    if cache and accepted.exists():
        # It parsed once, and the schema is part of the prompt that keyed this
        # directory, so it will parse again. Say so: a silent cache hit looks
        # like a model that ignored your edit.
        print(f"{name}: cached ({directory})", file=sys.stderr)
        text = accepted.read_text(encoding="utf-8")
        return parser.parse(text) if parser is not None else _save(text, output, name)

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompt.md").write_text(prompt_text, encoding="utf-8")
    text = model.call(prompt_text)

    if parser is None:
        accepted.write_text(text, encoding="utf-8")
        return _save(text, output, name)

    # Only a response that parses is worth keeping; a poisoned cache would
    # replay the same failure and spend the repair budget again.
    for attempt in range(1, max_repairs + 2):
        try:
            value = parser.parse(text)
        except ValidationError as exc:
            (directory / f"rejected.{attempt}.txt").write_text(text, encoding="utf-8")
            if attempt > max_repairs:
                raise PromptError(
                    f"{name}: no {_name(parser.annotation)} could be parsed out of the response "
                    f"after {max_repairs} repair(s). See {directory}."
                ) from exc
            repair = f"{prompt_text}\n\n---\n\n{REPAIR.format(response=text, error=exc)}"
            (directory / f"repair.{attempt}.md").write_text(repair, encoding="utf-8")
            text = model.call(repair)
        else:
            accepted.write_text(text, encoding="utf-8")
            return value


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


def _resolve(model: Model | str | None) -> Model:
    if model is None:
        return get_model()
    return get_model(model) if isinstance(model, str) else model


def _digest(prompt_text: str, model: Model) -> str:
    """Content address for one call. The schema rides along inside the prompt."""
    fingerprint = "\n".join([prompt_text, model.backend, model.model or ""])
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]


def _name(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation))

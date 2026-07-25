# slick

Prompts as typed Python functions.

A function's parameters are the template variables, a Jinja template file is the
prompt, and the return annotation is the output contract. Nothing else is
inferred, and the prompt is never rewritten behind your back.

```python
# app.py
from pydantic import BaseModel
from slick import prompt

class Summary(BaseModel):
    headline: str
    key_points: list[str]

@prompt(template="summarize.md.j2")
def summarize(document: str, audience: str = "an engineer") -> Summary:
    """Summarize a document for one audience."""
```

```jinja
{# prompts/summarize.md.j2 #}
Summarize the document below for {{ audience }}.
{% include "shared/house_style.md" %}

# Document
{{ document }}

{{ output_format }}
```

```python
result = summarize(text)          # -> Summary, validated
result.headline
```

## Installation

```bash
pip install slick-ai
```

Models are CLI-backed: calls shell out to `claude -p` or `codex exec`, so they
run on your existing subscription rather than metered API tokens. Install
whichever CLI you use and slick will drive it.

## Where the prompt lives

Prompt files are resolved against `slick.prompts.TEMPLATE_ROOT`, which defaults
to `prompts/`. Subdirectories, `{% include %}`, `{% extends %}`, and `{% import %}`
all work relative to that root, so shared preambles and house style live in one
place. Templates are loaded per call — edit one and the next call picks it up
without a restart.

For a prompt short enough to read beside the code, omit `template=` and the
docstring becomes the template:

```python
@prompt(model="claude")
def headline(document: str) -> str:
    """
    Write one headline for the document below.

    # Document
    {{ document }}
    """
```

## The four things the decorator reads

| Python | Role |
|---|---|
| parameters | typed template variables |
| `template=`, else the docstring | the Jinja template, verbatim |
| return annotation | output contract |
| body | `...`, or `return {...}` to add computed variables |

```python
@prompt(template="summarize.md.j2")
def summarize(document: str) -> Summary:
    """Summarize a document."""
    return {"words": len(document.split())}   # optional: extra template variables
```

Undefined variables raise rather than rendering blank, so a typo can't silently
drop your context.

Four attributes hang off a decorated function:

```python
summarize.render(text)      # the exact prompt, no model call — useful in tests
summarize.source()          # the template source
summarize.template_name     # "summarize.md.j2", or None for a docstring template
summarize.returns           # the return annotation
```

## Return types

- **`str`** — the response text, untouched. No schema is injected.
- **anything Pydantic can validate** — a `BaseModel`, `list[Model]`, `Literal`,
  `bool`, `int`, a dataclass: the JSON schema is rendered wherever the template
  says `{{ output_format }}` (appended if it never does), and the response is
  parsed and validated against it.

CLI-backed models have no constrained decoding, so parsing is text-level:
fenced blocks and surrounding prose are tolerated, and a bare scalar answer
(`approve`) is coerced. A `ValidationError` is handed back to the model once,
with the bad response and the error, as a repair. Tune with `max_repairs=`.

Passing `output=` saves a text response to a file as well as returning it; an
empty response raises rather than writing an empty file.

```python
headline(text, output="headline.txt")
headline(text, model="claude")     # per-call backend override
```

## Runs are logged, and the log is the cache

Every call writes its rendered prompt and accepted response under
`slick.prompts.LOG_DIR` (default `logs/prompts/`), in a directory keyed by the
hash of the prompt, backend, and model:

```
logs/prompts/summarize-843f0ce2d996/
  prompt.md        # exactly what was sent
  response.txt     # what came back and parsed
  rejected.1.txt   # if a response failed to parse
  repair.1.md      # the repair prompt it was sent
```

An identical prompt is served from that directory instead of being paid for
again, and says so on stderr. Only responses that parsed are cached, so a
failure can't replay itself. Disable with `cache=False`.

## Models

```python
from slick import get_model, set_default

set_default(backend="claude", model="claude-opus-4-8")
get_model().call("one prompt in, one string out")
```

Resolution order: the explicit argument, then `set_default()`, then
`$SLICK_BACKEND` / `$SLICK_MODEL`, then the built-in default (`codex`).

A `Model` is one method — `call(prompt: str) -> str`. Anything with that method
plus `backend` and `model` attributes works as a stand-in, which is how tests
avoid launching a CLI. `Model.execute()` is the lower-level engine and takes a
workdir and sandbox, for long-running workspace tasks.

## CLI

```bash
slick model                        # what a bare call resolves to
slick call "summarize this"        # prompt as an argument
cat document.md | slick call       # or on stdin
slick call --backend claude --output out.md "..."
```

## Development

```bash
poetry install --with dev
poetry run pytest
poetry run ruff check .
```

## Releasing

Bump `version` in `pyproject.toml`, then:

```bash
poetry build
poetry publish --repository testpypi   # dry run
poetry publish
```

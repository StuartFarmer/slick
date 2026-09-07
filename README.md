# slick

Jinja templates to LLMs and back, as ordinary Python functions.

```python
from slick import Prompt
from slick.backends import OpenAI

model = OpenAI(model="YOUR_MODEL_ID")
answer_prompt = Prompt("answer.j2")
text = answer_prompt(question="How does authentication work?", documents=documents)
answer = await model.acall(text)
```

Prompts render arguments into text. Backends execute text and return responses.
Your Python code connects them and owns retrieval, history, sequencing, and
concurrency. Jinja handles context composition; `parse` validates results when needed.

## Install

```bash
pip install slick-ai                  # Jinja, Pydantic, and CLI backends
pip install 'slick-ai[openai]'         # optional OpenAI SDK
pip install 'slick-ai[anthropic]'      # optional Anthropic SDK
pip install 'slick-ai[api]'            # both API SDKs
```

API backends use `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` when called. Model IDs
are explicit application choices. Importing Slick or constructing an API backend
does not load its SDK, require credentials, or make a request.

## Render, execute, parse

Templates live under `slick.prompts.TEMPLATE_ROOT`, which defaults to `prompts/`.

```jinja
{# prompts/answer.j2 #}
{% include "shared/instructions.j2" %}

{% for document in documents %}
<document>{{ document }}</document>
{% endfor %}

Question: {{ question }}
```

```python
from slick import render, parse

text = render("answer.j2", question=question, documents=documents)
response = model.call(text)           # synchronous execution
response = await model.acall(text)    # asynchronous execution (a separate call)

value = parse(response)               # text unchanged
numbers = parse("[1, 2, 3]", list[int])
```

`render` performs no model calls or output-format injection. It shares the
same Jinja environment as decorated functions: includes, imports, macros and
inheritance resolve against `TEMPLATE_ROOT`, missing variables raise errors,
and template edits take effect on the next render. Leading/trailing whitespace
is stripped from the rendered prompt. Backend response text is returned unchanged.

`parse` accepts Pydantic-compatible types and raises `pydantic.ValidationError`
for invalid structured output. It retains support for fenced JSON, prose around
JSON, and compatible bare scalar values. It never calls a backend or repairs output.

## Reusable prompts and application classes

`Prompt("answer.j2")` stores only a template filename. Calling it is equivalent
to `render("answer.j2", **variables)`: it returns text synchronously and never
executes a backend. Files are loaded at call time, so edits and changes to
`TEMPLATE_ROOT` affect existing Prompt objects too. Template arguments such as
`backend`, `model`, and `output` are ordinary data, not execution settings.

Keep state and arbitrary operations in your own classes:

```python
class QuestionAnswerer:
    def __init__(self, backend):
        self.backend = backend
        self.history = []
        self.answer_prompt = Prompt("conversation.j2")
        self.critique_prompt = Prompt("critique.j2")

    async def ask(self, question):
        text = self.answer_prompt(question=question, messages=self.history)
        answer = await self.backend.acall(text)
        self.history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        return answer

    async def critique(self, answer):
        return await self.backend.acall(self.critique_prompt(answer=answer))
```

The class owns the backend and history. Its methods combine ordinary Python and
LLM calls. This example assumes sequential calls on each instance; a failed
exchange is not appended, and critique does not modify history. The templates
are ordinary files; see the [runnable example](examples/question_answerer.py).

## Existing executing decorator

The lowercase `@prompt` decorator remains available with its existing behavior:
it combines rendering, execution and parsing. Use uppercase `Prompt` for a
backend-independent renderer.

```python
from pydantic import BaseModel
from slick import prompt

class Summary(BaseModel):
    headline: str
    points: list[str]

@prompt(backend=model, template="summarize.j2")
async def summarize(document: str, audience: str = "an engineer") -> Summary:
    """Summarize a document for one audience."""

summary = await summarize(document)
print(summary.headline)
```

```jinja
{# prompts/summarize.j2 #}
Summarize this document for {{ audience }}:
{{ document }}

{{ output_format }}
```

Parameters and defaults supply template variables. The return annotation supplies
the output contract. `str` requests plain text; other types add JSON instructions
through `{{ output_format }}` (appended if omitted) and are validated locally.
This release uses prompt instructions and local validation, not provider-native
constrained decoding. Invalid results raise `PromptError`; its `.response` holds
the rejected text and its cause holds the validation error.

A `def` declaration uses `backend.call`; an `async def` declaration uses
`backend.acall`. Unsupported modes fail clearly; Slick never runs a blocking
backend in a hidden thread or starts an event loop for you.

```python
text = await summarize.render(document)  # render without backend execution
source = summarize.source()             # read the template source
name = summarize.template_name
result_type = summarize.returns
```

For sync declarations, `.render(...)` is synchronous. A named template leaves
the docstring free for documentation. Omit `template=` to use the docstring as
the Jinja template. Computed-context bodies returning a mapping remain supported;
an async body is awaited, including during `.render()`. Prefer ordinary Python
functions around `render` and `call`/`acall` when you want explicit preparation
or postprocessing.

## Backends

```python
from slick.backends import OpenAI, Anthropic
from slick import get_model

reader = OpenAI(model="YOUR_OPENAI_MODEL_ID", timeout=60, max_output_tokens=2048)
writer = Anthropic(model="YOUR_ANTHROPIC_MODEL_ID")
coder = get_model("codex")
claude = get_model("claude")
```

API adapters use [OpenAI Responses](https://developers.openai.com/api/docs/libraries)
and [Anthropic Messages](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python).
Both expose `call(text)` and `acall(text)`. Refused, incomplete, or unexpected
tool outputs raise `BackendError` (an alias of `ModelError`); native transport
exceptions remain accessible through the chained cause. No tools are configured
by these adapters.

SDK transport retries default to zero; set `max_retries=` explicitly to enable
them. Owned SDK clients are created and closed per call. For connection reuse,
pass `client=` and/or `async_client=` from the provider's SDK; the application
owns and closes them. Slick applies its timeout/retry settings using the SDK's
`with_options` method. Backend objects retain configuration, not conversation history.

CLI models retain their commands and permissions. `acall` uses native async
subprocesses. Timeout or cancellation kills and reaps the owned process; POSIX
cleanup also targets its process group. Processes that escape that group and
Windows descendants require application-level management. The lower-level
`execute`/`aexecute` methods accept `workdir` and `sandbox` and return an
`ExecutionResult`. CLI arguments and permission semantics remain CLI-specific.

A custom backend only needs `call(text) -> str`, `acall(text) -> str`, or both.
No inheritance, registration or metadata is required for ordinary calls.

## Explicit execution options and compatibility

`@prompt(backend=...)` and new async declarations default to **no disk logging,
no caching, and no automatic output repair**. Backend selection is fixed on the
decorator; passing both `backend=` and `model=` or overriding a modern declaration
with per-call `model=` raises an error.

Explicit options remain available:

```python
@prompt(backend=model, template="summarize.j2", max_repairs=1)
def summarize(document: str) -> Summary:
    """Summarize a document, allowing one formatting repair."""
```

`log_dir=` opts into prompt/response files. `cache=True` opts into reuse of accepted
responses (under `log_dir` or `LOG_DIR`). `max_repairs=` permits additional backend
calls after validation failure; it is separate from SDK transport retries.
`output=` on a text-returning function explicitly saves its response to a file.
Caching is an application decision, especially for harnesses or external state.

Persistent calls need a stable, JSON-serializable backend `identity()` excluding
secrets, or legacy `backend`/`model` metadata. Built-in API adapters provide one.
Injected clients with different endpoints or external state need separate cache
directories or an application-defined identity; Slick does not inspect credentials
or infer those differences. Accepted responses are written only after validation.

Existing **synchronous** bare `@prompt` and `@prompt(model=...)` declarations keep
their original defaults: disk logging, caching, one repair, per-call `model=`
overrides, and default model resolution. `get_model`, `set_default`, environment
settings, and the CLI continue working.

## Examples

Run independent examples without credentials or model calls:

```bash
python -m examples.question_answerer
python -m examples.primitives
python -m examples.self_refine --rounds 2
python -m examples.react
python -m examples.tree_of_thoughts
```

The [examples guide](examples/README.md) covers all nine prompting patterns,
their application classes, shared Jinja macros, and CLI options. All pattern code
and templates stay under `examples/`. Pass `--backend openai --model YOUR_MODEL_ID`
(or another supported backend) for real calls.

The original functional examples also remain available:

```bash
python -m examples.core
```

[examples/core.py](examples/core.py) shows a typed summary, a conversation supplied
as a list of messages, and application-owned document retrieval. Its templates
use shared Jinja includes. To use a real backend, set `TEMPLATE_ROOT` to
`examples/prompts` and pass that backend to `examples.core.run`.

## CLI and development

```bash
slick model
slick call "summarize this"
cat document.md | slick call --backend codex

poetry install --with dev
poetry run pytest
poetry run ruff check slick tests examples
```

Ordinary tests use fake SDK clients and local Python subprocesses. Optional SDK
transport tests run when API extras are installed; no tests call paid endpoints.

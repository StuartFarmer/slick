# slick

Jinja templates to LLMs and back, as ordinary Python functions.

```python
from slick import Prompt
from slick.providers import OpenAIAPI

provider = OpenAIAPI(model="YOUR_MODEL_ID")
answer_prompt = Prompt("answer.j2")
text = answer_prompt(question="How does authentication work?", documents=documents)
answer, _ = await provider.acall(text)
```

Prompts render arguments into text. Providers execute text and return responses.
Your Python code connects them and owns retrieval, history, sequencing, and
concurrency. Jinja handles context composition; `parse` validates results when needed.

## Install

```bash
pip install slick-ai                  # base package with CLI providers
pip install 'slick-ai[openai]'         # SDK used by OpenAIAPI and OpenRouterAPI
pip install 'slick-ai[anthropic]'      # optional Anthropic SDK
pip install 'slick-ai[api]'            # both API SDKs
pip install 'slick-ai[litellm]'        # optional multi-provider SDK (no proxy server)
```

API providers use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` when called. Model IDs
are explicit application choices. Importing Slick or constructing an API provider
does not load its SDK, require credentials, or make a request.

The LiteLLM extra supports Python 3.10–3.14 and LiteLLM 1.100.x. Its SDK dependencies
are optional; the base install does not include provider SDKs.

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
response, _ = provider.call(text)           # synchronous execution
response, _ = await provider.acall(text)    # asynchronous execution (a separate call)

value = parse(response)               # text unchanged
numbers = parse("[1, 2, 3]", list[int])
```

`render` performs no model calls or output-format injection. It shares the
same Jinja environment as decorated functions: includes, imports, macros and
inheritance resolve against `TEMPLATE_ROOT`, missing variables raise errors,
and template edits take effect on the next render. Leading/trailing whitespace
is stripped from the rendered prompt. Provider response text is returned unchanged.

`parse` accepts Pydantic-compatible types and raises `pydantic.ValidationError`
for invalid structured output. It retains support for fenced JSON, prose around
JSON, and compatible bare scalar values. It never calls a provider or repairs output.

## Reusable prompts and application classes

`Prompt("answer.j2")` stores only a template filename. Calling it is equivalent
to `render("answer.j2", **variables)`: it returns text synchronously and never
executes a provider. Files are loaded at call time, so edits and changes to
`TEMPLATE_ROOT` affect existing Prompt objects too. Template arguments such as
`provider`, `model`, and `output` are ordinary data, not execution settings.

Keep state and arbitrary operations in your own classes:

```python
class QuestionAnswerer:
    def __init__(self, provider):
        self.provider = provider
        self.history = []
        self.answer_prompt = Prompt("conversation.j2")
        self.critique_prompt = Prompt("critique.j2")

    async def ask(self, question):
        text = self.answer_prompt(question=question, messages=self.history)
        answer, _ = await self.provider.acall(text)
        self.history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        return answer

    async def critique(self, answer):
        text, _ = await self.provider.acall(self.critique_prompt(answer=answer))
        return text
```

The class owns the provider and history. Its methods combine ordinary Python and
LLM calls. This example assumes sequential calls on each instance; a failed
exchange is not appended, and critique does not modify history. The templates
are ordinary files; see the [runnable example](examples/question_answerer.py).

## Python functions as tools

Prepare ordinary functions or bound methods using their annotations and docstrings:

```python
from slick.tools import prepare_tools

class Documents:
    def __init__(self, records):
        self.records = records

    def read(self, key: str) -> str:
        """Retrieve a document by its key."""
        return self.records[key]

documents = Documents({"intro": "Slick renders Jinja templates."})
tools = prepare_tools([documents.read])

schema = tools["read"].parameters       # JSON Schema for {"key": ...}; no self
text = tools["read"].invoke({"key": "intro"})
text = await tools["read"].ainvoke({"key": "intro"})
```

Preparation validates the entire list without executing the functions. Names must
be unique, and each tool needs a description and typed input parameters. Use
`Tool(documents.read, name="read_document", description="...")` from `slick` for
explicit overrides. Direct Python calls such as `documents.read("intro")` are
unchanged. The selected bound methods retain their instance and its state.

Tool arguments are JSON dictionaries. Slick rejects extra arguments and invalid
types before execution, constructs declared nested Pydantic models, and preserves
Python defaults when arguments are omitted. Nullable parameters without defaults
are still required. Strings, integers, finite floats, booleans, null, typed lists,
string-keyed dictionaries, literals, enums, Pydantic models, and TypedDicts are
supported. `Annotated[T, Field(...)]` supplies descriptions and constraints.

Declared return types are validated. Strings are returned unchanged; other
supported results become JSON text. Without a return annotation, the result must
already be text or JSON-compatible data. Arbitrary objects are never silently
converted with `str()`. Binary/date/path types, tuples, sets, untyped inputs, and
other deferred types are listed in the [tool contract](docs/superpowers/specs/2026-09-07-callable-tools.md).

`ToolError` exposes `.name` and `.phase` (`definition`, `arguments`, `execution`,
or `result`) and preserves the underlying exception as its cause. Functions are
never retried automatically. A result error occurs after the function has run;
it does not undo its effects. `ainvoke` awaits async functions and runs sync
functions inline. Wrap blocking work explicitly with `asyncio.to_thread` when
needed. Cancellation propagates normally.

API providers accept these functions through `tools=` and convert their schemas
into the native API format. `Session` coordinates execution and result submission.

## Automatic sessions

`Session` records interactions and tracks tool work automatically. Register normal
functions once, supply context, then decide when to execute the requested tools:

```python
from slick import Session

async def search(query: str) -> str:
    """Search documents."""
    return await remote_worker.search(query)

session = Session(provider=model, tools=[search])

text, requests = await session.acall(context)
results = await session.resolve_pending()
text, requests = await session.acall(next_context)
```

The second call automatically includes completed results. No response recording
or result submission is required. `resolve_pending()` executes outstanding work
sequentially, returns the results it produced, and returns `[]` when nothing is
pending. Individual execution is also available:

```python
for request in requests:
    result = await session.resolve(request)
```

`acall()` performs one model call and never executes tools. The application owns
its loop, stopping conditions and context selection. Session sends exactly the
context supplied; it never automatically replays, summarizes or renders history.
Use Jinja to render whichever observations your application needs. Recorded
`exchange["context"]` is the full prompt already sent, so don't recursively insert
those prompts into future ones.

| Property | Contents |
| --- | --- |
| `history` | Detached exchange dictionaries containing `context`, `text`, and `work` |
| `pending_requests` | Requests that have not executed |
| `ready_results` | Completed results awaiting the next successful model call |
| `tools` | Registered Tool wrappers in a fresh list |

Each work record contains its canonical `request`, its `result` (or `None`), and
a `submitted` flag. The exchange scopes request IDs; different exchanges may
reuse them. `resolve(request)` matches the full request value in the current
exchange and returns a cached result if that work is already complete. Use the
current response or `pending_requests`; identical request values reissued in a
later exchange denote new work.

Tool failures become error results. Cancellation records an interrupted result
with a possible-effects warning, then propagates; later work stays pending.
`session.cancel_pending("run stopped")` marks unstarted work as stopped without
executing it. Resolve or cancel pending requests before the next Session call.
Failed provider calls leave ready results intact. No tool or model call is
retried automatically.

Switch providers at a call boundary:

```python
text, requests = await session.acall(next_context, provider=other_model)
```

An override applies to that call only. Canonical results are converted by the
selected provider; its tool/input capability limits still apply. Use a raw
provider call or a separate Session for summarization so it doesn't consume the
main Session's ready results.

Snapshots contain data only and can be stored using ordinary JSON:

```python
import json

payload = json.dumps(session.to_dict())
restored = Session.from_dict(
    json.loads(payload), provider=other_model, tools=[search],
)
await restored.resolve_pending()  # Executes only work without a recorded result.
text, requests = await restored.acall(next_context)
```

Snapshot methods perform no I/O or execution. Reconnect live providers and tools
explicitly; application state, credentials, functions and resource connections
are not serialized. Save between operations, including between tools in a batch.
This does not guarantee exactly-once external effects across a crash before a
tool's result is recorded.

Session is asynchronous and allows one operation at a time per instance. Sync
tools run inline as they do with `Tool.ainvoke`; concurrency and process pools
remain future additions. Provider clients belong to the application, and Session
does not close them. Raw `provider.call/acall` remains available independently.

Run the complete offline example with `python -m examples.session`, or see the
[coding harness](examples/coding_harness/README.md) for budgets, verification and a TUI.

## Existing executing decorator

The lowercase `@prompt` decorator remains available with its existing behavior:
it combines rendering, execution and parsing. Use uppercase `Prompt` for a
provider-independent renderer.

```python
from pydantic import BaseModel
from slick import prompt

class Summary(BaseModel):
    headline: str
    points: list[str]

@prompt(provider=provider, template="summarize.j2")
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

A `def` declaration uses `provider.call`; an `async def` declaration uses
`provider.acall`. Unsupported modes fail clearly; Slick never runs a blocking
provider in a hidden thread or starts an event loop for you.

```python
text = await summarize.render(document)  # render without provider execution
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

## Providers

A provider executes a prompt. A model ID selects the model used by that provider.
All implementations live in `slick.providers`:

| Provider kind | Examples |
| --- | --- |
| CLI tools | `CodexCLI`, `ClaudeCLI`, and custom `Command` subclasses |
| Remote inference | `OpenAIAPI`, `AnthropicAPI`, `OpenRouterAPI`, or `LiteLLMAPI` connected to a hosted endpoint |
| Local inference | `LiteLLMAPI` connected to Ollama, LM Studio, vLLM, or another local server |

LiteLLM can connect to local or remote endpoints using the same adapter. These
categories describe configuration, not separate inheritance trees. OpenCode and
other CLI tools can be integrated through a `Command` subclass; they do not yet
have bundled adapters.

```python
from slick.providers import Provider, OpenAIAPI, AnthropicAPI, CodexCLI, ClaudeCLI, get_command

reader = OpenAIAPI(model="YOUR_OPENAI_MODEL_ID", timeout=60, max_output_tokens=2048)
writer = AnthropicAPI(model="YOUR_ANTHROPIC_MODEL_ID")
coder: Provider = CodexCLI(workdir=".")
claude = ClaudeCLI(workdir=".")
default_cli = get_command()         # configured CLI provider and model
```

All built-in providers inherit `Provider`. Both `call(context, tools=None,
tool_results=None)` and `acall(...)` return `(text, tool_requests)`. Tool arguments
are keyword-only. Text-only responses use an empty request list.

Native API adapters use OpenAI Responses, Anthropic Messages, or Chat Completions.
They validate input before opening a client, perform one exchange, and never
execute Python functions. Refused, incomplete, or unsupported output raises
`ProviderError`; transport exceptions remain accessible through the chained cause.

Native SDK transport retries default to zero; set `max_retries=` explicitly to enable
them. Owned native SDK clients are created and closed per call. For connection reuse,
pass `client=` and/or `async_client=` from the provider's SDK; the application
owns and closes them. Slick applies its timeout/retry settings using the SDK's
`with_options` method. Provider objects retain configuration, not conversation history.

CLI providers retain their commands and permissions. `acall` uses native async
subprocesses. Async timeout or cancellation kills and reaps the owned process;
POSIX async cleanup also targets its process group. Processes that escape that group and
Windows descendants require application-level management. The lower-level
`execute`/`aexecute` methods accept `workdir` and `sandbox` and return an
`ExecutionResult`. The configured directory is used unless an execution supplies
another one; `None` uses the configured default and `""` selects the current directory.
Codex receives its sandbox flag; Claude's existing command controls permissions
and does not enforce the `sandbox` argument. Authentication and subscription/API
billing are owned by the invoked CLI.

`Command` is the abstract CLI provider base. `get_command`, `get_default`, and
`set_default` retain CLI-only default selection: explicit arguments take priority,
then `set_default(provider=..., model=...)`, then `SLICK_PROVIDER` / `SLICK_MODEL`,
then the built-in default. Construct API providers explicitly with a model ID.

A custom provider implements `call(context)` and/or `acall(context)`, returning
`(text, tool_requests)`. Providers supporting tools also accept `tools=` and
`tool_results=`. Existing string-returning custom providers must return `(text, [])`.
No inheritance, registration or metadata is required for ordinary calls.

### OpenRouter API

```python
from slick.providers import OpenRouterAPI

router = OpenRouterAPI("PROVIDER/MODEL")
answer, _ = await router.acall("Explain Python generators.")
```

Install `slick-ai[openai]` and set `OPENROUTER_API_KEY`, or supply `api_key=`.
Replace `PROVIDER/MODEL` with an OpenRouter catalog ID, without LiteLLM's
additional `openrouter/` prefix. Calls go directly to
`https://openrouter.ai/api/v1/chat/completions` using the optional OpenAI SDK,
as described in [OpenRouter's quickstart](https://openrouter.ai/docs/quickstart).
No LiteLLM installation or local gateway process is required.

`call` and `acall` use the shared Slick Chat Completions converter for text and
tool requests. Defaults are `timeout=60`, `max_output_tokens=2048`, and `max_retries=0`.
The retry setting controls SDK transport retries; OpenRouter's own upstream routing
is managed by its service.

For connection reuse, supply an OpenAI SDK `client=` or `async_client=`. Slick
always sets the OpenRouter endpoint. An explicit key takes precedence over
`OPENROUTER_API_KEY`. One of those keys is required even with an injected client;
Slick never carries its existing credentials over to OpenRouter.
The application owns injected clients; Slick closes clients it creates itself.

```bash
slick call "Explain generators" --provider openrouter --model PROVIDER/MODEL
```

### LiteLLM providers and local endpoints

`LiteLLMAPI` wraps the SDK inside Slick's process and calls provider APIs.
It does not run or require a separate gateway server.

```python
from slick.providers import Provider, LiteLLMAPI

local: Provider = LiteLLMAPI(
    "ollama_chat/qwen3:8b",
    api_base="http://localhost:11434",
)
private = LiteLLMAPI(
    "openai/private-model",
    api_base="http://localhost:8000/v1",
    api_key="local",
)
router = LiteLLMAPI("openrouter/PROVIDER/MODEL")

answer, _ = await local.acall("Explain Python generators.")
```

Replace `PROVIDER/MODEL` with an available OpenRouter catalog ID and `private-model`
with a model served by your endpoint. Slick does not maintain a model allowlist.
Start Ollama or your other inference server yourself; Slick does not load weights
or start a server. See LiteLLM's [providers](https://docs.litellm.ai/docs/providers),
[OpenRouter](https://docs.litellm.ai/docs/providers/openrouter), and
[compatible endpoints](https://docs.litellm.ai/docs/providers/openai_compatible).

LiteLLM resolves credentials from provider environment variables such as
`OPENROUTER_API_KEY` when `api_key` is omitted. `api_base` overrides the endpoint.
`timeout=60` and `max_retries=0` are the defaults. Supply inference settings through
`options`, for example `options={"temperature": 0.2, "max_tokens": 256}`. No default
output token limit is imposed. Provider-specific settings pass to LiteLLM;
model and provider capabilities determine whether they are supported.

Calls use the SDK's `completion`/`acompletion` functions without requiring a proxy.
They reject truncated, refused, or malformed responses with `ProviderError`,
retaining provider exceptions as chained causes. Text is returned unchanged inside
the tuple. Tools and their results use the same Chat Completions conversion as
OpenRouter. Streaming and fallback routing cannot be set in `options`.
Slick requests `drop_params=False`, but individual LiteLLM provider adapters can
still translate or filter parameters.

The SDK owns its internal clients, caches, and callbacks; Slick does not change
process-wide LiteLLM settings. For offline local inference, set
`LITELLM_LOCAL_MODEL_COST_MAP=True` **before the first call** to use the SDK's bundled
model metadata. A local inference endpoint alone does not disable the SDK's other
network activity. See the [cost-map loader](https://github.com/BerriAI/litellm/blob/v1.100.0/litellm/litellm_core_utils/get_model_cost_map.py).

LiteLLM's `chatgpt/` and `github_copilot/` providers manage their own authentication;
they do not run your installed CLI. Their availability and behavior are
provider-specific. The evaluated ChatGPT adapter injects default instructions and
discards output token limits, so Slick rejects those explicit limit options for
`chatgpt/`. Use `CodexCLI` or `ClaudeCLI` when you want execution through that installed CLI.

## Tool requests and results

```python
text, requests = await provider.acall(context, tools=[lookup])
```

`tools` contains ordinary functions, bound methods, or prepared `Tool` instances.
Each returned request is a dictionary:

```python
{"id": "a", "name": "lookup", "arguments": {"name": "blue"}}
```

Your application executes requests and supplies results on its next call:

```python
import asyncio
from slick import ToolError
from slick.providers import AnthropicAPI
from slick.tools import prepare_tools


def lookup(name: str) -> str:
    """Look up a local color description."""
    return {"blue": "a cool primary color"}.get(name, "unknown")


async def main():
    provider = AnthropicAPI(model="YOUR_MODEL_ID")
    tools = prepare_tools([lookup])
    context = "Look up blue and describe it."
    results = []
    for _ in range(5):
        text, requests = await provider.acall(
            context, tools=list(tools.values()), tool_results=results,
        )
        if not requests:
            print(text)
            return
        results = []
        for request in requests:
            error = request.get("argument_error")
            if not error and request["name"] not in tools:
                error = "Unknown tool"
            if not error:
                try:
                    content = await tools[request["name"]].ainvoke(request["arguments"])
                except ToolError as exc:
                    error = str(exc)
            results.append({
                "request": request,
                "content": error if error else content,
                "is_error": bool(error),
            })
        # Render different context here if desired; only this text is sent next.
    raise RuntimeError("Request budget exhausted")


asyncio.run(main())
```

Install the appropriate SDK extra and configure credentials before running real
calls. The same interface is available on OpenAIAPI, OpenRouterAPI, and LiteLLMAPI.

Results contain the originating request because an ID alone does not tell a fresh
provider instance the function name or its arguments. There is no separate outgoing
`tool_calls` argument. Dictionaries survive JSON save/load and can be submitted to
another instance without replaying conversation history. `ToolRequest` and
`ToolResult` in `slick.tools` are optional TypedDict annotations, not wrapper objects.

`content` is serialized text. Tool invocation already converts structured return
values to JSON; do not encode that text twice. `is_error` defaults to false.
Malformed model JSON arguments retain their raw string and an `argument_error`;
return an error result without executing them. Anthropic's object-only tool input
cannot represent malformed raw JSON from another provider and is rejected explicitly.

Slick validates dictionary fields and unique IDs within each batch. It does not
remember earlier IDs or infer missing results. The application chooses which
requests to answer, in what order, and with what context. Empty context is allowed
with results; an empty call with no results fails locally. Results may be supplied
with `tools=[]` to make no functions available for the next response.

Provider converters build the corresponding assistant tool requests and outputs
internally. OpenAI uses `function_call_output`, Anthropic uses `tool_result`, and
Chat Completions uses tool-role messages. OpenAI definitions use `strict=False`
to retain Python optional/default arguments; local Tool validation remains strict.

This portable interface covers text and local function tools. It does not expose
usage metadata, hosted tools, media, or native reasoning replay. Responses that
combine tool requests with reasoning data requiring replay raise `ProviderError`
instead of silently dropping that data. Final text from reasoning responses remains
supported. Command providers return `(text, [])` and reject nonempty Python tools
or tool results before launching a process.

The former `aturn`, turn dataclasses, and provider history validation have been
removed. Migrate string consumers to `text, requests = ...`; prompt decorators
still return their declared Python output type and reject pending tool requests
before parsing, repair, or caching. `Prompt`, `render`, and `parse` remain independent
text operations.

The [coding harness example](examples/coding_harness/README.md) adds error recovery,
cancellation, workspace tools, verification, saved app state, and a TUI as ordinary
Python. Run its offline demo with
`python -m examples.coding_harness --dry-run --headless --task 'Fix total'`.

## Explicit execution options and compatibility

`@prompt(provider=...)` and new async declarations default to **no disk logging,
no caching, and no automatic output repair**. Provider selection is fixed on the
decorator; passing both `provider=` and `model=` or overriding a modern declaration
with per-call `model=` raises an error.

Explicit options remain available:

```python
@prompt(provider=provider, template="summarize.j2", max_repairs=1)
def summarize(document: str) -> Summary:
    """Summarize a document, allowing one formatting repair."""
```

`log_dir=` opts into prompt/response files. `cache=True` opts into reuse of accepted
responses (under `log_dir` or `LOG_DIR`). `max_repairs=` permits additional provider
calls after validation failure; it is separate from SDK transport retries.
`output=` on a text-returning function explicitly saves its response to a file.
Caching is an application decision, especially for harnesses or external state.

Persistent calls need a stable, JSON-serializable provider `identity()` excluding
secrets, or legacy `provider`/`model` metadata. Built-in API adapters provide one.
Injected clients with different endpoints or external state need separate cache
directories or an application-defined identity; Slick does not inspect credentials
or infer those differences. Accepted responses are written only after validation.

Existing **synchronous** bare `@prompt` and `@prompt(model=...)` declarations keep
their original defaults: disk logging, caching, one repair, per-call `model=`
overrides, and default model resolution. `get_command`, `set_default`, environment
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
and templates stay under `examples/`. Pass `--provider openai --model YOUR_MODEL_ID`
(or another supported provider) for real calls.

The original functional examples also remain available:

```bash
python -m examples.core
```

[examples/core.py](examples/core.py) shows a typed summary, a conversation supplied
as a list of messages, and application-owned document retrieval. Its templates
use shared Jinja includes. To use a real provider, set `TEMPLATE_ROOT` to
`examples/prompts` and pass that provider to `examples.core.run`.

## CLI and development

```bash
slick provider
slick call "summarize this"
cat document.md | slick call --provider codex
slick call "Explain generators" --provider litellm --model ollama_chat/qwen3:8b --api-base http://localhost:11434
slick call "Summarize this" --provider openai --model YOUR_MODEL_ID

poetry install --with dev
poetry run pytest
poetry run ruff check slick tests examples
```

Explicit `litellm`, `openai`, `anthropic`, and `openrouter` CLI choices require `--model` and use
provider environment credentials. `--api-base` is available only with `litellm`.
Omitting `--provider` and the `slick provider` command show the configured CLI defaults;
`get_command` remains the Codex/Claude resolver.

Ordinary tests use fake SDK clients and local Python subprocesses. Optional SDK
transport tests run when their API/LiteLLM extras are installed, with external
connections blocked in the LiteLLM checks; no tests call paid endpoints.

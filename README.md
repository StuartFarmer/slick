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
are explicit application choices. Imports and provider construction do not load
optional SDKs or make requests. Each SDK loads when its provider sends a request,
so install the corresponding extra before calling it. Configuration passes through
without constructor validation; the SDK handles invalid settings.

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
for invalid structured output. Structured responses must be JSON, including quoted
JSON strings for string literals. It never calls a provider or repairs output.

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

Use `@tool` to create a tool from a function's name, docstring, and annotations:

```python
from slick import tool

documents = {"intro": "Slick renders Jinja templates."}

@tool
def lookup(key: str) -> str:
    """Retrieve a document by its key."""
    return documents[key]

text = lookup.invoke({"key": "intro"})
```

`@tool()` also works. Override metadata with
`@tool(name="read_document", description="Read a document.")`; both overrides
default to `None`, which uses the function's metadata. The decorator returns a
`Tool` instance, ready to pass to a session or provider.

You can also prepare ordinary functions or bound methods without decorating them:

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

Preparation builds schemas without executing functions. Names and descriptions are
used as supplied; missing docstrings become empty descriptions. Duplicate names
use the last supplied function. Use `tool(documents.read, name="read_document")`
for an override, or `Tool(documents.read, name="read_document", description="...")`
to supply both fields explicitly. Bound methods retain their instance and state.

Pydantic generates the parameter schema directly from the callable. Annotations
and `Annotated[T, Field(...)]` describe inputs to the model; they do not validate
or convert arguments during invocation. Arguments go straight to `function(**arguments)`,
using Python parameter names and defaults. Nested dictionaries stay dictionaries;
construct Pydantic models inside the function if needed. Python, the function,
or the provider handles invalid input when it encounters it.

Return annotations are not enforced. Strings pass through unchanged; other
results use Pydantic's standard JSON serialization. There are no custom checks
for JSON keys, nonfinite numbers, defaults, names, or dictionary fields.

`ToolError` exposes `.name` and `.phase` (`definition`, `execution`, or `result`)
and preserves the underlying exception as its cause. Functions are never retried
automatically. A serialization error occurs after the function has run and does
not undo its effects. `ainvoke` awaits async functions and runs sync functions
inline. Wrap blocking work with `asyncio.to_thread` when needed. Cancellation
propagates normally.

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

## Executing decorator

The lowercase `@prompt` decorator combines rendering, execution and parsing.
Supply a configured `provider=` to execute it. Use uppercase `Prompt` for a
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
constrained decoding. Invalid results raise Pydantic's `ValidationError` directly.
The application owns retries and repairs.

A `def` declaration uses `provider.call`; an `async def` declaration uses
`provider.acall`. Unsupported modes fail clearly; Slick never runs a blocking
provider in a hidden thread or starts an event loop for you.

```python
text = await summarize.render(document)  # render without provider execution
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
from slick.providers import Provider, OpenAIAPI, AnthropicAPI, CodexCLI, ClaudeCLI

reader = OpenAIAPI(model="YOUR_OPENAI_MODEL_ID", timeout=60, max_output_tokens=2048)
writer = AnthropicAPI(model="YOUR_ANTHROPIC_MODEL_ID")
coder: Provider = CodexCLI(workdir=".")
claude = ClaudeCLI(workdir=".")
```

All built-in providers inherit `Provider`. Both `call(context, tools=None,
tool_results=None)` and `acall(...)` return `(text, tool_requests)`. Tool arguments
are keyword-only. Text-only responses use an empty request list.

Providers are ordinary classes with handwritten constructors. The implementation
lives in `slick/providers/base.py`, `api.py`, and `cli_tool.py`; `__init__.py`
exports the provider classes.

Native API adapters use OpenAI Responses, Anthropic Messages, or Chat Completions.
They pass input to the SDK and extract available text and function calls. Status,
finish reasons, and extra response blocks are not validated. Partial text is
returned as supplied; unsupported blocks are ignored. SDK and decoding exceptions
raise `ProviderError` with the original exception as their cause.

Native SDK transport retries default to zero; set `max_retries=` explicitly to enable
them. Each native API call creates and closes its own SDK client with the configured
timeout and retry settings. `call` uses the synchronous SDK and `acall` uses the
asynchronous SDK. Provider objects store configuration, not clients or conversation history.

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

`Command` runs an ordinary command with the prompt on stdin and returns stdout.
It shares process execution, timeout, and cancellation handling between CLI providers.
`ClaudeCLI` adds Claude's model flag; `CodexCLI` builds `codex exec` arguments and
extracts the final message from JSONL output.

All `command=` values use shell-style quoting and are split into arguments without
launching a shell. For a Python wrapper, pass `command='python "path/to/wrapper.py"'`;
the provider does not choose an interpreter from the filename. The process starts
in `workdir`, so Codex needs no additional `--cd` flag.

Construct providers explicitly. API providers require a model ID; CLI providers
use the invoked CLI's model default when `model` is omitted.

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

Slick creates an OpenAI SDK client pointed at the OpenRouter endpoint for each call.
An explicit `api_key` takes precedence over `OPENROUTER_API_KEY`; one is required.

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
They return the first choice's available text and tool calls, including partial
output, using the same Chat Completions conversion as OpenRouter. `options` is an
ordinary dictionary passed through without copying or validation. Its entries
override request defaults; explicit `api_base` and `api_key` fields take precedence.
Unsupported options or response formats fail in the SDK or during decoding.
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
discards output token limits. Slick passes these options through to the SDK.
Use `CodexCLI` or `ClaudeCLI` when you want execution through that installed CLI.

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
`ToolResult` in `slick.tools` are optional TypedDict annotations. Provider and
session APIs use plain dictionaries.

`content` is serialized text. Tool invocation already converts structured return
values to JSON; do not encode that text twice. `is_error` defaults to false.
JSON syntax errors retain their raw string and an `argument_error`;
return an error result without executing them. Anthropic expects object input;
its SDK or API handles malformed raw JSON passed from another provider.

Request and result dictionaries pass through without field or uniqueness checks.
JSON arguments use ordinary `json.loads` behavior. The application chooses which
requests to answer, in what order, and with what context. Empty context can be used
with results; providers handle empty calls. Results may be supplied with `tools=[]`
to make no functions available for the next response.

Provider converters build the corresponding assistant tool requests and outputs
internally. OpenAI uses `function_call_output`, Anthropic uses `tool_result`, and
Chat Completions uses tool-role messages. OpenAI definitions use `strict=False`
to retain Python optional/default arguments.

This portable interface covers text and local function tools. It does not expose
usage metadata, hosted tools, media, or native reasoning replay. Decoders extract
text and function calls and ignore other blocks, including reasoning data.
Command providers use only the context and return `(text, [])`; `tools` and
`tool_results` are ignored.

The former `aturn`, turn dataclasses, and provider history validation have been
removed. Migrate string consumers to `text, requests = ...`; prompt decorators
still return their declared Python output type and reject pending tool requests
before parsing or caching. `Prompt`, `render`, and `parse` remain independent
text operations.

The [coding harness example](examples/coding_harness/README.md) adds error recovery,
cancellation, workspace tools, verification, saved conversations, and a TUI around
an explicit provider/tool loop. Run its offline demo with
`python -m examples.coding_harness --dry-run --headless --task 'Fix total'`.

## Execution and persistence

`@prompt` renders, calls the supplied `provider=`, and validates the response.
Applications own caching, logging, output files, and retries. Custom providers only
need `call` or `acall`; the decorator does not require `identity()`.

The decorator no longer accepts `cache=` or `log_dir=`, and `LOG_DIR` is removed.
`output` is now an ordinary template argument, with no file-writing behavior.
The `.source()`, `.template_name`, and `.returns` inspection attributes are also
removed; `.render()` remains available, and the wrapped function retains its
annotations and docstring.

For file output, write the returned string in application code, or serialize a
typed result explicitly. To preserve a model's exact JSON text, use
`render`, `provider.call/acall`, and `parse` separately and save the raw response
after validation.

Template errors, bad function arguments, and missing provider methods propagate
directly from Jinja or Python.

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
slick call "summarize this" --provider codex
cat document.md | slick call --provider codex
slick call "Explain generators" --provider litellm --model ollama_chat/qwen3:8b --api-base http://localhost:11434
slick call "Summarize this" --provider openai --model YOUR_MODEL_ID

poetry install --with dev
poetry run pytest
poetry run ruff check slick tests examples
```

Explicit `litellm`, `openai`, `anthropic`, and `openrouter` CLI choices require `--model` and use
provider environment credentials. `--api-base` is available only with `litellm`.
`--provider` is required. For `codex` and `claude`, `--model` is optional and
omitting it uses the invoked CLI's model default.

Ordinary tests use fake SDK clients and local Python subprocesses. Optional SDK
transport tests run when their API/LiteLLM extras are installed, with external
connections blocked in the LiteLLM checks; no tests call paid endpoints.

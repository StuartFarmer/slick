# slick

Jinja templates to LLMs and back, as ordinary Python functions.

```python
from slick import Prompt
from slick.providers import OpenAIAPI

provider = OpenAIAPI(model="YOUR_MODEL_ID")
answer_prompt = Prompt("answer.j2")
text = answer_prompt(question="How does authentication work?", documents=documents)
answer = await provider.acall(text)
```

Prompts render arguments into text. Providers execute text and return responses.
Your Python code connects them and owns retrieval, history, sequencing, and
concurrency. Jinja handles context composition; `parse` validates results when needed.

## Install

```bash
pip install slick-ai                  # base package with CLI providers
pip install 'slick-ai[openai]'         # optional OpenAI SDK
pip install 'slick-ai[anthropic]'      # optional Anthropic SDK
pip install 'slick-ai[api]'            # both API SDKs
pip install 'slick-ai[litellm]'        # optional multi-provider SDK (no proxy server)
```

Native API providers use `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` when called. Model IDs
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
response = provider.call(text)           # synchronous execution
response = await provider.acall(text)    # asynchronous execution (a separate call)

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
        answer = await self.provider.acall(text)
        self.history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        return answer

    async def critique(self, answer):
        return await self.provider.acall(self.critique_prompt(answer=answer))
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

This is the local tool primitive. API providers still accept text only; native
tool registration, provider schema adaptation, and the model/tool loop are a
separate milestone. `parameters` provides a fresh local schema, which may require
adaptation for a provider's supported schema subset.

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
| Remote inference | `OpenAIAPI`, `AnthropicAPI`, or `LiteLLMGateway` connected to a hosted endpoint |
| Local inference | `LiteLLMGateway` connected to Ollama, LM Studio, vLLM, or another local server |

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

All built-in providers inherit `Provider`, an abstract class requiring `call` and
`acall`. Pass the complete prompt positionally; both methods return final text.
CLI agents may perform multiple steps or use tools before returning that text.
The shared class does not require `aturn`, `execute`, or a metadata interface.

Native API adapters use [OpenAI Responses](https://developers.openai.com/api/docs/libraries)
and [Anthropic Messages](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python).
Both expose `call(text)` and `acall(text)`. Refused, incomplete, or unexpected
tool outputs raise `ProviderError`; native transport
exceptions remain accessible through the chained cause. The text-only methods
do not configure tools; use `aturn` for native tool requests.

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

A custom provider only needs `call(text) -> str`, `acall(text) -> str`, or both.
No inheritance, registration or metadata is required for ordinary calls.

### LiteLLM providers and local endpoints

```python
from slick.providers import Provider, LiteLLMGateway

local: Provider = LiteLLMGateway(
    "ollama_chat/qwen3:8b",
    api_base="http://localhost:11434",
)
private = LiteLLMGateway(
    "openai/private-model",
    api_base="http://localhost:8000/v1",
    api_key="local",
)
router = LiteLLMGateway("openrouter/PROVIDER/MODEL")

answer = await local.acall("Explain Python generators.")
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
They reject truncated, refused, malformed, or tool-call responses with
`ProviderError`, retaining provider exceptions as chained causes. Successful text
is returned unchanged, including whitespace. The adapter does not configure
tools, streaming, or fallback routing; those controls cannot be set in `options`.
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

`LiteLLMGateway` currently supports text calls only. It has no `aturn` or automatic
persistence identity. Use modern `@prompt(provider=...)` defaults; opting into
decorator logging/caching requires an application-defined provider identity.
This avoids deriving cache keys from arbitrary credential-bearing options.

## Native tool turns

The native OpenAI and Anthropic providers also expose
`aturn(history, tools=..., instructions=...)`. It prepares
callables and performs one request, returning text, tool calls, provider-native
items and available usage. The application owns history and executes the functions:

```python
import asyncio
from slick import ToolError
from slick.providers import OpenAIAPI
from slick.tools import prepare_tools
from slick.turns import ToolResult, UserMessage

def lookup(name: str) -> str:
    """Look up a local color description."""
    return {"blue": "a cool primary color"}.get(name, "unknown")

async def main():
    provider = OpenAIAPI(model="YOUR_MODEL_ID")
    tools = prepare_tools([lookup])
    history = [UserMessage("Look up blue and describe it.")]
    for _ in range(5):
        turn = await provider.aturn(history, tools=list(tools.values()))
        history.append(turn)
        if not turn.tool_calls:
            print(turn.text)
            return
        for call in turn.tool_calls:
            if call.argument_error or call.name not in tools:
                result = ToolResult(call.id, call.argument_error or "Unknown tool", True)
            else:
                try:
                    output = await tools[call.name].ainvoke(call.arguments)
                except ToolError as error:
                    result = ToolResult(call.id, str(error), True)
                else:
                    result = ToolResult(call.id, output)
            history.append(result)
    raise RuntimeError("Turn budget exhausted")

asyncio.run(main())
```

Install the API extra and set credentials before running this live example.
`AnthropicAPI` supports the same interface. Plain functions and bound methods may
also be passed directly in `tools=[...]`. Native records live in `slick.turns`.
Preserve returned turns unchanged: their opaque provider payloads carry information
needed by subsequent requests. Supply one result for every requested tool before
the next turn. Foreign provider/model histories and broken result groups fail
before network I/O. There is no automatic tool execution or retry loop.
OpenAI tool definitions use `strict=False` to preserve Python optional/default
arguments; Slick's local argument and return validation remains strict. These
native methods support local function tools, not provider-hosted tools or media.

The [coding harness example](examples/coding_harness/README.md) adds error recovery,
cancellation, workspace tools, verification, sessions and a TUI as ordinary Python.
Its offline demo runs with `python -m examples.coding_harness --headless --task 'Fix total'`.

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

Explicit `litellm`, `openai`, and `anthropic` CLI choices require `--model` and use
provider environment credentials. `--api-base` is available only with `litellm`.
Omitting `--provider` and the `slick provider` command show the configured CLI defaults;
`get_command` remains the Codex/Claude resolver.

Ordinary tests use fake SDK clients and local Python subprocesses. Optional SDK
transport tests run when their API/LiteLLM extras are installed, with external
connections blocked in the LiteLLM checks; no tests call paid endpoints.

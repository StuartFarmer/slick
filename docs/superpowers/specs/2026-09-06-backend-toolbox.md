# Slick backend toolbox and native tools design

> Superseded for current implementation by the [functional core](../specs/2026-09-07-functional-core.md). This document records earlier exploration; Agent, native tools and richer request protocols are deferred.

Date: 2026-09-06. Status: revised proposed design incorporating the user's scope corrections. Slick owns native function tools; orchestration and application state are ordinary Python. No runtime implementation is included.

**Goal:** enable model inference, tool-using agents and workspace harnesses to be called as typed Python functions and mixed freely with ordinary application code.

**Global constraints**

- Python >=3.10 remains supported.
- Jinja2 and Pydantic remain the only mandatory third-party dependencies.
- Provider SDKs are optional extras and are imported lazily; native Slick tools require no agent framework.
- The primary function API remains `@prompt`.
- Existing `model=`, `get_model()`, `Model.call()`, `Model.execute()`, CLI commands and template behavior remain available.
- Backend selection never silently changes local tool access or credentials.
- Library execution never prints model output to stdout; tracing and streaming are deferred.
- Modern backend= agent/harness work is never automatically rerun to repair its final representation.
- All interfaces shown below are proposed, not currently implemented.

**Public experience**

A toolbox is an application-owned Python module. Objects store configuration; importing the module must not launch a process, make a request or require credentials before the backend is used.

```python
# toolbox.py — proposed Slick imports
from slick.backends import OpenAI, Anthropic, Codex, ClaudeCode

extractor = OpenAI(model="YOUR_OPENAI_MODEL_ID")
writer = Anthropic(model="YOUR_ANTHROPIC_MODEL_ID")
repo_reader = Codex(workdir=".", sandbox="read-only")
coder = ClaudeCode(workdir=".", allowed_tools=["Read", "Edit", "Bash"])
```

The model IDs are application choices, not model recommendations. Harness permissions are backend-specific configuration: `allowed_tools` is not a portable filesystem sandbox, and a raw API model has no workspace setting.

```python
# workflows.py — definitions of Finding and Report are application-owned
from slick import prompt
from toolbox import extractor, writer, repo_reader

@prompt(backend=extractor, template="extract.md.j2")
def extract(document: str) -> list[Finding]:
    ...

@prompt(backend=repo_reader, template="investigate.md.j2")
async def investigate(question: str) -> list[Finding]:
    ...

@prompt(backend=writer, template="report.md.j2")
def write_report(findings: list[Finding]) -> Report:
    ...
```

No toolbox registry, service container or YAML definition is needed. Reuse objects by importing them. Different functions can use the same backend with different output types. A configured backend represents reusable execution configuration, not a shared conversation.

The common abstraction is behavioral, with parallel implementations. The model → agent → harness progression does not require an inheritance chain. Agents use models; harnesses run agents. Slick integrates at each boundary.

**Shared backend contract**

Use a `slick/backends/` package with a small `base.py`. Proposed definitions:

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

@dataclass(frozen=True)
class Request:
    prompt: str
    returns: Any = str

    @property
    def schema(self) -> dict[str, Any] | None:
        from pydantic import TypeAdapter
        return None if self.returns is str else TypeAdapter(self.returns).json_schema()

@dataclass(frozen=True)
class Response:
    text: str
    usage: dict[str, int] | None = None
    raw: Any = field(default=None, repr=False, compare=False)

class Backend(Protocol):
    kind: Literal["model", "agent", "harness"]

    def identity(self) -> dict[str, Any]:
        """Stable, JSON-serializable configuration; excludes secrets."""
        ...

    def run(self, request: Request) -> Response:
        ...

    async def arun(
        self, request: Request
    ) -> Response:
        ...
```

`Request.returns` preserves the original Python output type for framework adapters; `schema` gives API/CLI adapters its JSON Schema. Provider schema transformations belong to the adapter. Shared validation always uses the original type. The general backend protocol carries no conversation history. Slick's native tool representation and provider turn handling are internal to the Agent/model path described below.

`Response` represents completed work. Refusal, incomplete output, unsupported operation and operational failure raise `BackendError` with a reason and chained native cause. Cancellation propagates as cancellation after cleanup. Keep native error/response details accessible without blindly logging secrets.

Both execution methods are deliberate. Built-in API backends use their SDK's sync/async clients; CLI backends use sync/async subprocess APIs. The native Agent backend offers both modes using the corresponding provider path. No event sink, run context, trace correlation or streaming interface is part of this delivery.

**Prompt integration and migration**

Add `backend=` to the decorator, accepting an object implementing the common contract. Keep legacy `model=` resolution through the existing path and a small legacy adapter. Passing both decorator keywords is an error.

Keep backend selection at declaration time for the first release. Do not add a reserved per-call `backend` keyword: existing template functions may already have an argument named backend. Existing legacy per-call `model=` overrides continue working on legacy declarations. On a declaration using `backend=`, reject a per-call model override with a clear message rather than silently switching execution authority.

Keep `.render()`, `.source()`, `.template_name`, `.returns` and `output=`. Add accurate `ParamSpec`/`TypeVar` overloads so decorated sync functions remain statically typed functions and async declarations remain awaitable typed functions.

For modern `backend=` declarations, `cache=None` and `max_repairs=None` select conservative defaults: model backends cache equivalent calls and allow one output repair; agent/harness backends do neither. Explicit cache settings remain available. Reject positive automatic repair counts for agent/harness backends in the initial version. Legacy calls retain their current default cache and repair behavior, documented as compatibility behavior.

An async prompt body may compute additional context before rendering. On async declarations, `await fn.render(...)` is the no-backend render interface; on sync declarations, `fn.render(...)` stays synchronous. Do not change return shape depending on ambient event-loop state.

**Structured output and failures**

Retain current Jinja text and `{{ output_format }}` behavior. Supply the same output contract separately to native backends. Add a shared `outputs.py` only when splitting the existing parser makes adapter work clearer; it owns Pydantic validation and general envelope handling, not every provider's schema dialect.

Provider adapters normalize their wire schema without mutating the original schema. Root scalars, arrays and unions can use a private root object with a `value` property, and references must remain valid after wrapping. Local validation checks the unwrapped value. Detect unsupported schemas; require an explicit prompted-output fallback setting before weakening native enforcement. Keep the selected output mode explicit in backend configuration and cache identity.

OpenAI and Anthropic provide native schema output. Codex exposes `--output-schema`, and Claude Code exposes `--json-schema` with its structured result field. Preserve native completion/refusal information before parsing JSON. [API/CLI evidence](../../../outputs/models-and-harnesses-comparison.md)

Use distinct policies for SDK transport retries, inference output repairs and application-level review loops. Keep attempt limits explicit; do not multiply independent retry defaults unknowingly. A malformed result from an agent run fails with its captured output. A later explicit formatting call can turn that captured text into a typed value without sending the original task back to the agent.

**Async, lifecycle and bounds**

Built-in API adapters create and close owned SDK clients per call initially. This works across sync calls and separate async loops without hidden persistent sessions. Accept injected sync/async clients for applications that want connection pooling; the application owns and closes those clients.

Harness runs own their subprocess and temporary schema files. Sync calls enforce subprocess timeouts; async calls use `asyncio.create_subprocess_exec`. Timeout/cancellation interrupts or terminates the process as appropriate, waits for exit, and cleans up owned resources. Test child-process behavior on supported platforms. Cancelling local execution is not rollback of already-completed external actions.

Each backend has a wall-clock timeout. API output limits and harness/runtime turn limits are configured where actually supported. Unsupported limits fail before work starts. The native Agent bounds model turns, dispatched tool calls and wall-clock duration separately.

**Native tools: pass Python functions**

Slick owns function discovery, argument schemas, validation, dispatch and result serialization. Agent frameworks are not required. No tool decorator or base class is needed for ordinary typed functions.

```python
# Proposed API. orders is an application-owned data source.
from slick.backends import Agent, OpenAI

def get_order(order_id: str) -> dict:
    """Fetch an order by its ID."""
    return orders.get(order_id)

support = Agent(
    model=OpenAI(model="YOUR_OPENAI_MODEL_ID"),
    tools=[get_order],
    max_steps=8,
)
```

The callable is described by its name, docstring and JSON Schema parameters. OpenAI uses `parameters` in its function-tool representation; Anthropic uses `input_schema`. Their call identifiers and result envelopes also differ. Slick follows those established function-calling interfaces through provider adapters; it does not invent a wire protocol or require the formats to be identical. [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [Anthropic tool definitions](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools), [Anthropic tool results](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)

Public Agent constructor: `Agent(*, model, tools, max_steps=8, max_tool_calls=32, timeout=120)`. `model` must expose Slick's provider turn capability; OpenAI and Anthropic implement it. A generic harness backend cannot be substituted into the model slot just because it can produce final text. Harnesses already own their loop.

Rules for a supplied tool:

- Inspect its real signature, annotations and docstring, including wrapped Slick functions. Parameters must have supported type annotations. Resolve a bound method's signature without exposing its bound receiver.
- Derive a Pydantic input model and JSON Schema. Preserve defaults and nested references. Reject duplicate names, unsupported signatures, unresolved annotations and unsupported schemas before starting execution.
- Name, description and parameter schema go to the model. Function bodies, closure variables and application clients do not.
- Match a requested name only against the explicitly supplied functions. Bind and validate incoming JSON arguments before executing any tool in that returned batch; never evaluate model-supplied Python.
- Execute multiple requested calls sequentially in the initial version. Support `def` and `async def`. Async execution awaits async tools and can offload synchronous tools with `asyncio.to_thread`; cancellation cannot forcibly stop an already-running synchronous function, so do not claim that it does.
- Validate annotated return values, then serialize JSON-compatible data through Pydantic; pass strings as text. An unsupported output type fails clearly. The model consumes serialized results; ordinary callers continue receiving the original Python values.
- A tool exception stops the agent call with the tool name and chained cause. There is no automatic retry of a possibly effectful tool. Applications can write wrappers that return recoverable error values when desired.
- Native strict-schema options are used only where the signature/default behavior can be represented faithfully; local argument validation always remains authoritative.

The Agent backend owns only the bounded loop: request a model turn, validate/execute requested tools, return their results, and continue until a final response. Each invocation has independent turn state. Provider adapters preserve provider-native continuation data, including OpenAI reasoning items when required and Anthropic content blocks/call IDs; a transcript flattened into text is insufficient.

This requires a private model-turn interface in addition to the completed-backend interface. It returns tool requests or final output plus opaque provider continuation state. Direct model backends still reject unexpected tool requests when no Agent is running them. The Agent shares one overall deadline across model turns and tool dispatch, bounds provider requests and awaited operations by the remaining time, checks the tool-call budget before dispatching a batch, and raises on exhaustion. Arbitrary synchronous tools cannot be forcibly interrupted: their deadlines are cooperative, checked before and after execution. Nested agents have their own limits; do not claim a universal global token budget.

**An agent as a tool is a function in the tool list**

```python
# Proposed API. Prompt templates are ordinary application files.
from slick import prompt
from slick.backends import Agent, OpenAI, Codex

@prompt(backend=Codex(workdir=".", sandbox="read-only"),
        template="inspect_repo.md.j2")
async def inspect_repo(question: str) -> str:
    """Inspect the repository to answer a code question."""

engineer = Agent(
    model=OpenAI(model="YOUR_OPENAI_MODEL_ID"),
    tools=[inspect_repo],
)
```

Calling `await inspect_repo(question)` from application code invokes the function directly. Supplying `inspect_repo` in `tools` lets the outer model decide when to invoke that same function. Slick validates its arguments, awaits its backend execution, and returns its result to the outer model. The outer agent then continues. There is no special agent-as-tool wrapper, automatic history sharing, conversation handoff or new delegation paradigm.

The same mechanism works for a function backed by a raw model or another Slick Agent. Using `template=` leaves the docstring free to describe the function as a tool without publishing its internal prompt template as its tool description.

**Python owns coordination and state**

Pipelines, branches, feedback loops, concurrency, memory/context construction, waiting for a webhook, and communicating with another service are ordinary application code. A coroutine can await any signal; Slick has no named human-review step or special external-event abstraction. Memory providers and context stores are ordinary dependencies passed or closed over by application functions. Common patterns may become optional addons/plugins later, without becoming core concepts or a plugin-management subsystem now.

Durable execution belongs to another library. Slick does not build a durable runtime, checkpoint interface or durable-execution integration milestone.

**Harness tools and process boundaries**

Codex and Claude Code retain their native tool loops. Host Python callables need a transport to run from those processes. Reuse the same Slick function descriptions through an optional MCP bridge when exposing local functions to harnesses; do not create a second tool-definition system. The native OpenAI/Anthropic Agent path works in-process and has no MCP requirement. Until that bridge is implemented, a harness does not accept local Python functions silently; it uses its already configured native/MCP tools. MCP integration is a transport extension, separate from the native tool feature.

**Tracing and streaming: deferred**

No new `run` context manager, event sink, call tree, run journal, streaming interface or tracing dependency is included. Existing prompt logs/cache behavior stays available. Do not reserve event-related arguments or create empty modules for later instrumentation.

Maintain correct caching within the existing mechanism: cache identity includes prompt, output schema/mode, provider/model identity and output-affecting options; writes become visible only after successful validation. Agent and harness execution remain non-cached by default. A separate execution journal is deferred.

**Where the landscape features belong**

| Feature | Ownership | Delivery |
| --- | --- | --- |
| Backend toolbox | Slick | Common typed call interface and ordinary configured objects |
| OpenAI / Anthropic inference | Slick provider adapters | Optional provider SDK dependencies |
| Codex / Claude Code harnesses | Slick harness adapters | Process lifecycle, schemas and explicit settings |
| Python tools and model-selected calls | Slick | Functions → provider tool schemas → validated invocation |
| Tool-using agent | Slick Agent backend | Small bounded model/tool loop using provider-native continuation |
| Agent/function as a tool | Same Slick tool feature | Pass a decorated callable in `tools=[...]` |
| Pipelines, routing and critic loops | Application Python | No Slick paradigm or separate feature |
| Parallel work | Python/asyncio | Slick functions remain awaitable |
| Context and memory | Application Python and chosen libraries | Possible optional addons later; no core memory system |
| Awaiting people, webhooks, services or agents | Application Python | Ordinary await/callback integration, no named review abstraction |
| Durable execution | Another library | Outside Slick |
| Tracing and streaming | Deferred | No API scaffolding now |

**Delivery and acceptance**

| Milestone | Scope | Acceptance |
| --- | --- | --- |
| A: typed backend foundation | Common contract, decorator dispatch, sync/async, OpenAI/Anthropic, CLI schema output, repair/cache policy, compatibility | Typed functions mix API inference and harness calls; old behavior remains available |
| B: native tools and Agent | Function schemas/dispatch, OpenAI and Anthropic turn adapters, bounded Agent backend, decorated functions as tools | The same Python tool works with either provider, and a nested Slick function returns to the caller correctly |
| C: optional harness tool transport | MCP bridge reusing Slick's function tool descriptions, only when required | A harness can invoke an explicitly exposed local function without a second definition system |

A is detailed in [the foundation plan](../plans/2026-09-06-backend-foundation.md); B is detailed in [the native tools plan](../plans/2026-09-06-native-tools.md). C is an optional transport extension, not a prerequisite for in-process tools. Python composition examples may document usage, but are not separate orchestration deliverables.

**Repository map**

- `slick/prompts.py`: decorator, render/context binding, sync/async call dispatch and typed return validation.
- `slick/backends/base.py`: shared request/response/error/protocol.
- `slick/backends/openai.py`, `anthropic.py`: optional SDK adapters.
- `slick/backends/cli.py`: native modern harness adapters.
- `slick/backends/legacy.py`: compatibility bridge for existing call-only models.
- `slick/models.py`: existing public compatibility surface; reuse/move subprocess helpers only when needed.
- `slick/tools.py`: function signature/schema binding, validated invocation and serialization.
- `slick/backends/agent.py`: bounded native Agent loop; provider-specific turns stay with their existing adapters.
- No `slick/runs.py` or AG2 dependency is introduced.
- `examples/` and `tests/`: grow with each deliverable.
- `README.md`, `CHANGELOG.md`, `pyproject.toml`, `slick/__init__.py`, `slick/cli.py`: update with the milestone that changes their behavior.

Avoid empty modules for later milestones.

**Non-goals for this delivery**

A scheduler, graph DSL, memory system, review/event-waiting abstraction, durable runtime, tool marketplace, persistent agent service, model router and automatic backend fallback are outside this scope. Tracing and streaming are deferred. Application code can use any external library; commonly repeated patterns can be optional addons later.

**Evidence and validation**

The current scope follows the user's latest corrections in this conversation and supersedes broader delivery suggestions in the earlier landscape research. Existing behavior comes from the current repository. External capabilities are documented in the [API/harness investigation](../../../outputs/models-and-harnesses-comparison.md) and [workflow landscape](../../../outputs/agentic-workflow-landscape-comparison.md). Backend API sketches here are proposals, not tested integrations.

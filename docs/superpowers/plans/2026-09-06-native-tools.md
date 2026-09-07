# Native Tools and Agent Backend Implementation Plan

> Superseded for current implementation by the [functional core](../specs/2026-09-07-functional-core.md). This document records earlier exploration; Agent, native tools and richer request protocols are deferred.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pass ordinary Python functions to a Slick Agent backed by OpenAI or Anthropic, including functions that themselves call models, agents or harnesses.

**Architecture:** derive tool definitions from signatures and docstrings, validate and execute their arguments locally, and translate calls/results through provider-native adapters. A small Agent backend owns the bounded model/tool loop. Application Python owns sequencing, concurrency, context, state and external waits.

**Tech Stack:** Python >=3.10, inspect, asyncio, Pydantic 2, optional OpenAI/Anthropic SDKs; existing pytest/ruff tooling. No agent-framework dependency.

**Spec:** [Backend toolbox and native tools design](../specs/2026-09-06-backend-toolbox.md). Read the entire spec before implementation.

**Prerequisite:** [Backend foundation](2026-09-06-backend-foundation.md), including typed sync/async prompt functions and both API adapters. This document is a proposed plan; no native tool feature is implemented yet.

## Global Constraints

- Preserve the foundation's public API and Python >=3.10 support.
- Ordinary typed functions need no tool decorator, wrapper class or registration service.
- Provider SDKs stay optional and lazily imported. Pydantic is already a dependency.
- Agent calls use `@prompt(backend=...)` and return the function's declared type.
- Never execute a name absent from the explicitly supplied tool list.
- Tool failures and invalid final output never automatically replay effectful work.
- No workflow DSL, concurrency scheduler, memory interface, review step, durable-execution integration, tracing or streaming scaffold.
- Harness tool transport is separate, optional future work; in-process tools require no MCP.

## Task 1: derive and invoke tools from Python functions

**Files:** create `slick/tools.py`, `tests/test_tools.py`; inspect `slick/prompts.py` for signature/annotation preservation.

**Produces:** internal tool specifications and invocation helpers. Keep these internal until a concrete public use requires exporting them.

- [ ] Write focused tests for ordinary functions, defaults, keyword-only arguments, bound methods, nested Pydantic types, async functions and decorated Slick functions. Reject positional-only parameters, variadic signatures, missing/unresolved parameter annotations and duplicate names before model execution. Missing return annotations may use JSON-compatible runtime serialization; reject a present but unsupported return annotation during construction.
- [ ] Include tests that unknown names, extra arguments and invalid values cannot invoke any function. Verify that all calls in a model-returned batch are validated before the first function runs.
- [ ] Run `.venv/bin/python -m pytest tests/test_tools.py -q` and confirm the new feature tests fail before implementation.
- [ ] Implement signature discovery using `inspect.signature` and resolved type hints. Store the original callable, name, description, generated argument model/schema and optional return adapter. Bind bound methods correctly and preserve Python defaults. Use a Pydantic argument model with extra fields forbidden; choose strict validation for ordinary primitive arguments while preserving supported JSON representations of declared structured types, and test that policy explicitly.
- [ ] Implement sync and async invocation helpers. Sync Agent execution rejects async-only tools before making a request; async execution awaits coroutine functions and offloads ordinary synchronous functions with `asyncio.to_thread`. Do not use ambient-loop detection or nest `asyncio.run`. Treat unusual synchronous callables that return awaitables as unsupported initially, with a clear error and cleanup of newly created coroutine objects.
- [ ] Validate annotated return values and serialize supported data using Pydantic; preserve string results as text. Include nested structured values and serialization failures in tests. Model-visible arguments/results are data, never executable Python source.
- [ ] Preserve tool name and chained cause on validation/execution failures. Applications can supply their own error-returning wrappers; do not invent automatic tool repair or retry behavior.
- [ ] Run `.venv/bin/python -m pytest tests/test_tools.py -q` and review only the task's changes.

## Task 2: support native provider turns with tools

**Files:** modify `slick/backends/openai.py`, `anthropic.py`; create private `slick/backends/_turns.py` only for types both adapters actually share; extend `tests/test_api_backends.py`.

**Consumes:** the existing Request/output contract and internal tool specifications.

**Produces:** private sync/async turn operations for tool-capable models. Do not expand the general Backend protocol: a harness that returns final text need not expose a model turn.

- [ ] Write offline SDK-fake tests for a first turn returning a tool call, a follow-up returning final text, multiple calls, malformed calls, refusal/incomplete output, and invocation-local state. Assert exact argument schemas and result/call correlation for both providers.
- [ ] Define only the shared data needed: a tool call with ID/name/raw argument data; a serialized result with matching ID; a turn carrying final Response or tool calls plus opaque continuation state. Preserve raw argument data until validation, so malformed JSON raises before dispatch.
- [ ] Implement private `_turn` and `_aturn` operations accepting Request, tool definitions, previous state/results and remaining timeout. Provider-specific continuation lives in the adapter and invocation state, never on the reusable backend object.
- [ ] Translate tools to OpenAI Responses function definitions (`name`, `description`, `parameters`) and Anthropic definitions (`name`, `description`, `input_schema`). Normalize schema dialects without mutating the original local schema. Enable strict provider modes only when they preserve the supported Python contract, including defaults.
- [ ] Preserve all necessary OpenAI response/reasoning items or supported response references, and Anthropic assistant content blocks/tool-use IDs. Send results in each provider's required role/envelope. Never rebuild the conversation from flattened final text.
- [ ] Preserve ordinary assistant text that accompanies calls as continuation context; it is not a completed Agent result while calls remain pending. Reuse the foundation's native final-output schema handling and refusal/incomplete checks. Fail clearly when the chosen provider/model cannot combine the requested tools and output contract.
- [ ] Reuse existing SDK lifecycle ownership. If clients remain per-turn initially, close each owned client; never close injected clients. Bound request timeout by the remaining Agent deadline without mutating shared backend configuration.
- [ ] Keep direct model `run`/`arun` behavior unchanged: no tools and no dispatch. Unexpected calls still raise. Run `.venv/bin/python -m pytest tests/test_api_backends.py -q`.

Official interfaces to verify against the selected SDK versions at implementation: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [Anthropic tool definitions](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools), [Anthropic tool results](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls).

## Task 3: implement the bounded Agent backend

**Files:** create `slick/backends/agent.py`, `tests/test_agent_backend.py`; export Agent from `slick/backends/__init__.py`.

**Public constructor:** `Agent(*, model, tools, max_steps=8, max_tool_calls=32, timeout=120)`.

**Produces:** a `kind="agent"` backend with `identity`, `run` and `arun`, using the foundation's Request/Response contract.

- [ ] Write a scripted fake-model test: request a tool, capture its serialized result, then return a typed final answer. Assert the function executes once and the response reaches the existing prompt validator. Add a no-tool final-answer case.
- [ ] Write tests for invalid model capability/configuration, prevalidated batches, multiple sequential calls, exceptions, exhausted step/call budgets, deadline expiry, async cancellation, and independent concurrent invocations of one configured Agent. Use controlled futures/events rather than slow wall-clock sleeps.
- [ ] Validate configuration and all tool definitions before execution. `max_steps` counts provider turns, including the final turn; `max_tool_calls` counts dispatched calls. Both must be positive integers and timeout must be positive. Require a tool-capable model; generic final-text backends do not satisfy this dependency.
- [ ] Implement the minimal loop: request turn, finish on final Response, otherwise validate the whole batch, check remaining budgets, invoke each requested function sequentially, submit results, repeat. Raise a BackendError on limit exhaustion. Never parse a provider transcript in Agent or automatically rerun a failed tool.
- [ ] Use `time.monotonic` and one deadline per Agent invocation. Pass remaining duration into provider requests; bound awaits with Python 3.10-compatible `asyncio.wait_for`. Check before and after synchronous tools. Document that cancellation/timeout cannot kill a running thread, interrupt arbitrary synchronous code or roll back side effects; no worker-process sandbox is introduced.
- [ ] Preserve native cancellation. A nested async Slick function propagates cancellation to its backend; nested agents retain their own limits. Do not claim a global token/step budget across independent nested agents.
- [ ] Include model configuration, tool names/schemas and Agent limits in non-secret identity. Do not hash or serialize closures, clients or function source. Agent caching stays disabled by default; any explicit opt-in has the existing cache's limits and cannot infer changes in external state or tool implementation.
- [ ] Verify agent final-output failures do not trigger a second execution and the modern decorator's default cache is disabled. Run `.venv/bin/python -m pytest tests/test_agent_backend.py tests/test_tools.py tests/test_api_backends.py -q`.

## Task 4: demonstrate one tool mechanism for ordinary and Slick functions

**Files:** add `examples/native_tools.py` and its prompt templates; extend `tests/test_agent_backend.py` and `tests/test_prompts.py`; update README and changelog.

- [ ] Add an example with an ordinary annotated lookup function passed in `tools=[lookup]` and an `@prompt(backend=Agent(...))` function callable from normal Python. Use application-supplied model IDs and clearly separate any real paid run from offline tests.
- [ ] Add an async decorated specialist with an explicit `template=` and a short tool-facing docstring. Pass that specialist directly in another Agent's tool list. Show that `await specialist(question)` invokes it directly and model selection invokes the same callable through the same validation/serialization path. No agent-as-tool wrapper or handoff API.
- [ ] Test this example with fake backends so no credentials, network or installed CLI is needed. Verify the specialist's actual name/signature/docstring survive decoration, typed values serialize correctly, and the outer Agent continues after the specialist returns.
- [ ] Document that changing OpenAI to Anthropic preserves Python tool definitions. Explain the process boundary for CLI harnesses: they keep their native/configured tools; exposing these host callables requires the separately deferred transport bridge.
- [ ] Keep any ordering, branches, `asyncio.gather`, context construction or external waits in ordinary application code. Do not add recipe frameworks or separate integration milestones.
- [ ] Run `.venv/bin/python -m pytest -q` and `.venv/bin/python -m ruff check slick tests`. Verify clean imports without provider extras and perform the foundation's packaging/type checks if public annotations or packaging changed.

## Acceptance

The same plain Python function works as a tool with both API providers. A decorated Slick function also works as a tool and remains directly callable by application code. Native provider continuation and call IDs survive multiple turns. Arguments validate before execution, failures do not replay effects, and execution limits have tested, documented cancellation semantics. All integration tests run offline; live-provider smoke checks are separate.

No AG2 dependency, special delegation wrapper, orchestration abstraction, memory subsystem, review step, durable runtime, event sink or streaming API is part of this milestone.

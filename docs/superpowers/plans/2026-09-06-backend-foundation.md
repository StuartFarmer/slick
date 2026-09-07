# Backend Foundation Implementation Plan

> Superseded for current implementation by the [functional core](../specs/2026-09-07-functional-core.md). This document records earlier exploration; Agent, native tools and richer request protocols are deferred.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ship a backend toolbox that supports typed sync/async functions using direct API models and existing CLI harnesses.

**Architecture:** independent backends implement a shared request/result protocol; Slick keeps prompt rendering and typed validation. Existing model APIs remain a compatibility surface. Native function tools and Slick's own bounded Agent loop follow in milestone B; tracing/streaming are deferred and application orchestration is outside scope.

**Tech Stack:** Python >=3.10, Jinja2, Pydantic 2, optional OpenAI/Anthropic SDKs, stdlib subprocess/asyncio; existing pytest/ruff tooling.

**Spec:** [Backend toolbox design](../specs/2026-09-06-backend-toolbox.md). Read the entire spec before implementation.

## Global Constraints

- Python >=3.10 remains supported.
- Jinja2 and Pydantic remain the only mandatory third-party dependencies.
- Provider SDKs are optional extras and are imported lazily; native Slick tools require no agent framework.
- The primary function API remains `@prompt`.
- Existing `model=`, `get_model()`, `Model.call()`, `Model.execute()`, CLI commands and template behavior remain available.
- Backend selection never silently changes local tool access or credentials.
- Library execution never prints model output to stdout; tracing and streaming are deferred.
- Modern backend= agent/harness work is never automatically rerun to repair its final representation.
- All interfaces shown below are proposed, not currently implemented.

Baseline at planning: `.venv/bin/python -m pytest -q` passes the existing 54 tests. This is a baseline only; no planned feature has been implemented.

## File structure

Create `slick/backends/{__init__,base,legacy,openai,anthropic,cli}.py` as their tasks become active. Modify `slick/prompts.py`, `slick/__init__.py`, `pyproject.toml`, README and changelog. Keep `slick/models.py` public and stable; extract helpers only when reuse is concrete. Put adapter tests in `tests/test_backends.py`, provider request tests in `tests/test_api_backends.py`, and process tests in `tests/test_cli_backends.py`. Add no empty follow-on modules.

This plan covers milestone A. Milestone B is [native tools and the Agent backend](2026-09-06-native-tools.md). An optional MCP transport can later expose those same tools to harnesses. Do not implement tracing, streaming, memory, orchestration or durable-execution features in this plan.

## Task 1: establish the backend boundary and legacy bridge

**Files:** create `slick/backends/base.py`, `legacy.py`, `__init__.py`, `tests/test_backends.py`; read `slick/models.py` without changing its behavior.

**Consumes:** existing `Model.call(prompt) -> str` and duck-typed legacy models with `backend` and `model` attributes.

**Produces:** `Request`, `Response`, `Backend`, `BackendError`, `LegacyBackend`.

- [ ] Add the contract types below. Use the complete `Backend` signatures and response definitions from the spec; export them from `slick.backends`.

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from pydantic import TypeAdapter

@dataclass(frozen=True)
class Request:
    prompt: str
    returns: Any = str

    @property
    def schema(self) -> dict[str, Any] | None:
        return None if self.returns is str else TypeAdapter(self.returns).json_schema()

@dataclass(frozen=True)
class Response:
    text: str
    usage: dict[str, int] | None = None
    raw: Any = field(default=None, repr=False, compare=False)

class BackendError(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason

class Backend(Protocol):
    kind: Literal["model", "agent", "harness"]

    def identity(self) -> dict[str, Any]: ...
    def run(self, request: Request) -> Response: ...
    async def arun(
        self, request: Request
    ) -> Response: ...
```

- [ ] Add/run the following tests before implementing the bridge.

```python
from slick.backends import Request, Response, LegacyBackend

def test_schema_retains_the_declared_output_contract():
    assert Request("question").schema is None
    assert Request("question", list[int]).schema["type"] == "array"

def test_legacy_bridge_does_not_change_submitted_text():
    class ExistingModel:
        backend = "test"
        model = "test-model"

        def call(self, text):
            assert text == "exact prompt"
            return "answer"

    backend = LegacyBackend(ExistingModel())
    assert backend.run(Request("exact prompt")) == Response("answer")
```

Run: `.venv/bin/python -m pytest tests/test_backends.py -q`. Expect bridge failure until implemented.

- [ ] Implement the compatibility adapter. Sync delegates exactly once to `call`. Identity contains legacy backend/model and an adapter tag; never serialize the underlying object.

```python
class LegacyBackend:
    kind = "harness"

    def __init__(self, model):
        self.model = model

    def identity(self):
        return {
            "adapter": "legacy",
            "backend": self.model.backend,
            "model": self.model.model,
        }

    def run(self, request):
        return Response(self.model.call(request.prompt))

    async def arun(self, request):
        raise BackendError(
            "unsupported",
            "This legacy model is synchronous; use a modern async backend.",
        )
```

This explicit async limitation avoids pretending that cancellation stops an unknown synchronous implementation. Existing synchronous usage is preserved. Built-in modern CLI adapters gain native async in Task 4.

- [ ] Run `.venv/bin/python -m pytest tests/test_models.py tests/test_backends.py -q`.
- [ ] Review and commit only the task's files with message `feat: introduce the backend request and response contract`.

## Task 2: connect typed prompt functions to sync and async backends

**Files:** modify `slick/prompts.py`, `slick/__init__.py`; extend `tests/test_prompts.py` and `tests/test_backends.py`.

**Consumes:** Task 1 contract and existing Jinja/parser/cache code.

**Produces:** `prompt(..., backend: Backend | None = None)`, async declarations, original return-type propagation, modern default policies and compatibility dispatch.

- [ ] Add a deterministic backend in `tests/test_backends.py` and tests for public function behavior.

```python
import asyncio
from slick import prompt
from slick.backends import Request, Response

class RecordingBackend:
    kind = "model"

    def __init__(self):
        self.requests = []

    def identity(self):
        return {"adapter": "recording", "model": "fake"}

    def run(self, request):
        self.requests.append(request)
        return Response("7")

    async def arun(self, request):
        return self.run(request)

def test_sync_and_async_keep_the_same_python_contract(tmp_path):
    backend = RecordingBackend()

    @prompt(backend=backend, cache=False, log_dir=tmp_path)
    def count(text: str) -> int:
        """Count items in {{ text }}."""

    @prompt(backend=backend, cache=False, log_dir=tmp_path)
    async def acount(text: str) -> int:
        """Count items in {{ text }}."""

    assert count("apples") == 7
    assert asyncio.run(acount("apples")) == 7
    assert all(request.returns is int for request in backend.requests)
```

- [ ] Run that test; expect unsupported decorator keyword before changes.
- [ ] Split argument binding/template rendering from invoking the original function body. The sync wrapper invokes the body synchronously; the async wrapper awaits it, then passes the same rendered text/return type to `backend.arun`. Keep one shared validator. Selection is based on `inspect.iscoroutinefunction(fn)`, never whether an event loop happens to exist.

Core dispatch:

```python
request = Request(rendered_text, returns)
response = backend.run(request)       # sync wrapper
response = await backend.arun(request)  # async wrapper
value = response.text if returns is str else parser.parse(response.text)
```

These lines belong in separate wrappers, not in one mixed sync function.

- [ ] Add declaration-time validation: backend and model are mutually exclusive; modern backend declarations reject positive repair counts for agent/harness kinds. Preserve legacy default resolution and legacy per-call model overrides. Do not consume a function argument named backend.
- [ ] Implement modern default policy resolution as a small helper, with `cache`/`max_repairs` changing to optional decorator settings. Compatibility calls select their old defaults.

```python
def _policies(kind, cache, max_repairs, *, legacy=False):
    inference = kind == "model"
    use_cache = (legacy or inference) if cache is None else cache
    repairs = (1 if legacy or inference else 0) if max_repairs is None else max_repairs
    if repairs < 0:
        raise PromptError("max_repairs must be non-negative")
    if not legacy and not inference and repairs:
        raise PromptError("Agent and harness tasks cannot be rerun for output repair")
    return use_cache, repairs
```

- [ ] Preserve existing render/schema text and output-file behavior. Add no-model-call tests for sync `.render()` and awaited async `.render()`, including an async body returning computed context.
- [ ] For modern requests, hash canonical JSON containing prompt, original schema, and `backend.identity()`; adapter identity includes output mode and configured output-affecting options. Retain legacy cache identity when using legacy model calls. Use a temporary sibling file and atomic replacement for accepted response writes; do not publish a cache result before local validation succeeds.
- [ ] Test that invalid harness output invokes the harness once, while a model permits its configured repair; refusal/incomplete `BackendError` bypasses repair. Change an output-affecting option/schema and verify it misses cache. Add simultaneous async calls with distinct inputs and verify independent results.
- [ ] Add typing overloads preserving parameters and sync/async returned types using `ParamSpec` and `TypeVar`. Do not mask all decorated functions as `Callable[..., Any]`. Static checking of examples belongs in Task 5.
- [ ] Run `.venv/bin/python -m pytest tests/test_prompts.py tests/test_backends.py tests/test_models.py -q`.
- [ ] Review and commit with message `feat: support backend-backed sync and async prompt functions`.

## Task 3: add optional OpenAI and Anthropic model backends

**Files:** create `slick/backends/openai.py`, `anthropic.py`; update `backends/__init__.py`, `pyproject.toml`, `poetry.lock`; create `tests/test_api_backends.py`.

**Consumes:** `Request`, `Response`, `BackendError`.

**Produces:** `OpenAI` and `Anthropic`, both with `kind="model"`, `identity/run/arun`; optional `slick-ai[openai]` and `slick-ai[anthropic]` extras.

Constructor contract for both adapters: keyword-only `model: str`, `timeout: float = 60`, `max_output_tokens: int = 2048`, `structured: Literal["native", "prompted"] = "native"`, `client=None`, `async_client=None`. Anthropic maps the common budget to `max_tokens`. Native provider overrides belong in explicit adapter configuration; reject contradictory overrides of prompt/schema/tool fields.

- [ ] Select released SDK versions that support Python 3.10 and the documented request shapes. Add only optional dependencies and their extras. Regenerate the lock with project tooling; verify a clean base install does not bring in either SDK. Record the tested versions in the changelog, not inferred main-branch compatibility.
- [ ] Write SDK-independent recording clients, and a request/response test.

```python
from types import SimpleNamespace
from slick.backends import OpenAI, Request

def test_openai_passes_native_schema_and_preserves_usage():
    seen = {}
    raw = SimpleNamespace(
        status="completed",
        output=[],
        output_text='{"value": 7}',
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )
    def create(**kwargs):
        seen.update(kwargs)
        return raw
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    backend = OpenAI(model="test-model", client=client)
    response = backend.run(Request("count", int))
    assert seen["model"] == "test-model"
    assert seen["text"]["format"]["type"] == "json_schema"
    assert response.text == "7"
    assert response.usage["input_tokens"] == 3
    assert response.raw is raw
```

- [ ] Run `.venv/bin/python -m pytest tests/test_api_backends.py -q`; expect missing adapter failure.
- [ ] Implement native request construction using official SDKs with lazy imports. Owned clients are opened/closed per call; injected clients are never closed. Use native async client methods in `arun`.

Wire shapes:

```python
# OpenAI Responses: native schema mode.
parameters = {
    "model": self.model,
    "input": request.prompt,
    "max_output_tokens": self.max_output_tokens,
}
if wire_schema is not None:
    parameters["text"] = {
        "format": {
            "type": "json_schema",
            "name": "slick_output",
            "strict": True,
            "schema": wire_schema,
        }
    }
raw = client.responses.create(**parameters)

# Anthropic Messages: native schema mode.
parameters = {
    "model": self.model,
    "max_tokens": self.max_output_tokens,
    "messages": [{"role": "user", "content": request.prompt}],
}
if wire_schema is not None:
    parameters["output_config"] = {
        "format": {"type": "json_schema", "schema": wire_schema}
    }
raw = client.messages.create(**parameters)
```

`wire_schema` is derived from `request.schema` by an adapter-owned schema encoder. It is None for plain strings or explicit prompted mode. The encoder returns both the wire schema and whether final JSON must be unwrapped from a root `value` field. Keep root definitions accessible when wrapping references. Use documented SDK schema conversion where suitable; fail on unsupported constraints rather than inventing a general schema converter.

- [ ] Implement completion checks before extracting text. OpenAI must have a completed status and no refusal/unhandled tool-call output. Anthropic must have a final message stop reason (normally end_turn/stop_sequence); refusal, max_tokens, tool_use and pause_turn do not count as a completed no-tool inference call. Preserve the native object in `raw` and chain SDK exceptions in `BackendError`.
- [ ] Add equivalent Anthropic fake-client tests and native async tests; add refusal, truncation, list/scalar/nested-model outputs, schema references, unsupported schema, explicit prompted mode, client ownership and missing-extra tests. Ensure no API keys or real network are used.
- [ ] Compare emitted schemas with the selected SDK's supported dialect in offline request tests. Local Pydantic validation must still reject invalid custom constraints.
- [ ] Run `.venv/bin/python -m pytest tests/test_api_backends.py tests/test_backends.py tests/test_prompts.py -q`.
- [ ] Review and commit with message `feat: add optional OpenAI and Anthropic model backends`.

## Task 4: expose modern harness backends with native schemas and cancellation

**Files:** create `slick/backends/cli.py`, `tests/test_cli_backends.py`; update backend exports. Reuse existing builders/parsers from `slick/models.py` where behavior matches, without changing legacy classes' defaults.

**Consumes:** common contract and existing Claude/Codex process behavior.

**Produces:** `Codex`, `ClaudeCode`, both with `kind="harness"`, native sync/async execution and schema support.

Constructors:
- `Codex(*, workdir=None, model=None, sandbox="read-only", timeout=3600, command="codex")`.
- `ClaudeCode(*, workdir=None, model=None, allowed_tools=(), permission_mode="dontAsk", timeout=3600, command="claude")`.

Reject a generic `sandbox` argument on ClaudeCode. Use the CLI's documented tool availability controls plus approval configuration; allowed-tools approval alone must not be advertised as a tool inventory restriction. Limit the exposed tool inventory when an explicit list is supplied. Confirm the exact flags against the selected CLI versions and retain native execution evidence. [Claude CLI reference](https://code.claude.com/docs/en/cli-reference)

- [ ] Build a real fake CLI script in the test temporary directory that records argv/stdin and writes fixture output. Drive it through the actual subprocess path. Verify temporary schema contents and argument passing. Do not mock only the command-builder helper.
- [ ] Add process cancellation verification with a fake CLI that writes its PID/readiness marker and waits. Cancel only after the marker exists, then require the child to exit.

```python
# The fake process script used by the test.
import os
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
```

The test launches this script through the backend's process helper, waits asynchronously for the marker with a short test deadline, cancels the task, checks `CancelledError`, and asserts the process has exited. Add a spawned-child variant for process-group cleanup on POSIX. Keep cleanup in a test finally block.

- [ ] Implement Codex native output using a per-run schema file and `--output-schema`; parse JSONL final messages. Implement ClaudeCode JSON output, `--json-schema`, and extraction from `structured_output` for typed results, `result` for text. A missing typed payload is an error, not an empty success.
- [ ] Factor process execution into sync and async helpers with equivalent request/response behavior. Async uses `create_subprocess_exec`, communicates through stdin, and performs cleanup on timeout/cancellation.

Cancellation shape:

```python
process = await asyncio.create_subprocess_exec(*argv, **process_options)
try:
    stdout, stderr = await asyncio.wait_for(
        process.communicate(request.prompt.encode()), timeout=self.timeout
    )
except (asyncio.TimeoutError, asyncio.CancelledError):
    await stop_owned_process(process)
    raise
```

`process_options` supplies pipe descriptors, workdir and a new process session on POSIX. `stop_owned_process` is an adapter-private helper implemented in this task: signal the owned process/group, wait a short bounded grace period, then kill and reap. Map timeout to `BackendError("timeout", ...)` after cleanup, and let cancellation propagate. On Windows use supported process controls and document/test any child-tree limitation.

- [ ] Test CLI error, missing executable, malformed output, refusal/error result, permission settings, schema cleanup, timeout and cancellation. Ensure the modern Claude command never inherits the legacy unconditional bypass flag.
- [ ] Verify legacy `Model.call/execute` tests still pass; new harness errors are not repaired/replayed by modern prompts.
- [ ] Run `.venv/bin/python -m pytest tests/test_cli_backends.py tests/test_models.py tests/test_backends.py -q`.
- [ ] Review and commit with message `feat: add typed native harness backends and async cancellation`.

## Task 5: ship the toolbox example, compatibility documentation and validation

**Files:** create `examples/toolbox.py`, `examples/mixed_workflow.py`, their prompt templates and `tests/test_mixed_workflow.py`; update README, changelog, `slick/__init__.py`; inspect CLI compatibility.

**Consumes:** Tasks 1–4 backend classes and decorator.

**Produces:** a working mixed backend example, offline integration verification, accurate types and installation docs.

- [ ] Define application data types in the example: `Finding(description: str)` and `Report(summary: str, findings: list[Finding])`. Use explicit selected model IDs from environment/configuration in the runnable example; fail with a useful setup message when missing.
- [ ] Implement a three-stage async workflow: extract question from input through OpenAI, investigate through Codex, summarize through Anthropic. All three prompt declarations use `async def`; orchestration is normal calls and await.

```python
async def produce_report(document: str) -> Report:
    question = await extract_question(document)
    findings = await investigate(question)
    return await summarize_findings(findings)
```

In the fake-backed test, replace backend objects before constructing the decorated functions (or use a plain function that constructs the example functions from supplied backends). Each fake records its request and returns deterministic JSON; assert all three backend identities ran and the final value is a Report. Do not add a production registry solely for testing.

- [ ] Verify that the async functions compose with ordinary `asyncio.gather`; document it briefly within the example. This is a compatibility check, not a Slick concurrency feature or separate recipe milestone.
- [ ] Document optional extras, sync/async behavior, modern backend configuration, native vs prompted schemas, legacy `model=`, cache defaults, validation/repair failures and subprocess cancellation limits. Explain that new backend declarations preserve typed values while execution authority differs.
- [ ] Verify package import and CLI legacy commands without provider extras or keys. Do not add API model discovery or persistent CLI configuration.
- [ ] Add dev-only static type checking if needed to assert a sync prompt returns its annotation and an async prompt is awaitable with that annotation. Keep it out of runtime dependencies.
- [ ] Run the complete test suite, lint, and configured type checks. Run packaging/build checks with the existing Poetry workflow. Use a temporary environment to verify base-only and per-extra installs against the lockfile; report unavailable network/package checks explicitly.
- [ ] Inspect the final diff for accidental model calls, unconditional SDK imports, secret-bearing logging, changes to original templates and new global configuration layers.
- [ ] Review and commit with message `docs: demonstrate the backend toolbox and mixed workflows`.

## Milestone acceptance and follow-ons

Milestone A is complete when its mixed workflow is verified offline, old usage still passes, both native API adapters and CLI adapters have request/lifecycle coverage, and no model calls occur during ordinary tests/imports. Real-provider smoke runs are separate, explicit checks using configured credentials and temporary read-only workspaces.

Next, implement [native tools and the Agent backend](2026-09-06-native-tools.md): ordinary functions supply schemas and implementations; Slick runs a small bounded loop using OpenAI/Anthropic native tool calls. A decorated Slick function participates in the same tool mechanism. No AG2 runtime is required.

An optional MCP bridge may later expose the same tools to CLI harnesses. Tracing/streaming are deferred. Memory, arbitrary external waits, coordination and durable execution remain application Python or other libraries. Do not create event sinks or reserve interfaces for deferred work.

## Self-review

- [x] Each milestone-A spec requirement maps to Tasks 1–5.
- [x] Backend protocol, Request/Response names, run/arun methods and kind values match the spec.
- [x] Legacy compatibility and modern defaults are distinguished.
- [x] Optional dependencies and no-network tests are explicit.
- [x] Cancellation, output repair and cache behavior have targeted verification.
- [x] Native tools/Agent are the next deliverable; external orchestration and deferred observability are absent from the active implementation scope.


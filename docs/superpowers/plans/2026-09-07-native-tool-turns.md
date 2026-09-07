# Native Tool Turns Implementation Plan

Execution completed 2026-09-07. The original procedural checklist is retained below
as planning history; completion evidence appears at the end.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax. Delegation requires explicit authorization; do not infer it from this document.

**Goal:** let Python applications send callable tool definitions, receive one native model turn, and supply correlated results without an implicit execution loop.

**Architecture:** immutable record shells describe user messages, model turns and tool results. Small provider codecs translate those records; existing API backend objects retain client configuration and perform one request. The application owns history and dispatch.

**Tech Stack:** Python >=3.10, Pydantic >=2.0, stdlib dataclasses/json, existing optional OpenAI/Anthropic SDKs, pytest.

**Spec:** [Coding harness design](../specs/2026-09-07-coding-harness-design.md), sections 2–3, N01–N05. Also preserve [the callable-tool contract](../specs/2026-09-07-callable-tools.md).

## Global Constraints

- Python >=3.10; core remains compatible with Pydantic >=2.0.
- No new mandatory Slick dependency; existing call/acall and Tool contracts remain unchanged.
- All application code and Jinja templates live under examples/, never slick/.
- Core tests run without provider SDKs or Textual; extra-dependent test jobs install their dependencies explicitly.
- Automated tests use scripted models and local temporary workspaces, never paid endpoints or the developer's working tree.
- No core history, tool execution, repair, persistence, caching, or thread scheduling is introduced.

Status: implemented after the user requested execution. Changes remain uncommitted.
The detailed steps below preserve the original plan; actual execution and validation
are recorded at the end of this document.

## Files and interfaces

| File | Responsibility |
| --- | --- |
| Create slick/turns.py | Records, strict argument decoding, history invariants |
| Create slick/_openai_turns.py | Pure Responses schema/history/output conversion |
| Create slick/_anthropic_turns.py | Pure Messages schema/history/output conversion |
| Modify slick/backends.py | Async aturn method on each existing API backend |
| Create tests/test_turns.py | Record, argument, and history contracts |
| Create tests/test_native_turns.py | SDK-free request/parsing/dispatch-boundary checks |
| Modify tests/test_sdk_transport.py | Real optional SDK round trips over mock HTTP |
| Modify README.md, CHANGELOG.md | One-turn example and scope |

No backend package migration or generic Agent/runner class. Records stay in
slick.turns rather than expanding the top-level slick export list.

## Task 1: Records, strict argument decoding, and history validation

**Files:** slick/turns.py; tests/test_turns.py.

**Produces:** the four records plus ToolCall exactly as defined in the spec;
`decode_arguments(raw: str | dict) -> tuple[dict | None, str | None]`;
`validate_history(history: list, *, provider: str, model: str) -> None`.
Validation errors are ValueError with call/field context; backend boundaries wrap
these as BackendError. Frozen dataclasses do not imply deeply immutable payloads:
copy JSON structures at serialization/parsing boundaries.

- Write tests for required fields, independent list data, strict JSON object
  arguments, duplicate keys/NaN/Infinity/fenced JSON/arrays, valid empty objects,
  interleaved user messages, foreign provider/model, unknown/duplicate result IDs,
  duplicate call IDs, unresolved calls, and valid multi-call groups.

```python
from slick.turns import ModelTurn, ToolCall, ToolResult, UserMessage, validate_history


def test_history_requires_results_before_followup():
    import pytest
    turn = ModelTurn(
        provider="openai", model="test-model", text="",
        tool_calls=[ToolCall("call_1", "read_file", {"path": "a.py"})],
        items=[], stop_reason="tool_calls",
    )
    history = [UserMessage("Read a.py"), turn]
    with pytest.raises(ValueError, match="call_1"):
        validate_history(history + [UserMessage("continue")],
                         provider="openai", model="test-model")
    validate_history(history + [ToolResult("call_1", "contents")],
                     provider="openai", model="test-model")
```

- Run `python -m pytest tests/test_turns.py -v`; confirm missing implementation
  failures. Implement a single forward scan with pending/seen ID sets; require
  complete groups at request boundaries and reject unsupported record types.

```python
# Public result shape from the strict decoder, not a permissive prompt parser.
assert decode_arguments('{"limit": 2}') == ({"limit": 2}, None)
assert decode_arguments('{"limit": 1, "limit": 2}')[0] is None
```

- Decode with json.loads object_pairs_hook rejecting duplicate keys and
  parse_constant rejecting nonfinite constants; reject overflowed nonfinite floats
  too, including in nested objects. Dict input is checked/copied through strict
  JSON encoding and decoding. Never call parse() or eval().
- Run the new tests and `tests/test_tools.py tests/test_tool_helpers.py`.
  Review that no callable runs and no stored record is mutated.

## Task 2: OpenAI native turns

**Files:** slick/_openai_turns.py; slick/backends.py; tests/test_native_turns.py.

**Consumes:** prepare_tools and Task 1 records/validators.
**Produces:** `tool_definitions(prepared: dict) -> list[dict]`,
`encode_history(history: list) -> list[dict]`,
`decode_turn(response: dict, *, model: str) -> ModelTurn` in slick._openai_turns;
`OpenAI.aturn(history, *, tools: list, instructions: str = "") -> ModelTurn`.

- Write codec tests with literal response dictionaries. Assert the entire
  outgoing definitions and follow-up call IDs, retained reasoning items, optional
  arguments unchanged, no mutation, malformed arguments retained as failed calls,
  and rejection of incomplete/refusal/unknown output types. Include two calls.

```python
from slick._openai_turns import encode_history
from slick.turns import ModelTurn, ToolCall, ToolResult, UserMessage


def test_openai_followup_keeps_provider_items():
    items = [
        {"type": "reasoning", "id": "r1", "summary": [], "encrypted_content": "opaque"},
        {"type": "function_call", "id": "fc1", "call_id": "c1",
         "name": "read", "arguments": '{"key":"x"}', "status": "completed"},
    ]
    turn = ModelTurn("openai", "test-model", "", [ToolCall("c1", "read", {"key": "x"})],
                     items, "tool_calls")
    payload = encode_history([UserMessage("read x"), turn, ToolResult("c1", "found")])
    assert payload[1:3] == items
    assert payload[-1] == {"type": "function_call_output", "call_id": "c1", "output": "found"}
```

- Run `python -m pytest tests/test_native_turns.py -k openai -v`; observe red.
- Implement definition mapping as Responses function objects with name,
  description, parameters, strict=False. Preserve items from the JSON provider
  response, extracting text only for the display field; no reasoning extraction
  into prompt text. Validate status before accepting any calls.
- Implement aturn using prepare_tools, validate_history, then existing _client.
  Send model, input, instructions (omit if empty), tools, max_output_tokens,
  store=False, and parallel_tool_calls=False when tools are present. Request the
  compatible reasoning.encrypted_content include for stateless continuation.
  Convert SDK responses with model_dump(mode='json') before the pure decoder.
  Request construction and decoding stay outside transport-specific helper logic.
- Add a fake async SDK client returning real-shaped response dictionaries
  through a tiny model_dump wrapper. Record request bodies; assert a failing
  tool definition results in zero requests and zero tool executions.
- Verify ownership, timeout, max_retries, cancellation, no import-time SDK
  loading, and original acall(text) request bodies remain unchanged.
- Run `python -m pytest tests/test_turns.py tests/test_native_turns.py tests/test_api_backends.py`.

## Task 3: Anthropic native turns

**Files:** slick/_anthropic_turns.py; slick/backends.py; tests/test_native_turns.py.

**Consumes:** Task 1 records/validators and existing _client.
**Produces:** `tool_definitions(prepared: dict) -> list[dict]`,
`encode_history(history: list) -> list[dict]`,
`decode_turn(response: dict, *, model: str) -> ModelTurn` in slick._anthropic_turns;
`Anthropic.aturn(history, *, tools: list, instructions: str = "") -> ModelTurn`.

- Write literal Messages envelope tests for input_schema, system instructions,
  assistant content, consecutive results grouped into one user message, is_error,
  preserved thinking signatures/redacted blocks, malformed calls, multiple calls,
  and tool_use/end_turn/max_tokens/refusal/unsupported stop conditions.

```python
from slick._anthropic_turns import encode_history
from slick.turns import ModelTurn, ToolCall, ToolResult, UserMessage


def test_anthropic_groups_all_results_before_next_message():
    calls = [ToolCall("a", "read", {"key": "x"}), ToolCall("b", "read", {"key": "y"})]
    items = [{"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
             for call in calls]
    turn = ModelTurn("anthropic", "test-model", "", calls, items, "tool_calls")
    history = [UserMessage("read"), turn, ToolResult("a", "x"), ToolResult("b", "missing", True)]
    messages = encode_history(history)
    assert messages[-1] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "x", "is_error": False},
        {"type": "tool_result", "tool_use_id": "b", "content": "missing", "is_error": True},
    ]}
```

- Run `python -m pytest tests/test_native_turns.py -k anthropic -v`; observe red.
- Implement native tool definitions using name/description/input_schema,
  clone all content, and emit tool results adjacent to the matching assistant turn.
  Do not enable additional thinking modes or reconstruct signatures.
- Implement aturn with model, messages, max_tokens, optional system, tools, and
  tool_choice auto/disable_parallel_tool_use when tools are present. Omit controls
  with no tools. Reuse client ownership/settings; do not alter plain text methods.
- Add injected-client tests: no request on invalid history/definitions, errors
  preserve causes, cancellation propagates, SDK clients remain owned correctly.
- Run `python -m pytest tests/test_turns.py tests/test_native_turns.py tests/test_api_backends.py`.

## Task 4: Real SDK transport, executable documentation, compatibility gate

**Files:** tests/test_sdk_transport.py; README.md; CHANGELOG.md.
**Produces:** verified tool request/response serialization through installed SDKs,
and an example that an application can compose without invoking a hidden loop.

- Extend the existing httpx.MockTransport pattern to serve two responses per
  provider: first one tool call, then final text. Use fully shaped provider fixtures
  (IDs, status/stop reason, usage, output/content). Assert the second HTTP request
  contains the original call plus exactly one correlated result. No test contacts
  a real endpoint or needs a real key.
- Cover provider model_dump behavior at SDK version floors and current
  constrained versions. Keep SDK-free coverage in tests/test_native_turns.py;
  optional imports must not skip the entire functional acceptance suite.
- Add a small documentation example using records and an explicit Python loop:

```python
history = [UserMessage("Look up the introduction.")]
prepared = prepare_tools([documents.read])
while True:
    turn = await backend.aturn(history, tools=list(prepared.values()))
    history.append(turn)
    if not turn.tool_calls:
        answer = turn.text
        break
    for call in turn.tool_calls:
        if call.argument_error or call.name not in prepared:
            history.append(ToolResult(call.id, call.argument_error or "Unknown tool", True))
            continue
        try:
            output = await prepared[call.name].ainvoke(call.arguments)
        except ToolError as error:
            history.append(ToolResult(call.id, str(error), True))
        else:
            history.append(ToolResult(call.id, output))
```

  Label this a minimal protocol example; the coding harness adds budgets,
  cancellation bookkeeping, command decisions and verification. Bind documents
  and backend in the README example using the existing Documents/OpenAI examples.
- Document async-only native turns, provider-bound replay history, unsupported
  OpenRouter/CLI use here, explicit non-strict provider schemas with local strict
  validation, and no change to text rendering/call behavior.
- Run the complete pytest suite with and without optional SDKs, then with
  Python 3.10/Pydantic 2.0. Run Ruff and targeted mypy as in prior work:

```bash
python -m pytest
python -m ruff check slick tests examples
python -m mypy slick/turns.py slick/_openai_turns.py slick/_anthropic_turns.py slick/backends.py --follow-imports=silent
python -m build --wheel --no-isolation --outdir /tmp/slick-native-turns-dist
```

- Inspect the wheel for new core modules and unchanged optional-dependency
  boundaries. Record exact checks/results in this plan when execution happens.
  Hand off to the application plan only after native turns and existing APIs pass.

## Review checklist

- N01 preparation happens before I/O; no function execution in the backend.
- N02 complete history validation precedes request construction.
- N03 every provider continuation item and call ID survives replay.
- N04 partial/incomplete outputs never dispatch, malformed arguments remain recoverable.
- N05 legacy APIs, optional imports, clients and cancellation remain compatible.
- No automatic tool loop, no prompts, and no session store were added to core.


## Execution record — 2026-09-07

- [x] Task 1: native records, strict JSON argument decoding and complete-history validation.
- [x] Task 2: OpenAI Responses one-turn transport and preserved continuation items.
- [x] Task 3: Anthropic Messages one-turn transport and preserved content blocks.
- [x] Task 4: real SDK transports over mock HTTP, documentation and compatibility gate.

Implemented in slick/turns.py, slick/_openai_turns.py, slick/_anthropic_turns.py and
slick/backends.py. No tool execution, app state, prompts or automatic loop entered
core. Existing text methods retain their behavior. Records are intentionally shallow
frozen dataclasses; the application preserves provider items without mutation.

Validation completed without paid endpoints:

| Check | Result |
| --- | --- |
| Final complete suite, Python 3.14, current SDKs and Textual installed | 593 passed |
| Final complete suite, Python 3.10 / Pydantic 2.0, no SDKs or Textual | 565 passed, 2 optional modules skipped |
| Native records/adapters/HTTP transport, Python 3.10 / Pydantic 2.0 / OpenAI 2.0.0 / Anthropic 0.50.0 | 100 passed |
| Earlier focused native + callable-tool compatibility suite, current SDKs and SDK floors | 320 passed in each environment |
| Ruff across slick/tests/examples | Passed |
| mypy on the four changed native/backend modules | Passed |
| Wheel build and inspection | Native modules included; examples and Textual excluded |
| Wheel imported from an isolated target with the minimal environment | No SDK or Textual imports |

The README contains a bounded application dispatch example. The full coding harness
provides execution recovery and cancellation rather than putting them in transport.
No commits or publishing were performed.

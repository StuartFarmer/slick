# Coding Harness Application Implementation Plan

Execution completed 2026-09-07. The original procedural checklist is retained below
as planning history; completion evidence appears at the end.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax. Delegation requires explicit authorization; do not infer it from this document.

**Goal:** ship an independently runnable coding-agent example with real file edits, bounded command execution, test-driven repair, and a simple responsive TUI.

**Architecture:** CodingAgent is an ordinary stateful Python object over native backend turns and prepared bound tools. Workspace/process/verification helpers own effects. Textual displays application events and supplies command decisions; the same agent runs headlessly.

**Tech Stack:** Python >=3.10, Pydantic >=2.0, Jinja, asyncio/subprocess, Git, rg, Textual 8.2.8 as an example-only dependency, pytest and Textual Pilot.

**Spec:** [Coding harness design](../specs/2026-09-07-coding-harness-design.md), sections 4–10.
**Prerequisite:** [Native tool turns](2026-09-07-native-tool-turns.md), completed and verified first.

## Global Constraints

- Python >=3.10; core remains compatible with Pydantic >=2.0.
- No new mandatory Slick dependency; existing call/acall and Tool contracts remain unchanged.
- All application code and Jinja templates live under examples/, never slick/.
- Textual is example-only, pinned to 8.2.8 in examples/coding_harness/requirements.txt.
- Core tests run without provider SDKs or Textual; extra-dependent test jobs install their dependencies explicitly.
- Automated tests use scripted models and local temporary workspaces, never paid endpoints or the developer's working tree.
- No core history, tool execution, repair, persistence, caching, or thread scheduling is introduced.
- The first command-running application supports macOS and Linux; other platforms fail startup clearly before work begins.

Status: implemented after the user requested execution. Changes remain uncommitted.
The detailed steps below preserve the original plan; actual execution and validation
are recorded at the end of this document.

## File map

Create the application files and templates listed in design section 4. Tests:

| File | Coverage |
| --- | --- |
| tests/coding_harness/conftest.py | Temporary Git repo and workspace fixtures, no global environment mutation |
| tests/coding_harness/test_state.py | Limits/config/record validation |
| tests/coding_harness/test_process.py | Real subprocess output, timeout, cancellation and decisions |
| tests/coding_harness/test_workspace.py | Discovery, UTF-8 reads, precise edits, paths and diffs |
| tests/coding_harness/test_verification.py | Real checks, baseline and stale-workspace results |
| tests/coding_harness/test_agent.py | Native-turn dispatch, repair, budgets, failure/cancellation history |
| tests/coding_harness/test_context.py | Templates, skills, size thresholds and compaction |
| tests/coding_harness/test_session.py | Explicit safe JSON save/resume, no action replay |
| tests/coding_harness/test_cli.py | Offline repair demo, flags and exit behavior |
| tests/coding_harness/test_tui.py | Headless Pilot interactions, real effects and cancellation |

Modify examples/README.md, repository README.md and CHANGELOG.md with the runnable
entry point and scope. Do not add a `slick agent` command or package the examples
into the core wheel. New test directories need no runtime package initialization.

## Task 1: Application records, configuration, and subprocess boundary

**Files:** create examples/coding_harness/{__init__,state,process}.py;
create tests/coding_harness/{test_state,test_process}.py.

**Produces:** Pydantic config/result records from design sections 5–6;
`load_config(path: Path | None) -> HarnessConfig`;
`async run_process(argv: list[str], cwd: Path, timeout: int, *, output_limit: int = 8000) -> CommandResult`.

Also define in state.py:
- `CommandRequest(argv: list[str], cwd: str, timeout: int)`.
- `CommandDecision = Literal['once', 'session', 'deny']`.
- `HarnessEvent(kind: str, data: dict)` as an application-local dataclass.
- `Verification(results: list[CheckResult], fingerprint: str, passed: bool, stable: bool)`.
- `SessionState` with active history, archived histories, current task, counters,
  running flag, baseline, last_result, edit ledger and provider/workspace metadata.
  Keep native records as objects in memory; session.py owns explicit serialization.

Config uses extra='forbid', factories for mutable defaults, the exact defaults in
the spec, and positive field constraints. Check and Limits are validated models;
HarnessConfig defaults to no checks/skills and default Limits. Config paths are
explicit, and load_config(None) performs no filesystem discovery.

- Write config tests for defaults, unknown keys, duplicate checks, empty argv,
  negative/boolean limits and inverted context thresholds. Write subprocess tests
  using sys.executable, not shell commands or external services.

```python
import asyncio
import sys
from examples.coding_harness.process import run_process


def test_nonzero_exit_is_an_observation(tmp_path):
    result = asyncio.run(run_process(
        [sys.executable, "-c", "import sys; print('failure'); sys.exit(3)"],
        tmp_path, 5,
    ))
    assert result.exit_code == 3
    assert result.stdout.strip() == "failure"
    assert not result.timed_out
```

- Run `python -m pytest tests/coding_harness/test_state.py tests/coding_harness/test_process.py -v`;
  observe missing-module failures before writing the implementation.
- Implement bounded concurrent drains using incremental UTF-8 decoding with
  replacement for invalid command-output bytes. Cap retained text while draining
  all pipes. Validate argv/timeout and platform before process creation. Use
  start_new_session=True and stdin=DEVNULL. Factor cleanup into a small shared
  helper used by timeout and cancellation, with a five-second cleanup deadline.
- Test interleaved large stdout/stderr without deadlock, Unicode boundaries,
  timeout result, cancellation propagation, child process reaping, an escaped
  descendant holding a pipe, executable-not-found, and no startup on invalid args.
  Use explicit test-owned process cleanup in finally blocks so failures do not leak.
- Run both test modules. Do not reuse Model._arun_process directly: it captures
  all output and raises on nonzero exits, while commands require bounded observations.

## Task 2: Workspace discovery, precise edits, and command decisions

**Files:** examples/coding_harness/workspace.py;
tests/coding_harness/{conftest,test_workspace,test_process}.py.

**Consumes:** Task 1 records/run_process.
**Produces:** `Workspace(root: Path, *, checks: list[Check], decide, command_timeout: int = 120)`;
`async Workspace.initialize() -> None`; all seven tools and records in design
section 5; `async Workspace.fingerprint() -> str`;
`async Workspace.changed_paths() -> list[str]` relative to its initial catalog/digests.
`decide` is `Callable[[CommandRequest], Awaitable[CommandDecision]]`.

Workspace.initialize validates root is the Git worktree root, checks Git/rg, and
records HEAD (None allowed for an unborn repo), initial status and digests. It
creates no branches or files. Git catalog results are filtered by the same path
policy as file tools. Store allowed exact (resolved cwd, argv tuple) commands in
memory; only configured checks and explicit 'session' decisions populate this set.
Model-selected timeouts may shorten but never exceed configured command_timeout.

- Create test-only repo and workspace fixtures. The repo fixture initializes
  Git in tmp_path with a UTF-8 file and test data. Any test commits use per-command
  local identity overrides; never alter global Git config. The workspace fixture
  initializes Workspace with no checks and an async callback returning 'deny'.
- Write read/edit tests via Tool, so inferred contracts and actual filesystem
  behavior are exercised together.

```python
import pytest
from slick import Tool, ToolError


def test_stale_edit_does_not_overwrite_external_changes(workspace):
    target = workspace.root / "sample.py"
    target.write_text("value = 1\n")
    original = workspace.read_file("sample.py")
    target.write_text("value = 2\n")
    tool = Tool(workspace.edit_file)
    with pytest.raises(ToolError):
        tool.invoke({"path": "sample.py", "old": "value = 1", "new": "value = 3",
                     "expected_sha256": original.sha256})
    assert target.read_text() == "value = 2\n"
```

- Run `python -m pytest tests/coding_harness/test_workspace.py -v`; observe red.
- Implement path checks once in workspace.py and call them from every file
  tool. Implement exact single replacement and exclusive creation, preserving
  CRLF/LF bytes and file mode. Bound file sizes and returned output. Record only
  successful app edits in the edit ledger with before/after digest and diff.
- Implement async Git/rg tools with argv lists. Disable external diff/textconv;
  bound catalog/output and return explicit truncation. Fingerprinting streams full
  file bytes and does not infer unchanged contents from mtime alone. The task
  fingerprint itself is not truncated; fail visibly if the scan cannot complete.
- Implement run_command decisions before spawn. Show exact argv/cwd, deny
  without calling run_process, remember exact session approvals, and enforce the
  maximum timeout. Configured checks may run repeatedly without new decisions.
- Test traversal, absolute paths, symlinks, .git, missing/large/binary files,
  ambiguous/empty replacements, missing parent/new-file collision, permissions,
  dirty/staged/untracked diffs, ignored paths, rg no-match, Unicode, and cancellation
  during a decision. Test all exposed methods prepare through prepare_tools.
- Test that allowing one command does not allow a different argument or cwd;
  built-in Git inspection does not ask; model-selected Git commands follow normal
  decisions. Document local execution rather than claiming these checks sandbox it.
- Run process/workspace tests on macOS or Linux and the minimum Python version.

## Task 3: Verification, baselines, and actual failure feedback

**Files:** examples/coding_harness/verification.py;
examples/prompts/coding_harness/verification.j2;
tests/coding_harness/test_verification.py.

**Consumes:** Workspace, Check, CommandResult, Verification.
**Produces:** `async verify(workspace: Workspace, checks: list[Check]) -> Verification`;
`render_verification(current: Verification, baseline: Verification | None) -> str`.
Store fingerprints before/after the whole batch in every CheckResult. Run all
checks even when an earlier one fails, within the encompassing task deadline.
An empty list has passed=False; it cannot produce verified status.

- Write tests with real Python commands returning 0/1, timing out, and changing
  a tracked file during execution. Verify the fixture's permitted check argv exactly
  matches the configured argv; setup does not bypass the policy under test.

```python
import asyncio
from examples.coding_harness.verification import verify


def test_empty_verification_cannot_claim_success(workspace):
    result = asyncio.run(verify(workspace, []))
    assert result.results == []
    assert result.passed is False
```

- Run `python -m pytest tests/coding_harness/test_verification.py -v`; observe red.
- Implement sequential checks through Workspace.run_command, preserve raw
  bounded results and fingerprints, and render feedback with the Jinja template.
  No LLM grades the result. A pre-existing failed baseline is informational, not a waiver.
- Test all-pass/stable, nonzero, timeout, denied/unlaunchable commands, changed
  worktree, modified verification files, missing environment, and absence of checks.
  Check output must name the configured command and expose actual exit state.
- Run verification and workspace tests. Final freshness is checked again by
  CodingAgent immediately before publishing a verified result in Task 4.

## Task 4: CodingAgent and bounded repair loop

**Files:** examples/coding_harness/agent.py;
examples/prompts/coding_harness/{system,task}.j2;
tests/coding_harness/test_agent.py.

**Consumes:** backend.aturn, native records, Workspace, prepare_tools, verify,
HarnessConfig, SessionState, RunResult, HarnessEvent.
**Produces:** `CodingAgent(backend, workspace, config, *, emit)`;
`agent.state: SessionState`; `async run(task: str) -> RunResult`;
`async step() -> ModelTurn`. Task 5 adds compact(). `emit` defaults to a no-op
application callback and is not part of Slick. No inheritance requirement on backend.

- Write a test backend with an async aturn method that returns scripted
  ModelTurn records and captures independent copies of history. It never runs tools.
  Tests provide provider/model metadata matching the records. Use real workspace
  tools; count side effects in files, not mock function assertions.
- Cover one read, one edit, unknown tools, malformed/invalid arguments, multiple
  calls, final text, zero tools, and completion with/without configured checks.

```python
# Within a test with a scripted backend whose first turn edits a real file:
result = asyncio.run(agent.run("Fix sample.py"))
assert (workspace.root / "sample.py").read_text() == "value = 2\n"
assert result.status == "verified"
assert result.checks[0].command.exit_code == 0
```

  Define CheckResult.command as the CommandResult field; both verify() and the
  TUI use that name. Use a real configured Python check reading the fixture file.
- Run `python -m pytest tests/coding_harness/test_agent.py -v`; observe red.
- Implement run as the owner of baseline, task deadline, counters and final
  verification. Prepare tools once per instance; keep step focused on one request
  and its sequential result group. Use small helpers for dispatch, recording errors,
  verification decisions and completion. Count requests before awaiting backend.
- Preserve completed results if a later tool fails. On cancellation, finish the
  local history group with cancelled/unknown/not-started results without rerunning
  any callable, record last_result with status cancelled, then propagate cancellation
  after cleanup. The TUI and headless entry point display the recorded result.
  A budget/deadline stop likewise closes returned call groups without executing
  actions beyond the limit.
  A failed request never fabricates a returned model turn.
- When a turn has no calls, verify. Append real verification feedback to history
  for both pass and failure, and derive status independently of model prose.
  On failure use the verification template and request another turn. Stop at the
  repair limit or identical repeated failures on unchanged contents. Do not
  automatically retry ToolError execution/result or install dependencies.
- Test cancellation during API, decision and command; cancellation between
  calls; result serialization failure after an edit; reused IDs; request/time/tool
  limits; repair limit; no progress; external edits invalidating verification;
  concurrent run rejection; recovery after a failed run. Use asyncio Events to
  synchronize, not arbitrary sleeps. No model call or function invocation occurs
  after a stop condition.
- Run all headless application tests and native-turn tests together.

## Task 5: Skills, bounded context, and compaction

**Files:** examples/coding_harness/context.py; examples/coding_harness/agent.py;
examples/prompts/coding_harness/{compact,checkpoint}.j2;
examples/prompts/coding_harness/skills/python.j2;
tests/coding_harness/test_context.py.

**Produces:** `render_instructions(workspace, config) -> str`;
`context_size(history: list, *, instructions: str, tools: list[Tool]) -> int`;
`summary_prompt(state: SessionState, config: HarnessConfig) -> str`;
`ContextSummary` as specified; `async CodingAgent.compact() -> None`.
Use a fixed {'python': 'coding_harness/skills/python.j2'} include mapping. Ensure
TEMPLATE_ROOT is examples/prompts at the application boundary, restoring it in tests.

- Write tests showing selected skill text and check configuration appear in
  rendered instructions, unknown skill names fail before a request, and context
  measurements include tool definitions plus opaque provider item sizes.
- Write compaction tests with completed tool/result groups and a scripted
  summary turn. Seed literal summary fields; assert pinned instructions/checks and
  actual workspace state survive replacement. No test asks an actual model.

```python
before = list(agent.state.history)
# Configure the scripted backend to return invalid JSON for the summary request.
with pytest.raises(ValueError):
    asyncio.run(agent.compact())
assert agent.state.history == before
```

  Surface parse failures with their cause; the automatic caller applies the soft/
  hard-limit policy rather than quietly substituting an unvalidated summary.
- Run `python -m pytest tests/coding_harness/test_context.py -v`; observe red.
- Render summary views that exclude opaque reasoning; retain task, checks,
  recent complete exchanges, and edit ledger within the hard limit. Validate the
  summary via parse, build checkpoint text, and swap history only after success.
- Integrate soft-limit checks at complete history boundaries; count summary
  requests in max_turns, never recursively compact a summary request, and reject
  oversized pinned context before making a request. Keep original histories for
  explicit session inspection, not automatic replay.
- Test pending calls cannot compact, no back-to-back automatic compaction,
  summary errors preserve state, malformed summary, hard-limit block, manual idle
  compaction, and instructions re-rendering without changes to Tool schemas.
- Run context, agent and existing prompt/example tests.

## Task 6: Explicit session files and inert resume

**Files:** examples/coding_harness/session.py;
tests/coding_harness/test_session.py.

**Consumes:** SessionState, config/result records, native records, workspace metadata.
**Produces:** `save_session(path: Path, agent: CodingAgent) -> None`;
`load_session(path: Path) -> SavedSession`; `async restore_session(saved, backend, *, decide, emit) -> CodingAgent`.
SavedSession is a validated version-1 record of design S01 fields. Encode native
records with explicit kind tags and restore through a fixed tag-to-constructor
mapping. JSON size is capped at 20 MiB on input; unknown versions/tags/keys fail.
Never import a class named in saved data or deserialize arbitrary Python objects.

- Write round-trip tests for messages, multi-tool groups, opaque provider
  payloads, config, edit ledger and results. Include invalid schema/version,
  duplicate IDs, pending calls, foreign provider/model, missing workspace and an
  existing output path. Assert save occurs only while idle and outside the repo.

```python
save_session(session_path, agent)
loaded = load_session(session_path)
restored = asyncio.run(restore_session(loaded, backend, decide=deny, emit=events.append))
assert restored.state.history == agent.state.history
assert (workspace.root / "sample.py").read_bytes() == original_bytes
assert restored.workspace.allowed_commands == restored.workspace.configured_commands
```

  Here configured_commands is the exact set derived from configured checks;
  session-only approvals are intentionally absent after restoration. Provide
  deny/events/original_bytes as ordinary test-local values.
- Run `python -m pytest tests/coding_harness/test_session.py -v`; observe red.
- Write complete JSON to a mode-0600 temporary file in the destination directory,
  flush it, then publish with os.link(temp, destination) and unlink the temporary
  name. Link creation is atomic and fails if the destination exists on the supported
  POSIX platforms. Clean up temporary files on failure; never overwrite a session.
  Save no SDK clients/credentials/permissions/processes. Native record dictionaries
  are JSON data; enforce the history invariant after loading.
- Restore by rebuilding Workspace and CodingAgent with the saved config and
  provider/model, then restoring data. Compare HEAD/fingerprint; invalidate previous
  verified state and add a workspace-changed observation when different. No tool
  result is interpreted as a command to rerun.
- Test changing the workspace after save, stale success state, denied saved
  approval attempts, no file changes/no backend requests on load, and a failed save
  leaving both workspace and prior files untouched. Run session and agent tests.

## Task 7: Runnable CLI and an offline repair demonstration

**Files:** examples/coding_harness/{__main__,demo}.py;
tests/coding_harness/test_cli.py.

**Produces:** `build_parser() -> argparse.ArgumentParser`, `main(argv=None) -> int`;
`create_demo(root: Path) -> HarnessConfig`; `DemoBackend(root: Path)` with model='scripted',
identity() returning {'provider': 'demo', 'model': 'scripted'}, and aturn matching the
core contract. CodingAgent uses existing backend.identity() to identify API backends
as well; no new provider attribute is required on OpenAI/Anthropic. DemoBackend verifies the
expected prior result at each scripted step and reads current digests to construct
valid edit arguments. Script exhaustion is a visible error, never fake final success.

- Add CLI subprocess tests for the headless demo, invalid model/workspace
  combinations, absent Git/rg, missing config, non-POSIX startup, and missing
  Textual guidance. Ensure --help and --headless work without importing Textual.
- Define a concrete demo fixture: `total(values)` initially returns
  `sum(values) + 1`; unittest checks a normal list and an empty list. Scripted first
  edit removes the error only for nonempty lists; the actual empty-list test fails;
  the second edit returns `sum(values)` for both. The baseline and checks use
  sys.executable with `-m unittest -q` and execute only in the temporary root.
  Include .gitignore entries for __pycache__/ and *.pyc. Commit the demo fixture
  once with a per-command local test identity so the diff view shows the repair;
  never commit a user's worktree.

```python
import subprocess
import sys


def test_headless_demo_repairs_real_code():
    completed = subprocess.run(
        [sys.executable, "-m", "examples.coding_harness", "--backend", "demo",
         "--headless", "--task", "Fix the total calculation"],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Verified" in completed.stdout
    assert "repair" in completed.stdout.lower()
```

  Pair the subprocess test with an in-process DemoBackend test that asserts the
  intermediate check failed, final file contents changed, and the final real test
  command passed. The printout alone is not evidence of healing.
- Run `python -m pytest tests/coding_harness/test_cli.py -v`; observe red.
- Implement parser/startup and resource lifetime. Real backends require model
  and workspace. --resume selects saved backend/model/root and forbids conflicting
  launch overrides. --headless requires --task, defaults to deny for unapproved
  commands, and prints application events followed by the actual RunResult.
- Set examples/prompts as TEMPLATE_ROOT once at app startup. Keep TUI imports
  inside the interactive branch. Default demo mode owns a TemporaryDirectory and
  cleans it only after process/worker shutdown; do not remove any user directory.
- Implement --backend {demo,openai,anthropic}, --model, --workspace, --config,
  --headless, --task, --resume. No other example backend choices are reused here.
- Test exit codes 0/2/1/130; provider SDK not loaded for demo; no file changes
  in Slick; cleanup on failed/cancelled demo; saved demo sessions fail resume once
  their temporary root is gone. Run CLI and all headless tests.

## Task 8: Textual interface, command modal, and clean interaction

**Files:** examples/coding_harness/tui.py; examples/coding_harness/requirements.txt;
examples/coding_harness/__main__.py; tests/coding_harness/test_tui.py.

**Consumes:** CodingAgent.run/compact and HarnessEvent; Workspace command decision
callback; session.save_session; native backend selected by startup.
**Produces:** `HarnessApp(backend, workspace_root: Path, config: HarnessConfig, *, saved: SavedSession | None = None)`;
`app.agent: CodingAgent` after on_mount initializes/restores it;
`async request_command(request: CommandRequest) -> CommandDecision`.
Headless setup and TUI setup call the same agent/workspace factory in __main__.py:
`async create_agent(backend, root, config, *, decide, emit, saved=None) -> CodingAgent`.
Pass config.limits.command_timeout into Workspace; metadata comes from backend.identity().

- Add textual==8.2.8 to the example requirements. Keep optional importorskip
  only in test_tui.py; the UI test job must explicitly install it and assert tests
  ran. No Textual dependency in Slick's mandatory package metadata.
- Write Pilot tests using app.run_test at 80x24 and 120x40. Inject the real demo
  backend and a temporary fixture repository. Submit through Input and Enter,
  await a completed event or worker completion with a timeout, and assert real
  file/test effects plus re-enabled input. Do not add test-only methods to the app.

```python
import asyncio
from textual.widgets import Input


def test_enter_runs_task(app):
    async def scenario():
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.query_one("#task", Input).value = "Fix the total calculation"
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=10)
            assert app.agent.state.last_result.status == "verified"
            assert app.query_one("#task", Input).disabled is False
    asyncio.run(scenario())
```

  The app fixture initializes a real disposable demo. Assert the repaired file
  contents and verification CommandResult in this test as well.
- Run `python -m pytest tests/coding_harness/test_tui.py -v`; observe red.
- Build the one-column layout using Header/Static, a scrolling transcript,
  Input(id='task'), Footer, and a reusable detail/decision ModalScreen. Use standard
  ListView entries for selectable tool results if RichLog alone cannot provide
  details. Keep CSS inline in tui.py; do not add a theme subsystem.
- Implement one async worker for the active run and a guarded busy flag.
  Worker cleanup re-enables input in finally. Emit events as Textual messages;
  UI updates do not execute tools. Expected failures render and return to idle.
- Bind Enter, Esc, Ctrl+D and Ctrl+Q and implement /help, /new, /compact,
  /save PATH. Busy task submission is rejected; scrolling/diff/cancel stay usable.
  Cancelling a decision resolves Deny and permits cleanup. /new resets conversation
  state and captures a new baseline without modifying source files.
- Sanitize external terminal control sequences and render as plain text.
  Show bounded command output and truncation, actual verification states, elapsed
  waiting time, and reported token counts. No token streaming or hidden reasoning.
- Test keyboard decision allow/deny/session scope, deny causes no file side
  effect, cancel kills a real long-running subprocess, quit waits for cleanup,
  follow-up after cancellation, diff access, output inspection, /new preserves
  files, /save and resume interaction, narrow-layout hint and injected terminal
  escapes rendered inertly. Verify application input cannot start overlapping work.
- Run the UI test module with Textual installed, then run the headless CLI
  with Textual blocked/missing to confirm separation.

## Task 9: Documentation, full integration, and final acceptance

**Files:** examples/coding_harness/README.md; examples/README.md; README.md;
CHANGELOG.md; tests/coding_harness/test_cli.py and test_tui.py as needed.

- Document every supported command and the explicit configuration example in
  the design. Describe local command permissions, lack of OS sandbox/automatic
  rollback, pre-existing changes, verification coverage, bounded outputs, session
  save paths, and the meaning of verified/unverified/blocked/cancelled/failed.
- Include the architecture and exact run commands. Explain the Python class,
  bound tools, prompts, skills and application state so a developer can replace any
  of them. Clearly label the demo's scripted model and actual filesystem/check work.
- Run complete tests in the normal environment and Python 3.10/Pydantic 2.0.
  Run a no-SDK/no-Textual core/headless job, an SDK transport job, and a Textual job;
  record actual counts so optional skips cannot hide untested application features.

```bash
python -m pytest tests/coding_harness
python -m pytest
python -m ruff check slick tests examples
python -m examples.coding_harness --backend demo --headless --task 'Fix the total calculation'
python -m build --wheel --no-isolation --outdir /tmp/slick-harness-dist
```

- Confirm the wheel includes native-turn core files, excludes example prompts
  and UI dependencies, and still imports without extras. The example runs from a
  checkout as documented, not from an assumed installed examples package.
- Manually run the TUI in a real terminal: paste input, scroll output, inspect
  a diff, deny a command, cancel during execution, start a follow-up, save and quit.
  Headless Pilot cannot establish every terminal/paste behavior. If no real terminal
  is available, report that check as outstanding rather than claiming it passed.
- Optional live trials require an explicitly requested provider/model and
  disposable repo. Measure task success, check outcomes, repair attempts, calls,
  elapsed time and reported tokens separately from deterministic acceptance.
- Review diff and requirement coverage. Record test evidence in this plan and
  leave changes ready for user review; do not commit, publish, or run paid calls
  merely because this plan mentions them.

## Coverage and dependency map

| Tasks | Spec requirements | Depends on |
| --- | --- | --- |
| 1 | Global constraints, W04, config/records/budgets | Native-turn record definitions |
| 2 | W01–W06 | 1 |
| 3 | A03 verification mechanics | 1–2 |
| 4 | A01–A05 loop/repair/cancellation | 1–3 and native adapters |
| 5 | C01–C02, prompt/skill composition | 4 |
| 6 | S01–S02 | 4–5 |
| 7 | S03, CLI/lifetime | 4–6 |
| 8 | U01–U05 | 4–7 |
| 9 | All integration/release evidence | 1–8 |

The user can review the native-turn prerequisite independently of the app. The
headless repair demo is the functional gate before UI work; the final UI must
exercise that same agent and effects rather than a separate demonstration loop.


## Execution record — 2026-09-07

- [x] Task 1: strict application records, config and bounded subprocess boundary.
- [x] Task 2: Git workspace discovery, plain bound tools, precise edits and command decisions.
- [x] Task 3: baseline/final verification from real checks and file fingerprints.
- [x] Task 4: bounded native-turn dispatch, repair, interruption and completion state.
- [x] Task 5: example-local Jinja skills, context sizing and validated compaction.
- [x] Task 6: explicit JSON sessions, atomic no-overwrite save and inert resume.
- [x] Task 7: CLI and scripted model over a real disposable repair fixture.
- [x] Task 8: Textual interface, decisions, cancellation, details, diff and session commands.
- [x] Task 9: usage docs, integration checks, wheel boundary and real-terminal smoke.

All app modules live in examples/coding_harness and templates in
examples/prompts/coding_harness. requirements.txt pins example-only Textual 8.2.8.
The additional test_cli_edges.py isolates startup/exit/lifetime acceptance cases.
The existing checkout was used after Git worktree creation was denied by filesystem
permissions. Prior work was preserved; all changes remain uncommitted.

Validation evidence:

| Check | Result |
| --- | --- |
| Complete suite, Python 3.14 with optional SDKs and Textual | 593 passed in 46.93s |
| Complete suite, Python 3.10 / Pydantic 2.0, no SDKs or Textual | 565 passed, 2 optional modules skipped in 34.05s |
| Application cases included in the complete suite | 133 passed, including 22 Textual cases |
| Current SDK transport and declared SDK floor round trips | Passed over mock HTTP; floor job 100 native/transport cases |
| Ruff check slick tests examples | Passed |
| Ruff format check on changed native and application/test modules | 30 files passed |
| Native/backend mypy | Four modules passed |
| git diff --check | Passed |
| Headless demo in the minimal environment | Verified: 6 model turns, 4 tools, 1 repair; actual check exits 1 then 0 |
| Wheel build/inspection/import | Native files included, examples/Textual excluded, optional imports remain lazy |
| README Python examples | Nine snippets parsed, including intentional top-level await fragments |

The complete application tests cover real edits and effects, full digest checking,
overlapping matches, symlinks and path escapes, command decisions, bounded output,
launch failure, timeouts, descendant cleanup, API cancellation, interrupted multi-call
groups, duplicate call IDs, invalid returns after effects, budgets, no progress,
repair limits, stale verification, verification-script edits, context bounds,
compaction concurrency and transport failures, sessions and CLI failure cleanup.

Real PTY checks exercised the 80x24 UI: bracketed paste, successful demo repair,
diff and page scrolling, command denial, allow-once, cancellation of a real running
child, follow-up after confirming the child was reaped, session save and clean
Ctrl+Q/Ctrl+C exit. Saved JSON was validated and denied-command side effects were
absent. Session-wide permission scope, resume and both layout sizes are additionally
covered by Pilot. The PTY driver and evidence were temporary local artifacts.

Review findings were reproduced and fixed before final validation: compaction
concurrency, recoverable summary transport errors, incomplete exchange trimming,
malformed session payloads and symlink destinations, empty idle session saves,
inherited Git signing, unsupported-platform startup, ambiguous overlapping edits,
unbounded launch errors, leftover descendant processes, and hidden verification-file
change notices. Configured check script paths now participate in that notice.

Known scope matches the design: local commands are not sandboxed, file fingerprints
exclude ignored/external state, and passing configured checks is not a proof of
correctness. Sessions are explicit and not durable execution. Raw filenames that
cannot be represented unambiguously as UTF-8 (including literal U+FFFD) fail closed.
No live model trials were run; those are optional and require explicit provider/model
selection. There are no outstanding implementation or acceptance steps.

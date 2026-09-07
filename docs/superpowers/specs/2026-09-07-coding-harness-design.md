# Coding harness example with a terminal interface

Status: implemented following the user’s approval of the complete plan.
The linked plans record implementation and validation evidence. This design describes
the first complete example, not feature parity with Claude Code.

Implementation sequence:
1. [Native tool turns](../plans/2026-09-07-native-tool-turns.md).
2. [Coding harness application](../plans/2026-09-07-coding-harness-app.md).

## 1. Outcome and architectural decisions

A developer runs a Python example in a chosen Git worktree, enters a coding task,
and watches one agent inspect files, make edits, run commands, diagnose failures,
and verify the resulting code. The developer can inspect results and diffs,
cancel work, continue the conversation, and explicitly save/resume a session.
An offline demo performs a real edit/test/repair cycle in a temporary repository.

Slick owns Jinja rendering, local callable-tool contracts, and model communication.
The example owns the loop, workspace operations, verification, context, state,
command decisions, and terminal interface. There is no core Agent, Conversation,
workflow engine, skill registry, event bus, durable executor, or repair policy.

Three approaches considered:

| Approach | Assessment |
| --- | --- |
| Textual app over our own native tool loop | Selected: explicit Python behavior, independently testable UI, exercises Slick's primitives |
| Prompt for JSON actions using current text calls | Useful prototype, but duplicates the provider tool protocol and postpones the missing core boundary |
| Wrap the Claude Code/Agent SDK harness | Fast route to existing capabilities, but delegates the very harness this example is intended to demonstrate |

The app starts with one agent and sequential tool execution. Read/search tools,
precise edits, a bounded command runner, and externally configured checks are
sufficient. No vector database, language server, subagents, browser tools, MCP,
OpenRouter transport, embedded terminal emulator, model-token streaming, or
automatic Git commits/rollback are included. These are optional future work.
OpenRouter requires a separate Chat Completions adapter; the existing OpenAI
adapter uses Responses and must not be represented as OpenRouter-compatible.

## 2. Global constraints

- Python >=3.10; core remains compatible with Pydantic >=2.0.
- No new mandatory Slick dependency; existing call/acall and Tool contracts remain unchanged.
- All application code and Jinja templates live under examples/, never slick/.
- Textual is example-only, pinned to 8.2.8 in examples/coding_harness/requirements.txt.
- Core tests run without provider SDKs or Textual; extra-dependent test jobs install their dependencies explicitly.
- Automated tests use scripted models and local temporary workspaces, never paid endpoints or the developer's working tree.
- No core history, tool execution, repair, persistence, caching, or thread scheduling is introduced.
- The first command-running application supports macOS and Linux; other platforms fail startup clearly before work begins.
- Implementation was subsequently authorized; commits and publishing remain outside scope.

Textual provides async workers and headless keyboard/mouse tests, which makes it
suitable for a responsive, small interface. Its current package metadata supports
Python 3.10. [Testing](https://textual.textualize.io/guide/testing/),
[workers](https://textual.textualize.io/guide/workers/),
[package metadata](https://pypi.org/project/textual/).

## 3. The core prerequisite: one native model turn

Add `slick/turns.py` containing data records and small pure protocol helpers.
Keep provider codecs in `slick/_openai_turns.py` and `slick/_anthropic_turns.py`
so neither the shared records nor existing backends become a large dispatcher.
Extend the existing `OpenAI` and `Anthropic` classes in `slick/backends.py` with:

```python
async def aturn(
    self,
    history: list[UserMessage | ModelTurn | ToolResult],
    *,
    tools: list,
    instructions: str = "",
) -> ModelTurn:
    """Make one model request; never execute tools or mutate history."""
```

The records are standard dataclasses, imported explicitly from `slick.turns`:

| Record | Fields |
| --- | --- |
| UserMessage | `text: str` |
| ToolCall | `id: str`, `name: str`, `arguments: dict | None`, `argument_error: str | None = None` |
| ToolResult | `call_id: str`, `content: str`, `is_error: bool = False` |
| ModelTurn | `provider: str`, `model: str`, `text: str`, `tool_calls: list[ToolCall]`, `items: list[dict]`, `stop_reason: Literal['tool_calls', 'end_turn']`, `input_tokens: int | None = None`, `output_tokens: int | None = None` |

`items` contains a JSON copy of provider output needed to replay the turn.
It is opaque to application prompting and must not contain SDK client objects.
Provider parsing derives text and calls from these items; replay preserves their
order and provider-specific continuation data. The application appends the turn,
then a ToolResult for every requested call, then asks for the next turn.

**N01 — Preparation:** accept ordinary callables or Tool objects. Call prepare_tools
before opening a client or sending a request. Send schema copies, never mutate the
local Tool. Native turns initially use best-effort schemas; OpenAI explicitly sets
`strict: false` so optional Python arguments are not silently made required.
Local strict validation still happens when Tool.ainvoke is called. Provider schema
rejection is an actionable BackendError, never a reason to silently broaden types.

**N02 — History:** reject a foreign provider/model turn, duplicate call IDs,
results without calls, duplicate results, and a request with unresolved calls.
A new user message cannot separate calls from their results. Validate the full
history before sending. Multiple result records are grouped appropriately for
the provider. Empty history and textless/tool-less responses are errors.

**N03 — Requests:** OpenAI serializes user messages and complete response output
items into Responses input, followed by function_call_output items using call_id.
Keep store=False; preserve reasoning/encrypted continuation items. Anthropic uses
Messages assistant content followed immediately by a user message containing all
tool_result blocks. Instructions use the respective instructions/system field.
Ask for one tool at a time (`parallel_tool_calls=False` for OpenAI;
`tool_choice={type:'auto', disable_parallel_tool_use:True}` for Anthropic when tools
are present). Still parse multiple calls defensively and execute none in core.
With an empty tools list, omit tool-choice controls and reject unexpected calls.

**N04 — Responses:** preserve all supported text/reasoning/tool content needed for
continuation. Reasoning payloads are not displayed as user-facing conversation.
Refusal, truncation, incomplete tool blocks, unknown output kinds, and inconsistent
stop conditions fail without dispatching partial actions. Nonempty unknown tool
names remain calls for the harness to answer with an error. Malformed JSON,
duplicate JSON keys, non-object arguments, or nonfinite numbers become a ToolCall
with arguments=None and a concise argument_error; preserve the original provider
item for replay. Reject empty/duplicate IDs or missing names as protocol errors.
Do not use Slick's permissive prose/fenced-output parser for tool arguments.

**N05 — Compatibility:** preserve current lazy imports, client ownership,
timeouts, zero default SDK retries, cancellation, and plain text call/acall
behavior. No sync native-turn method or automatic tool loop is added in this
milestone. Existing CLI harness adapters remain unchanged and are not selectable
as native-turn backends in this application.

These are intentional app-independent protocol decisions. The OpenAI function
calling guide describes call/result correlation and strict schema requirements;
reasoning continuation must survive replay. Claude requires tool results adjacent
to their corresponding calls. [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling),
[OpenAI reasoning](https://developers.openai.com/api/docs/guides/reasoning),
[Claude tool results](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls).

## 4. Application files and responsibilities

```text
examples/coding_harness/
  __init__.py          # No UI import or initialization side effects
  __main__.py          # CLI selection, config, dependencies, lifetime/cleanup
  agent.py             # CodingAgent.run/step, dispatch, budgets, repair
  state.py             # Run/session records, events, limits, check configuration
  workspace.py         # Typed bound tools, paths, file versions, Git inspection
  process.py           # Bounded async subprocesses and cancellation cleanup
  verification.py      # Fixed checks, baseline, final checks, freshness
  context.py           # Render instructions/skills and compact complete history
  session.py           # Explicit versioned JSON save/load at stable boundaries
  tui.py               # Textual widgets, bindings, worker and command decisions
  demo.py              # Scripted native turns and temporary broken repository
  requirements.txt     # textual==8.2.8
  README.md            # Installation, commands, scope, examples
examples/prompts/coding_harness/
  system.j2
  task.j2
  verification.j2
  compact.j2
  checkpoint.j2
  skills/python.j2
```

Keep pure helpers small; no large catch-all manager classes. Dataclasses for
application records are not tool inputs/outputs. Exposed tool result types use
Pydantic BaseModel with fields supported by the existing callable-tool spec.

## 5. Workspace and tool contract

Require an explicitly selected Git worktree root for real backends. Startup
checks Git and rg availability, captures branch/HEAD and initial dirty status,
and displays the resolved path. Existing changes remain in place. Never stash,
reset, change branches, install project dependencies, or commit automatically.
The demo creates its own temporary Git repository and requires no workspace flag.

Expose only these bound methods:

```python
async def list_files(self, pattern: str = "*") -> FileList
async def search(self, query: str, glob: str = "*") -> SearchResult
def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> FileSlice
def edit_file(self, path: str, old: str, new: str, expected_sha256: str) -> EditResult
def create_file(self, path: str, content: str) -> EditResult
async def run_command(self, argv: list[str], timeout: int = 120) -> CommandResult
async def git_diff(self) -> DiffResult
```

Result records:
- FileList: `paths: list[str]`, `truncated: bool`.
- SearchHit: `path: str`, `line: int`, `text: str`; SearchResult: `hits: list[SearchHit]`, `truncated: bool`.
- FileSlice: `path: str`, `text: str`, `start_line: int`, `end_line: int`, `sha256: str`, `truncated: bool`.
- EditResult: `path: str`, `sha256: str`, `diff: str`.
- CommandResult: `argv: list[str]`, `exit_code: int | None`, `stdout: str`, `stderr: str`, `timed_out: bool`, `truncated: bool`.
- DiffResult: `unstaged: str`, `staged: str`, `untracked: list[str]`, `truncated: bool`.

**W01 — Discovery:** use Git's tracked/nonignored-untracked catalog and rg for
search. Do not crawl .git, ignored dependency trees, or symlink targets. Cap file
lists and search hits at 200; report truncation. Query/glob/path values are argv
arguments, never interpolated shell source. Treat rg exit 1 as no matches.

**W02 — Files:** paths must resolve within root; reject absolute paths, traversal,
.git access, symlinks, and nonregular files. Text tools handle UTF-8 files up to
1 MiB and output at most 20,000 characters. Preserve newline bytes and permissions
when editing. Read returns a digest of the full file, even for a slice.

**W03 — Edits:** require a matching digest and exactly one occurrence of nonempty
old text. Reject ambiguity or stale contents without writing. Stage replacement
bytes in the same directory and replace the file only after validation; clean up
on failure. Creation requires an absent path and an existing permitted parent;
use exclusive creation. Never silently create parent directories. These checks
handle ordinary concurrent edits; they are not an OS sandbox or an adversarial
filesystem race guarantee. No general patch parser or overwrite/delete tool yet.

**W04 — Processes:** run argv with create_subprocess_exec, stdin=DEVNULL, cwd=root,
a new POSIX session, and concurrently drained stdout/stderr. Retain at most 8,000
characters per stream in bounded buffers; continue draining after truncation.
Timeout or cancellation kills and reaps the owned process group and closes pipes.
Bound cleanup to five seconds so escaped descendants cannot hang exit indefinitely;
report incomplete cleanup. No background jobs, interactive PTY, or shell-string API.
A nonzero command exit is a CommandResult observation, not an exception/retry.

**W05 — Decisions:** read/search/edit/create and internal Git inspection run within
the selected workspace without repeated prompts. Every model-selected command
requires a visible decision unless its exact argv/cwd is already allowed for this
session or is an explicitly configured verification check. Show argv and cwd with
Allow once / Allow this command for session / Deny. No broad command-prefix
allowlisting. Decisions are an async application callback, never a Slick primitive.
Headless mode denies unapproved commands. Approval does not sandbox execution:
commands run with the user's permissions, including network access. For stronger
isolation, launch the whole app in a separately managed container/worktree.

**W06 — Diffs:** show current Git staged/unstaged changes and untracked paths,
labeling pre-existing dirty state. Use --no-ext-diff/--no-textconv. Do not claim
all displayed changes belong to the agent. Keep a separate app-edit ledger with
before/after digests for the report. No undo command that could erase user edits.

## 6. State, execution, verification, and repair

`CodingAgent(backend, workspace, config, *, emit)` owns session history and exposes
`async run(task: str) -> RunResult`, `async step() -> ModelTurn`, and
`async compact() -> None`. `emit(event: HarnessEvent) -> None` is a synchronous,
application-local callback. The UI posts events; headless mode collects/prints
them. Never make UI rendering the mechanism that advances the agent.
For session metadata the example also uses `backend.identity()['provider']` and
`backend.identity()['model']`; both existing API backends already supply these.
The demo implements identity() too. This is an example requirement, not a new
requirement on Slick's ordinary text backends.

Config: `HarnessConfig(checks: list[Check], skills: list[str], limits: Limits)`.
Check: `name: str`, `argv: list[str]`, `timeout: int = 120`.
Only an explicit `--config FILE` loads checks/skills; no project file executes
commands merely because it exists. Reject unknown keys, duplicate check names,
empty argv, and invalid limits. JSON avoids a new Python 3.10 TOML dependency.

Limits defaults: `max_turns=30`, `max_tool_calls=60`, `max_repairs=3`,
`task_timeout=900`, `command_timeout=120`, `context_soft_chars=80000`,
`context_hard_chars=120000`. Validate positive values and soft < hard.
Model request attempts, including compaction, count before making the request.
A task-wide deadline also bounds waiting for command decisions.

RunResult: `status` in verified/unverified/blocked/cancelled/failed;
`answer: str`, `checks: list[CheckResult]`, `turns: int`, `tool_calls: int`,
`repairs: int`, `changed_paths: list[str]`. CheckResult fields are `name: str`,
`command: CommandResult`, `before_fingerprint: str`, `after_fingerprint: str`.
Status is application
computed, never assigned from model prose. Budget exhaustion is blocked.

HarnessEvent: `kind: str`, `data: dict`. Defined kinds: user, assistant, status,
tool_started, tool_finished, verification, compacted, completed. Decision UI uses
the awaited callback rather than an event bus. Store bounded observations;
provider tokens are displayed only when reported, never as invented costs.

**A01 — Loop:** single active run per instance. Append rendered task as UserMessage,
obtain one ModelTurn, append it, execute requested tools sequentially in returned
order via prepared Tool.ainvoke, append exactly one ToolResult per call. Complete
a whole result group before another user message/request. No call becomes runnable
merely by appearing in model text. Unknown names and malformed/invalid arguments
produce concise error results without executing any function.

**A02 — Failures/cancellation:** ToolError arguments means nothing ran; execution
or result means effects may have occurred. Include phase/effects uncertainty in
feedback; never automatically rerun that action. Cancellation after a returned
model turn records completed results and cancellation/not-started results for the
rest, preserving valid history. Kill/reap an active command before accepting a new
run. API failure adds no fabricated assistant turn. User can continue after a
cancelled/failed run; no saved or pending tool calls execute on resume.
run() records a cancelled last_result before propagating CancelledError; the TUI
and headless entry point display that result. Budget/deadline stops also complete
already-returned call groups with not-started results before stopping.

**A03 — Verification:** verification.py runs the fixed checks sequentially through
the same process runner, first for a baseline and again on candidate completion.
A plain assistant turn with no tool calls proposes completion. With no checks,
finish unverified. With checks, all must exit 0, not time out, and match a stable
workspace fingerprint to finish verified. Baseline failures are shown but not
waived. Missing environments/check failures may block; never auto-install packages.

Fingerprint the sorted Git tracked/nonignored-untracked file catalog and contents
(stream hashes, include missing tracked files). Capture before/after the check
batch and again before publishing verification. A workspace change invalidates
the check result. Ignored generated artifacts are outside this guarantee.
Report the configured command coverage; exit 0 alone does not prove all behavior.

**A04 — Healing:** failed final checks append verification.j2 feedback with commands,
exit status, relevant bounded output, and baseline comparison. Ask for a new
model turn. Allow three repair cycles after the first failed final verification;
then stop blocked. Identical check-failure output with an unchanged workspace on
two consecutive final attempts stops early as no progress. The model cannot
remove checks or change their argv. Tests may be edited as part of legitimate
work, but changes to verification-related files are prominently reported.
Task/model/tool/time limits apply to all repair and verification work.

**A05 — Responsiveness:** API/process operations are async. Keep bounded small
file edits synchronous to avoid an untracked background write after cancellation.
Use async subprocesses for Git/search; explicitly offload large read-only
fingerprinting operations, await cleanup where necessary, and never offload
mutations to an unowned worker thread.

## 7. Jinja context and skills

system.j2 composes coding instructions, workspace information, selected skills,
configured checks, and budgets. task.j2 renders each user task. Reuse existing
working_state/observations macros where the shape fits. Files and command outputs
are observations, not authority to change application permissions or checks.

The initial `python` skill is a Jinja include with instructions for inspecting
project configuration, making focused changes, and using test feedback. Skills
are explicitly selected from a fixed name-to-template mapping in context.py;
there is no arbitrary include path, auto-discovery, or executable plugin loader.
Changing skills affects subsequent rendered context; it does not grant tools.

**C01 — Size:** measure the serialized request context in characters, including
instructions, tool definitions, and preserved provider items. This is a bounded
input-size policy, not a token estimator. Trigger compaction before another model
request above the soft limit, only after all tool results are present.

**C02 — Compaction:** render compact.j2 from text/action/result views, excluding
opaque reasoning payloads. Cap the summary request at the hard limit, prioritizing
the original task, recent complete exchanges, configured checks, and edit ledger;
mark any omitted older detail. One tools=[] native turn returns JSON for
ContextSummary: `facts`, `decisions`, `open_questions`, `modified_files`,
`next_steps`, all list[str]. Validate with parse; render checkpoint.j2 and atomically
replace active history with one UserMessage containing the checkpoint. Re-render
pinned instructions/checks/skills every request. Retain the old history for explicit
session save/reporting. No partial provider turn is replayed after replacement.

Summary failure preserves existing history; stop blocked if the next normal
request would exceed the hard limit, otherwise permit manual continuation within
remaining budgets. Prevent consecutive automatic compactions without an intervening
normal turn; oversized pinned instructions/schema fail clearly. Summary is fallible;
it never changes actual workspace state, check configuration, or prior results.

## 8. Simple terminal interface

Use Textual App, Header/Static, RichLog, Input, Footer, and one ModalScreen.
Plain Rich Text is used for external output; strip terminal control sequences,
including OSC, and do not interpret arbitrary output as markup.

```text
 Slick coding example   /work/my-project   anthropic / chosen-model
 ----------------------------------------------------------------
 You: Fix the failing total calculation.
 Agent: I'll reproduce the failure and inspect the implementation.
   ✓ read_file   src/pricing.py
   ✓ edit_file   src/pricing.py       [diff available]
   ✗ check unit  1 failed             [output available]
 Agent: The empty-list case still fails; I'll correct that branch.
   ✓ check unit  12 passed
 Verified · 7 model turns · 5 tools · 1 repair
 ----------------------------------------------------------------
 > enter a task or /help
 Enter send   Esc cancel   Ctrl+D diff   Ctrl+Q quit
```

**U01 — Input:** single-line Input with horizontal scrolling. Enter submits a task
while idle. Disable submission during an active run; keep scrolling, inspection,
cancellation, and quit responsive. To steer, cancel first then submit a follow-up.
No concurrent run queue or multiline editor in the first example.

**U02 — Feedback:** append completed assistant text and tool start/result entries;
show active operation immediately and elapsed time while waiting for a model.
Expandable result details and a scrollable diff modal show bounded output.
A simple list of result entries with selection and Enter-to-open details is
sufficient; no custom tree widget. Do not expose reasoning/opaque payloads.

**U03 — Controls:** Esc cancels the active run (idle: dismiss modal); Ctrl+D opens
a read-only diff; Ctrl+Q cancels/cleans up then exits. /help lists commands;
/new clears conversation/check history after the current run is stopped and
captures a fresh workspace baseline without changing files; /compact compacts
while idle; /save PATH saves while idle. --resume PATH restores at startup.
The decision modal offers Allow once, Allow this command for session, and Deny;
closing the modal denies. Session allowances are not persisted.

**U04 — Workers:** one async worker per run. Starting another run while busy
raises/shows a controlled error, rather than exclusive=True silently cancelling
ongoing edits. On exit await cleanup. Render expected backend/tool errors visibly
and return to idle; unexpected app exceptions show a failed status with a useful
local error rather than silently swallowing it. Don't advance the loop in handlers.

**U05 — Layout:** usable at 80x24 and 120x40; narrower terminals show a resize hint.
Use one column, no permanent sidebar. Keyboard navigation reaches input, results,
diff and decisions. Status is conveyed with text as well as color.

## 9. CLI, configuration, and explicit sessions

Commands run from a repository checkout because examples are not wheel packages:

```bash
python -m pip install -e '.[api]'
python -m pip install -r examples/coding_harness/requirements.txt
python -m examples.coding_harness --backend demo
python -m examples.coding_harness --backend anthropic --model YOUR_MODEL_ID --workspace /path/to/repo --config /path/to/checks.json
python -m examples.coding_harness --backend openai --model YOUR_MODEL_ID --workspace /path/to/repo
python -m examples.coding_harness --backend demo --headless --task 'Fix the total calculation'
python -m examples.coding_harness --resume /path/outside/repo/session.json
```

No backend flag means demo. Real providers require explicit model/workspace;
credentials are checked only when execution needs the SDK. --headless avoids
importing Textual. Demo forbids a user workspace and never modifies Slick itself.
No model/provider switching inside a live session. Exit codes in headless mode:
0 verified, 2 unverified/blocked, 1 failed, 130 cancelled.

Example explicit config:

```json
{
  "checks": [{"name": "unit", "argv": ["python", "-m", "pytest", "-q"], "timeout": 120}],
  "skills": ["python"],
  "limits": {"max_turns": 30, "max_tool_calls": 60, "max_repairs": 3}
}
```

**S01 — Saving:** /save requires an idle, valid history with no pending tool calls.
Write versioned JSON atomically to an explicit path outside the worktree; refuse
to overwrite an existing file. Save provider/model, resolved root, HEAD, workspace
fingerprint, config, active/archived history, edit ledger, and last result. Exclude
credentials, SDK clients, process handles and command allowances. No pickle or
callable serialization. There is no autosave or crash-recovery execution engine.

**S02 — Resume:** validate schema/version and history, require matching provider,
model, root and existing worktree, restore only inert data. Recompute HEAD and
fingerprint. If changed, retain history but invalidate old verification and append
an explicit workspace-changed observation before the next model call. Rebuild tools
from the current application, not saved definitions. Never replay old actions.

**S03 — Demo:** create a temporary repo with pricing.py and a unittest test module.
A .gitignore excludes Python bytecode, so verification does not invalidate itself.
Demo setup may commit its fixture once using a per-command test identity, only
inside its own temporary repository. A scripted backend first proposes a plausible incomplete edit; a real test fails;
the next scripted turn repairs the remaining case; tests pass. Scripted calls use
real file digests and validate expected prior observations, not a fake success flag.
Clean up the temporary workspace on exit; saved demo sessions are for inspection
and cannot resume after their workspace has been deleted.

## 10. Acceptance and completion gates

| ID | Required evidence |
| --- | --- |
| N01–N05 | Offline protocol tests for both providers; unchanged text/tool tests; real SDKs against mock HTTP transports |
| W01–W06 | Temporary repository/file tests, real subprocess timeout/cancel/output tests, stale edit and path tests |
| A01–A05 | Scripted turns execute actual tools; no duplicate effects; actual failing check repairs; budgets and stale verification stop honestly |
| C01–C02 | Jinja rendering and bounded compaction tests; pending calls never split; failed summary preserves state |
| U01–U05 | Textual Pilot sends input, inspects output/diff, denies/allows commands, cancels a running child, and quits cleanly at two sizes |
| S01–S03 | Explicit save/load tests, corrupted/mismatched state rejection, no action replay, real offline demo from CLI |

Core, headless app, and optional UI/provider tests are separate jobs so optional
skips cannot be mistaken for complete coverage. Run Python 3.10/Pydantic 2.0 and
the normal development environment for core/headless tests; install pinned Textual
for the UI job and both optional SDKs for transport tests. Use asyncio.run in tests
so a pytest-asyncio dependency is unnecessary. Manually smoke-test a real terminal
for paste, scrolling, color and Ctrl+C/quit behavior after headless UI tests.

A live model trial is explicitly opt-in, uses a disposable repository and configured
budgets, and reports results separately from deterministic acceptance. No model
quality claim follows from scripted tests alone. The implementation is ready for
review only when the demo visibly repairs a real failure, the TUI can cancel cleanly,
both native adapters pass transport tests, and all existing Slick tests still pass.

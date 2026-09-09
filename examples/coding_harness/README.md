# Coding harness example

An ordinary `CodingAgent` class combines Jinja instructions, explicit provider calls,
bound Python tools, workspace state, and a bounded verification/repair loop.
The same agent runs in a simple Textual TUI or headlessly. All application code
and templates live under `examples/`; the installed Slick package supplies only
the renderer, callable tools, provider transport, and Session interaction bookkeeping.

## Run with a model

From a repository checkout, on macOS or Linux with Python 3.10+, Git and
[ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) on PATH:

```bash
python -m pip install -e '.[api]'
python -m pip install -r examples/coding_harness/requirements.txt
```

Normal runs use **OpenAI by default**. Set `OPENAI_API_KEY` in your environment
and choose an explicit model supporting native function tools. To use Anthropic,
set `ANTHROPIC_API_KEY` and pass `--provider anthropic`.
Real runs make multiple model requests.

```bash
python -m examples.coding_harness --model YOUR_MODEL_ID --workspace /absolute/repo --config /absolute/checks.json
python -m examples.coding_harness --provider anthropic --model YOUR_MODEL_ID --workspace /absolute/repo --config /absolute/checks.json
```

For **LiteLLM with OpenRouter**, install the optional extra and set
`OPENROUTER_API_KEY` in your environment:

```bash
python -m pip install -e '.[litellm]'
python -m examples.coding_harness --provider litellm --model 'openrouter/openai/gpt-oss-120b:nitro' --workspace /absolute/repo --config /absolute/checks.json
```

LiteLLM needs Python 3.10–3.14. Its `openrouter/` prefix selects the OpenRouter
route; the remaining `openai/gpt-oss-120b:nitro` is the OpenRouter model ID.
See [LiteLLM's OpenRouter setup](https://docs.litellm.ai/docs/providers/openrouter).
Slick translates tool request/result dictionaries to LiteLLM's Chat Completions
format. The harness executes tools and retains its verification/repair loop.
Saved sessions retain the provider and model; credentials stay in the environment.

`--workspace` can point to a new project directory. On a fresh launch, the harness
creates missing directories and initializes Git if the directory is not already
in a repository. Existing repositories must be selected at their exact worktree
root. Existing staged, unstaged and untracked changes are preserved and included
in the diff view. No automatic commit, branch switch, dependency installation or rollback occurs.
Use a disposable repository for a first live trial.

The optional JSON config is loaded only from `--config`; the app does not discover
or execute configuration from repository files. For example:

```json
{
  "checks": [
    {"name": "unit", "argv": ["python", "-m", "pytest", "-q"], "timeout": 120}
  ],
  "skills": ["python"],
  "limits": {
    "max_turns": 30,
    "max_tool_calls": 60,
    "max_repairs": 3,
    "task_timeout": 900,
    "command_timeout": 120,
    "context_soft_chars": 80000,
    "context_hard_chars": 120000
  }
}
```

Use the intended environment's Python executable in check argv. Missing check
dependencies produce real failures; the app does not install them. Omitted config
means no checks and no skills. Omitted limits use the defaults above. Limits must
be positive integers; context soft limit must be below the hard limit.

## Run the offline demo with --dry-run

```bash
python -m examples.coding_harness --dry-run
```

Enter `Fix the total calculation`. The demo creates a temporary Git repository
with a broken `total()` function and two unittest checks. Its **scripted model**
first makes an incomplete edit, observes a real failing test, then repairs the
remaining error. Reads, edits, subprocesses, diffs and verification are real;
the demo makes no API calls and needs no API key. It demonstrates this fixed task,
not arbitrary coding. The temporary repository is removed when the application exits.

`--dry-run` replaces `--provider demo` and cannot be combined with `--provider`,
`--model`, `--workspace`, `--config`, or `--resume`. It runs the disposable fixture;
it does not preview changes to your own workspace.

Headless mode needs no Textual installation, and the dry run only needs the base
package (`python -m pip install -e .`):

```bash
python -m examples.coding_harness --dry-run --headless --task 'Fix the total calculation'
```

## Interaction

| Control | Action |
| --- | --- |
| Enter | Submit a task or follow-up |
| Esc | Cancel the current operation; in a modal, close it or deny the command |
| Ctrl+D | Inspect current staged/unstaged diffs and untracked paths |
| Tab, arrows, Enter | Select a tool/check result and inspect its output |
| Ctrl+Q / Ctrl+C | Cancel active work, wait for cleanup, then quit |
| `/help` | Show controls |
| `/new` | Clear conversation and capture a fresh workspace baseline; retain files |
| `/compact` | Summarize idle history into a validated checkpoint |
| `/save PATH` | Save an idle session to a new file outside the workspace |

The transcript, status and elapsed time remain visible while
a task runs. There is one active task. Tool/check output and diffs have scrollable
detail views. Token streaming is not implemented. A terminal at least 80×24 is
recommended; smaller terminals show a resize hint.

Commands are argv lists with the workspace as cwd and stdin disconnected. Exact
configured check commands are permitted automatically. Other commands present
the complete argv, cwd and timeout with **allow once**, **allow for session**, and
**deny** choices. Session permission matches that exact argv and cwd. Headless mode
denies commands beyond the configured checks.

This is a local process boundary, not an OS sandbox: approved commands run with
your user permissions and can access the network and files outside the workspace.
File tools enforce workspace-relative paths, but cannot restrict what an approved
program does. Cancellation/timeout kills and reaps the owned process group;
programs that deliberately escape it cannot be reliably controlled here. Background
jobs are unsupported. Completed edits remain after cancellation.

## Verification and limits

The agent runs checks before work, then after every candidate completion. Failures
feed actual bounded output back into the next turn. A failing baseline is recorded
but never waives a final failure. Repair stops at the configured budget or a repeated
identical failure on unchanged files.

| Status | Meaning | Headless exit |
| --- | --- | --- |
| `verified` | All configured checks passed on a stable, fresh workspace fingerprint | 0 |
| `unverified` | Candidate completion with no configured checks | 2 |
| `blocked` | Budget/deadline, context limit, or unresolved checks stopped the run | 2 |
| `failed` | Provider/protocol/application error | 1 |
| `cancelled` | Execution was interrupted; inspect retained changes | 130 |

Verification establishes only what those checks cover. Changes to recognizable
test/config paths and explicit configured check script paths are highlighted in the
final report. This detection is not a complete dependency analysis. Checks modified
to pass can still pass; inspect the diff and the checks' meaning.

File tools read UTF-8 regular files up to 1 MiB, reject symlinks, traversal and `.git`
paths, and require a full-file SHA256 plus a unique exact match for edits. Creating
a file requires an existing parent and never overwrites a file. File listings and
search hits are capped at 200; visible file/diff output at 20,000 characters;
command stdout and stderr each at 8,000 characters, with truncation metadata.
Git catalogues with undecodable names or literal U+FFFD names are rejected
conservatively. Fingerprints cover tracked and nonignored untracked file contents;
ignored files, external dependencies and environmental changes are outside that
guarantee. Concurrent external filesystem writes are not transactionally locked.

Context budgets count serialized characters, including instructions, tool schemas
and pending tool results; they are not exact model token budgets. Compaction
uses a separate tool-free request, validates the summary, then atomically replaces
active context notes and advances a cursor over Session history. A failed summary
keeps the original context; an oversized input stops before sending it. Session
history and ready results remain intact for inspection and explicit saving.

## Save and resume

`/save /absolute/path/session.json` writes a version 3 JSON file with a nested Slick
Session snapshot, application context notes, archives, config, counters, workspace
metadata and edit observations. The path must
be new and outside the workspace. Publication is atomic and the file is mode 0600;
an existing destination or symlink is rejected. Sessions are capped at 20 MiB.
They contain conversation and observed source/output, so choose the path accordingly.
The app serializes no SDK clients or credentials; application command approvals
are not retained.

```bash
python -m examples.coding_harness --resume /absolute/path/session.json
python -m examples.coding_harness --resume /absolute/path/session.json --headless --task 'Continue the fix'
```

Resume defaults to the saved provider/model/root/config. Override the provider and
model with `--provider anthropic --model MODEL`, or use `--model MODEL` alone to
change models on the same provider. Changing providers requires an explicit model.
Workspace/config overrides and `--dry-run` are rejected on resume.
The saved workspace and Git repository must still exist; resume does not recreate them.
Loading validates the snapshot and never executes tools. On the next task, pending
requests execute and recorded results are reused. Version 1 and 2 files migrate
on load: old history remains context, and only explicit pending results become
ready work. Archived actions are never scheduled, and source files are not rewritten.
Changed HEAD or file fingerprints invalidate saved verification and add a fresh
workspace observation. A saved demo session cannot resume after its temporary
workspace has been removed. Nothing is saved automatically.

## Change the application

| File | Responsibility |
| --- | --- |
| `agent.py` | Session calls, tool budgets/UI events, verification and repair decisions |
| `workspace.py`, `process.py` | Bound Python tools, command decisions and subprocess cleanup |
| `verification.py` | Real check results and file fingerprints |
| `context.py` | Jinja instructions, selected skills, history presentation and size bounds |
| `session.py`, `state.py` | Plain application records and explicit JSON persistence |
| `tui.py`, `__main__.py` | Interaction, events, CLI and resource lifetime |
| `demo.py` | Offline scripted responses and disposable fixture |
| `../prompts/coding_harness/` | System/task/check/summary/checkpoint templates and Python skill |

Add a normal annotated, documented method to `Workspace` and pass its bound method
in `CodingAgent`'s `Session(..., tools=[...])` list. The method keeps access to instance state;
Slick derives its schema and validates arguments/results. No decorator is needed.
Edit the Jinja files to change instructions. Add a skill template to the explicit
`SKILLS` mapping to make it selectable. Retrieval, memory, extra application methods
and alternate verification are ordinary Python changes here.

`CodingAgent.session` owns model responses, tool requests and their results.
`Session.acall(context)` records a response and supplies ready results automatically;
`Session.resolve(request)` executes and records one tool. The harness keeps a short
loop around it for budgets and TUI events. Simpler apps can use
`await session.resolve_pending()`.

Application context notes contain tasks, verification feedback and checkpoints.
The Jinja renderer combines those with selected Session observations and never
replays entire previously rendered prompts. Compaction calls the raw provider
separately, preserving the main Session's pending/ready work. Workspace permissions,
verification, repair limits and command cleanup remain ordinary application code.

OpenRouter is available through LiteLLM. CLI harness providers, parallel tool
execution, durable execution, multimodal inputs and hosted/server tools are
outside this example. Provider capability limits still apply when changing models.

## Tests

```bash
python -m pytest tests/coding_harness
python -m pytest tests/test_session_calls.py tests/test_session_tools.py tests/test_session_snapshot.py
python -m pytest tests/test_tool_protocol.py tests/test_tool_providers.py tests/test_tool_exchange.py tests/test_sdk_transport.py
```

Tests use scripted responses, temporary repositories and real local processes. Provider
transport tests use mock HTTP with the optional installed SDKs. UI tests require
the example requirements; other tests and the headless demo run without Textual.
No automated tests call paid endpoints.

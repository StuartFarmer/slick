# Coding harness

A coding agent expressed as a conversation list and a loop:

1. Render the conversation with Jinja and call a Slick provider.
2. Run the requested workspace methods and append their results.
3. When the model finishes, run the configured checks.
4. Give failures back to the model and repeat within a repair budget.

The code uses Slick's prompts, provider transport, and tool schema generation.
The loop owns the conversation explicitly. See [`../session.py`](../session.py)
for the separate example of Slick's automatic `Session` bookkeeping.

## Try it offline

From a checkout on macOS or Linux, with Python 3.10+, Git and ripgrep installed:

```bash
python -m pip install -e .
python -m examples.coding_harness --dry-run --headless --task 'Fix the total calculation'
```

The scripted provider reads and edits a real temporary repository. Its first edit
fails a real unittest check; it reads the failure, repairs the code, and passes.
No credentials or model calls are involved. The temporary repository is removed
on exit. The demo handles this fixed task; it is not an intelligent offline model.

For the terminal interface:

```bash
python -m pip install -r examples/coding_harness/requirements.txt
python -m examples.coding_harness --dry-run
```

## Use a model

Install `python -m pip install -e '.[api]'`, set `OPENAI_API_KEY`, and supply a
model that supports native function tools:

```bash
python -m examples.coding_harness --model YOUR_MODEL_ID --workspace /absolute/repo
```

OpenAI is the default. For Anthropic, set `ANTHROPIC_API_KEY` and add
`--provider anthropic`. For LiteLLM/OpenRouter, install the `litellm` extra, set
`OPENROUTER_API_KEY`, and use:

```bash
python -m examples.coding_harness --provider litellm --model 'openrouter/openai/gpt-oss-120b:nitro' --workspace /absolute/repo
```

Add `--headless --task 'Your task'` to run without the terminal frontend.
Headless execution does not import Textual and denies commands beyond configured
checks. Real runs make multiple model requests.

A new workspace is created and initialized with Git. An existing repository must
be selected at its worktree root. Existing user changes are retained. The harness
does not automatically commit, switch branches, install dependencies, or roll back.
`--dry-run` owns its temporary workspace and cannot be combined with live-provider,
model, workspace, config, or resume options.

## Checks and limits

Pass `--config /absolute/checks.json`. Only this explicitly selected file is read:

```json
{
  "checks": [{"name": "unit", "argv": ["python", "-m", "pytest", "-q"], "timeout": 120}],
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

Omitted settings use these limits, with no checks or skills. Use the intended
Python executable in check commands. Missing dependencies produce failures.
Limits are positive integers; the soft context limit must be below the hard limit.
Context limits approximate request size in serialized characters, not model tokens.

Checks run before the task and after each proposed completion. Success requires
passing checks on a stable, fresh workspace fingerprint. A failing baseline never
waives a final failure. Repeated failure without changed files and exhausted budgets
stop the loop. Changes to recognizable test/config files or explicit check scripts
are highlighted; this is a heuristic, not complete dependency analysis.

| Result | Meaning | Headless exit |
| --- | --- | --- |
| `verified` | Configured checks passed on stable, fresh files | 0 |
| `unverified` | No checks were configured | 2 |
| `blocked` | Checks still fail, or a budget was reached | 2 |
| `failed` | Provider or application error | 1 |
| `cancelled` | Interrupted; completed edits remain | 130 |

## Conversation and commands

The terminal shows conversation, tool output, and check results directly. Enter
submits a task. Escape cancels active work or denies an approval. Ctrl+D shows the
current diff; Ctrl+Q quits after cleanup. `/help` lists the available controls.

- `/new` clears the conversation and captures a fresh baseline; files remain.
- `/compact` replaces older messages with a summary, retaining the most recent
  assistant/tool exchange. Failed summaries leave the conversation intact.
- `/save PATH` writes an idle conversation to a new file outside the workspace.

```bash
python -m examples.coding_harness --resume /absolute/session.json
python -m examples.coding_harness --resume /absolute/session.json --headless --task 'Continue'
```

Sessions store the conversation, configuration, workspace fingerprint and provider
identity. They contain source/output observations, but no credentials or approval
allowances. Save is atomic, refuses existing files/symlinks, and uses mode 0600;
save/load is limited to 20 MiB. Nothing is saved automatically.

**This redesign uses session format 2. Earlier harness snapshots are rejected with
an explicit error.** Resume restores conversation, never schedules recorded tools,
and starts fresh task budgets. Interrupted calls stay observations; the model must
explicitly request any retry. The workspace must still exist. Changed files add a
fresh warning, and every task reruns checks. Saved verification is never trusted.
Compacted text replaces old messages; there is no separate in-memory archive.

Resume defaults to the saved provider/model. Override the model with `--model`, or
supply both `--provider` and `--model` to change providers. Workspace/config overrides
are rejected. A saved demo cannot resume after its temporary repository is removed.

## Local execution boundaries

File tools reject traversal, symlinks and `.git` paths. Reads are bounded UTF-8;
edits require a fresh full-file SHA256 and one unique exact match. Edits are staged
and replaced atomically; creates never overwrite. Commands are argv lists, with
workspace cwd and disconnected stdin. Exact configured checks run automatically;
other commands require allow-once or allow-for-session approval. Denial executes
nothing. The complete argv, cwd and timeout are shown before approval.

Approved commands run with host permissions, including network access; this is
not an OS sandbox. Timeout/cancellation kills and reaps the owned process group.
Processes that escape it and concurrent external filesystem writes are outside
these guarantees. Background jobs are unsupported. Output is bounded and reports
truncation. Fingerprints cover tracked and nonignored untracked files, not ignored
files, dependencies or environmental changes.

## Read and change the code

| File | Job |
| --- | --- |
| `agent.py` | One conversation, explicit provider/tool/check loop, task budgets |
| `workspace.py` | Direct file/search/command/diff methods and their guards |
| `checks.py` | Run checks and format real feedback |
| `session.py` | Save/load a conversation and summarize older messages |
| `ui.py` | Named output methods; plain headless output |
| `tui.py` | Terminal input, approvals, diff display, and cancellation |
| `process.py` | Bounded subprocess output and owned-process cleanup |
| `config.py` | Validate explicitly supplied settings |
| `__main__.py`, `demo.py` | Startup/provider selection and the offline fixture |
| `../prompts/coding_harness/` | System, conversation, summary, and skill templates |

To add a tool, write an annotated method with a short docstring on `Workspace`
and add the bound method to the explicit list in `CodingAgent`. Return ordinary
data; Slick derives the schema and serializes the result. No tool subclasses or
registration decorators are required.

Prefer direct functions and visible sequencing. Use a class when it owns state or
integrates with the terminal framework. Validate at inputs and effect boundaries.
Explain why a safeguard exists instead of narrating the code. Add a module when it
makes a responsibility easier to understand, not to satisfy a file-length target.

```bash
python -m pytest tests/coding_harness
python -m ruff check examples/coding_harness tests/coding_harness
```

Tests cover real workspace effects, approvals, cancellation, repair, conversation
continuity, inert save/load and failed compaction. Provider tests use offline
responses or mock HTTP; they do not call paid endpoints.

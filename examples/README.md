# Runnable patterns

These are ordinary Python application classes using Slick's `@prompt` and `Session`.
Decorators render templates, generate typed results, and pass them to Python bodies
as `generated` for validation and state updates. Each class owns its state and methods. Sequencing, retrieval,
tool execution, voting, and search are Python code you can change directly.
All templates live here in `examples/prompts/`; none are part of the Slick package.

From the repository root, with Slick installed (`pip install -e .`), run any
example independently:

| Example | Command | Object and operations |
| --- | --- | --- |
| Shared context primitives | `python -m examples.primitives` | Render all seven parts without a provider |
| Conversation | `python -m examples.question_answerer --critique` | `QuestionAnswerer.ask`, `critique`; instance-owned history |
| Automatic Session | `python -m examples.session` | `@prompt` with a tool-enabled session and a four-turn limit |
| Few-shot | `python -m examples.few_shot` | `FewShotAnswerer.ask`; classify using input/output examples |
| Chain-of-thought | `python -m examples.chain_of_thought` | `ReasoningSolver.solve`; worked examples and a concise, checkable explanation |
| Retrieval-grounded answering | `python -m examples.rag` | `GroundedAnswerer.retrieve`, `ask`; local retrieval and cited answers |
| Least-to-most | `python -m examples.least_to_most` | `LeastToMost.decompose`, `solve_next`, `run`; earlier answers feed later calls |
| Self-Refine | `python -m examples.self_refine --rounds 2` | `SelfRefiner.draft`, `critique`, `revise`, `run`; answer and feedback history |
| Self-consistency | `python -m examples.self_consistency --samples 5` | `SolutionSampler.sample`, `choose`, `run`; repeated solutions and answer voting |
| ReAct | `python -m examples.react --max-steps 4` | `Researcher.step`, `run`; selected actions, Python tools, actual observations |
| Reflexion | `python -m examples.reflexion --attempts 3` | `ReflectiveSolver.attempt`, `evaluate`, `reflect`, `run`; checker feedback and retained lessons |
| Tree of Thoughts | `python -m examples.tree_of_thoughts --depth 2 --width 2 --breadth 2` | `ThoughtSearch.expand`, `evaluate`, `select`, `run`; bounded beam search |
| Paper to plan | `python -m examples.paper_plan` | Explicit assessments, ordered tasks, and feedback-driven revision |

The default `--provider demo` uses canned responses and requires no credentials.
Templates, validation, state changes, tool execution, and loops still run. Canned
answers demonstrate the default tasks; changing the task does not make the demo
responses intelligent. Use `--help` on each module to see its input and budget flags.
The functional example is available as `python -m examples.core`.

## OCR paper to implementation plan

Run the fictional paper with canned responses, without credentials:

```bash
python -m examples.paper_plan
```

Analyze your own UTF-8 OCR Markdown using a real provider:

```bash
python -m examples.paper_plan /path/to/paper.md --provider openai --model YOUR_MODEL_ID > plan.md
```

[paper_plan.py](paper_plan.py) has four explicit operations, each using its own Jinja
prompt declared with `@prompt`. Each call generates before running its Python body:

1. `assess_method()` identifies implementation requirements and missing information.
2. `assess_evaluation()` identifies claims, baselines, and measurements.
3. `generate_plan(method, evaluation)` drafts an ordered list of tasks with acceptance criteria.
4. `revise(draft, feedback, provider=...)` returns a new plan using the original assessments and your feedback.

`run()` runs both assessments concurrently, waits for them, then generates the plan:
three model calls when no tools are requested. The provider must support concurrent calls. Each assessment
contains source quotations, which Python checks against the supplied paper. Missing
information stays visible as plain questions; proposed choices are listed as assumptions.
Tasks are an ordered list, with no IDs, routes, or dependency graph.

```python
from pathlib import Path
from slick import prompts
from examples.paper_plan import PaperPlanner, format_report

prompts.TEMPLATE_ROOT = Path("examples/prompts")
paper = Path("paper.md").read_text(encoding="utf-8")
planner = PaperPlanner(paper, provider)

draft = await planner.run()
print(format_report(draft))

# After reviewing the draft, make one explicit revision call.
revised = await planner.revise(
    draft, "Separate data preparation from model fitting.", provider=provider,
)
print(format_report(revised))
```

Direct calls to decorated methods require either `provider=` or `session=`.
Assessments use the planner's provider directly. Generation and review revisions
use an optional caller-supplied session, or a fresh session with that provider.
Function bodies check generated evidence or wrap the
generated plan. Revision uses `@validate_call` to reject blank feedback before generation
and preserves the original draft. Provider errors, invalid JSON, and invalid
quotations propagate to the caller. Pydantic validates the output structure; it does
not establish scientific correctness or plan completeness. Your application decides
whether to retry, approve, or save a result with `result.model_dump()`.

To let planning inspect project constraints, supply a session with ordinary tools:

```python
from slick import Session

def project_notes() -> str:
    """Read the project's implementation constraints."""
    return Path("PROJECT.md").read_text(encoding="utf-8")

session = Session(provider=provider, tools=[project_notes])
draft = await planner.run(session=session)
revised = await planner.revise(draft, "Check project constraints.", session=session)
# Or pass session=session to planner.review(draft, inbox).
```

Generation and revision decorators allow up to 20 model turns. The session runs
requested tools and feeds results back before the body receives a typed `Plan`.
The assessments remain paper-only; the supplied session retains planning history
across revisions. A `Workflow` records individual model exchanges and completed
tools, so an interrupted conversation resumes with its prior results.

The shared `--provider`, `--model`, and `--timeout` options work as above. The example
reads the entire paper, so it must fit the model's context window. By default it does
not browse links or execute experiments. CLI providers retain their own harness capabilities,
although these prompts request paper-only analysis. The offline demo contains only
the three initial responses; use a real provider for revisions or your own paper.

### External review

Pass a SQLite inbox path to pause after generation:

```bash
python -m examples.paper_plan --inbox reviews.db
```

The example awaits `inbox.request(draft, output_type=ReviewDecision)`. A CLI,
webhook, or API adapter discovers drafts through the same database file:

```python
from slick import Inbox
from examples.paper_plan import PaperPlan, ReviewDecision, format_report

inbox = Inbox("reviews.db")
for channel, message in inbox.pending():
    draft = PaperPlan.model_validate(message)
    print(format_report(draft))
    decision = ReviewDecision.model_validate_json(input("Decision JSON: "))
    inbox.send(channel, decision)
```

Decisions are `{"action": "approve"}`, `{"action": "reject"}`, or
`{"action": "revise", "feedback": "Separate preparation from fitting."}`.
Revision requires nonblank feedback and a real provider (or additional scripted
responses). Each revision publishes a new draft on a new channel, discoverable on
the next call to `pending()`. Approval returns the current plan; rejection exits with status 1.
Invalid decisions and model errors propagate. The inbox itself accepts any JSON;
`ReviewDecision` and the loop belong entirely to this example.

In application code, call `await planner.review(draft, inbox)` after `planner.run()`.
Listing requests does not consume them; they remain discoverable until a reply is
sent. `pending()` is a snapshot, so multiple reviewers can see the same draft.
Identical duplicate submissions are harmless; conflicting replies are rejected.
Invalid typed replies reopen the request and propagate a validation error. Reviewer
assignment belongs to the adapter.

### Restart recovery

Add a stable run ID to record execution and recover after a process restart:

```bash
python -m examples.paper_plan --inbox reviews.db --run-id paper-42
```

Run that same command again to recover. The CLI orchestration and `review()` use
`@workflow`; the ordinary calls and loop stay intact. Completed generation and
revision results are restored as `PaperPlan` objects. An unanswered review reconnects
to its existing channel, and a reply sent while the planner was stopped is retained.
Without `--run-id`, each invocation starts a new plan.

The initial `planner.run()` is one recorded operation, including its concurrent
assessments. A failure inside that operation before its result is saved can repeat
all three initial calls. Once the draft is recorded, restarting at a review point
does not regenerate it. Each subsequent revision and review request is recorded
separately. Code between recorded awaits executes again and must be deterministic.

The paper is saved as a run input and must match on recovery. Supply providers again;
their live clients are not serialized. Use a new run ID after changing the paper,
workflow, or prompt templates. Two processes cannot own the same run concurrently.
The inbox database and its adjacent `.locks` directory must remain on local storage.

## Coding harness

The [coding harness example](coding_harness/README.md) adds a native tool loop,
workspace edits, real checks and bounded repair to an ordinary `CodingAgent` class.
It defaults to OpenAI: set `OPENAI_API_KEY` and supply a model and Git workspace.

```bash
python -m pip install -e '.[api]'
python -m pip install -r examples/coding_harness/requirements.txt
python -m examples.coding_harness --model YOUR_MODEL_ID --workspace /absolute/repo
```

Use `--dry-run` for the offline scripted demo:

```bash
python -m examples.coding_harness --dry-run
python -m examples.coding_harness --dry-run --headless --task 'Fix the total calculation'
```

Textual is optional for headless runs. The harness has its own
CLI options, sessions and explicit check configuration; it supports OpenAI Responses
and Anthropic Messages, plus LiteLLM tool calls (including OpenRouter).
See its README for the OpenRouter setup. Its prompts remain under `examples/prompts/`.
The harness keeps a plain conversation list for display and saving. A fresh Slick
`Session` per task records exchanges, resolves tools, and submits their results.
Its explicit loop enforces request/tool budgets, updates the UI, and checks edits.
Compaction uses `@prompt` with a session offering no tools. Version-2 saved
conversations retain their format; loading never replays tools.

## Use a real provider

Every pattern command accepts `--provider`, `--model`, and `--timeout` (seconds per
call). The primitives command only renders text. For API calls, install the
optional SDKs and set the corresponding `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`:

```bash
pip install -e '.[api]'
python -m examples.self_refine --provider openai --model YOUR_MODEL_ID --task "Explain DNS"
python -m examples.rag --provider anthropic --model YOUR_MODEL_ID --question "How do templates compose?"
```

API model IDs are required. CLI harnesses use their installed executable and
authentication, with an optional model override:

```bash
python -m examples.question_answerer --provider codex --question "Explain DNS"
python -m examples.self_refine --provider claude --task "Explain DNS" --rounds 1
```

These options make real calls. Multi-step patterns make multiple calls; loop
budgets belong to the example, while the timeout applies to each provider call.
Structured examples declare `output_type=` on `@prompt`; the decorator supplies
the JSON schema and validates responses before their Python bodies run.

## Reuse the classes

Set the template root when using the examples from your own Python code; the
example commands do this for you. For example, in an async application launched
from the repository root:

```python
from pathlib import Path

from slick import prompts
from examples.self_refine import SelfRefiner

prompts.TEMPLATE_ROOT = Path("examples/prompts")
writer = SelfRefiner("Explain DNS", provider, ["Accuracy", "Clarity"])
draft = await writer.draft(provider=provider)
feedback = await writer.critique()
answer = await writer.revise()
```

Here `provider` is any object with `async acall(context) -> tuple[str, list[dict]]`.
Direct decorated calls require exactly one of `provider=` or `session=`, as in
`paper_plan.py`. Method templates access instance state through `instance` and
arguments by name. `output_type=` controls generated data; return annotations
describe the postprocessed result. Templates use `{{ output_format }}` to place
the decorator's output instructions without duplicating schemas.

Ordinary orchestration methods such as `run()`, `critique()`, `solve_next()`, and
`step()` supply their instance's provider to decorated calls. Retrieval and
precondition/budget checks run before generation; citation checks, voting, and
state updates stay in Python. Instances are intended for sequential use, and their
state stays in memory. A new CLI invocation starts a new instance.

`examples.session` passes a tool-enabled `Session` to a decorated function; Slick
runs the tool conversation up to `max_turns=4` and returns the final answer. Use
`Prompt` directly for rendering alone, as in `examples.primitives` and the paper
report formatter.

## Shared Jinja parts

Each part is a macro in `prompts/parts/<name>.j2`:

| Macro | Plain Python input |
| --- | --- |
| `conversation(messages)` | List of dictionaries with `role`, `content` |
| `evidence(documents)` | List of dictionaries with `id`, `source`, `text` |
| `examples(pairs)` | List of dictionaries with `input`, `output` |
| `instructions(task, audience, constraints)` | Two strings and a list of constraint strings |
| `rubric(criteria)` | List of criterion strings |
| `working_state(state)` | Dictionary with `facts`, `decisions`, `questions`, each a list of strings |
| `observations(records)` | List of dictionaries with `action`, `result` |

Compose them in any template:

```jinja
{% from "parts/conversation.j2" import conversation %}
{% from "parts/evidence.j2" import evidence %}
{% from "parts/examples.j2" import examples %}
{% include "instructions/research.j2" %}

{{ conversation(messages) }}
{{ evidence(documents) }}
{{ examples(demonstrations) }}

Question: {{ question }}
```

Python selects the data. These macros only present it as text: conversation roles
are rendered labels, not provider-native messages. The reusable research
instructions are a regular include. Executable tools are separate Python methods;
see `Researcher.tools` and `search_documents` for the explicit dispatch loop.

## What the examples demonstrate

These are small adaptations of prompting patterns, not reproductions of research
benchmarks. Chain-of-thought uses a worked example and asks for a brief explanation;
its usefulness depends on the model and task. Self-consistency casefolds and
collapses whitespace before voting, with ties resolved by first appearance. It
does not normalize mathematical equivalence or guarantee diverse samples.

RAG uses lexical retrieval so it runs independently; replace `retrieve` with a
vector store or any other Python lookup. RAG and ReAct validate citation identifiers,
which does not establish that the cited text supports every claim. ReAct's only
example tool searches a small local corpus, and observations contain its actual
results.

Reflexion has a deterministic toy checker: exactly three distinct positive even
integers summing to 18. Its `--task` supplies extra guidance within that task;
change `evaluate` to support a different problem. Lessons feed subsequent attempts
as template data. Self-Refine uses model feedback instead of an independent checker.

Tree of Thoughts expands partial states, scores them, and retains a bounded beam.
Scores come from the provider, so a reported complete answer is not independently
verified. ReAct and Reflexion exit with status 1 when their budgets are exhausted;
Tree of Thoughts does so when it has only a partial state at cutoff. Invalid
structured responses raise validation errors; these examples do not add retries.

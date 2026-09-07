# Runnable patterns

These are ordinary Python application classes using Slick's `Prompt`, backend
`acall`, and `parse`. Each class owns its state and methods. Sequencing, retrieval,
tool execution, voting, and search are Python code you can change directly.
All templates live here in `examples/prompts/`; none are part of the Slick package.

From the repository root, with Slick installed (`pip install -e .`), run any
example independently:

| Example | Command | Object and operations |
| --- | --- | --- |
| Shared context primitives | `python -m examples.primitives` | Render all seven parts without a backend |
| Conversation | `python -m examples.question_answerer --critique` | `QuestionAnswerer.ask`, `critique`; instance-owned history |
| Few-shot | `python -m examples.few_shot` | `FewShotAnswerer.ask`; classify using input/output examples |
| Chain-of-thought | `python -m examples.chain_of_thought` | `ReasoningSolver.solve`; worked examples and a concise, checkable explanation |
| Retrieval-grounded answering | `python -m examples.rag` | `GroundedAnswerer.retrieve`, `ask`; local retrieval and cited answers |
| Least-to-most | `python -m examples.least_to_most` | `LeastToMost.decompose`, `solve_next`, `run`; earlier answers feed later calls |
| Self-Refine | `python -m examples.self_refine --rounds 2` | `SelfRefiner.draft`, `critique`, `revise`, `run`; answer and feedback history |
| Self-consistency | `python -m examples.self_consistency --samples 5` | `SolutionSampler.sample`, `choose`, `run`; repeated solutions and answer voting |
| ReAct | `python -m examples.react --max-steps 4` | `Researcher.step`, `run`; selected actions, Python tools, actual observations |
| Reflexion | `python -m examples.reflexion --attempts 3` | `ReflectiveSolver.attempt`, `evaluate`, `reflect`, `run`; checker feedback and retained lessons |
| Tree of Thoughts | `python -m examples.tree_of_thoughts --depth 2 --width 2 --breadth 2` | `ThoughtSearch.expand`, `evaluate`, `select`, `run`; bounded beam search |

The default `--backend demo` uses canned responses and requires no credentials.
Templates, validation, state changes, tool execution, and loops still run. Canned
answers demonstrate the default tasks; changing the task does not make the demo
responses intelligent. Use `--help` on each module to see its input and budget flags.
The earlier functional example remains available as `python -m examples.core`.

## Use a real backend

Every pattern command accepts `--backend`, `--model`, and `--timeout` (seconds per
call). The primitives command only renders text. For API calls, install the
optional SDKs and set the corresponding `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`:

```bash
pip install -e '.[api]'
python -m examples.self_refine --backend openai --model YOUR_MODEL_ID --task "Explain DNS"
python -m examples.rag --backend anthropic --model YOUR_MODEL_ID --question "How do templates compose?"
```

API model IDs are required. CLI harnesses use their installed executable and
authentication, with an optional model override:

```bash
python -m examples.question_answerer --backend codex --question "Explain DNS"
python -m examples.self_refine --backend claude --task "Explain DNS" --rounds 1
```

These options make real calls. Multi-step patterns make multiple calls; loop
budgets belong to the example, while the timeout applies to each backend call.
Structured examples put their JSON schema in the prompt and validate responses
locally with `parse`.

## Reuse the classes

Set the template root when using the examples from your own Python code; the
example commands do this for you. For example, in an async application launched
from the repository root:

```python
from pathlib import Path

from slick import prompts
from examples.self_refine import SelfRefiner

prompts.TEMPLATE_ROOT = Path("examples/prompts")
writer = SelfRefiner("Explain DNS", backend, ["Accuracy", "Clarity"])
draft = await writer.draft()
feedback = await writer.critique()
answer = await writer.revise()
```

Here `backend` is any object with `async acall(text) -> str`. A prompt stores only
its template filename; the class decides when to render and execute it. Instances
are intended for sequential use, and their state stays in memory. A new CLI
invocation starts a new instance.

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
Scores come from the backend, so a reported complete answer is not independently
verified. ReAct and Reflexion exit with status 1 when their budgets are exhausted;
Tree of Thoughts does so when it has only a partial state at cutoff. Invalid
structured responses raise validation errors; these examples do not add retries.

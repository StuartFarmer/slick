# Jinja primitives and prompting patterns for Slick

Date: 2026-09-07. Proposal only; the core is implemented, the template kit and recipes below are not.

The next reusable layer can be a small collection of Jinja macros/templates and ordinary Python recipes. Most suggestions need no change to `render`, `parse`, `@prompt`, or the text backend contract. A template primitive has explicit data arguments and produces text; a recipe controls calls and returns an ordinary Python value.

## Reusable template/data primitives

| Primitive | Arguments | Rendered purpose | Python responsibility |
|---|---|---|---|
| Conversation | `messages: list[{role, content}]` | Format prior turns in order | Load, append, retain or trim messages |
| Evidence | `documents: list[{id, text, source}]` | Present references with stable citation IDs | Retrieve/select documents and check citations |
| Examples | `examples: list[{input, output}]` | Demonstrate expected behavior and format | Select relevant examples |
| Instructions / skill | Explicit task parameters, audience, constraints | Reusable procedure or domain instructions | Choose the template and executable capabilities |
| Rubric | Criteria and optional weights | Define how a candidate should be evaluated | Interpret scores, run real checks, decide whether to revise |
| Working state | Goal, known facts, decisions, unresolved items | Carry selected task context between calls | Update it explicitly; summarize only when requested |
| Action/observation history | Ordered requested actions and returned results | Ground the next action in previous execution | Execute approved functions and append actual results |
| Output contract | Existing return annotation / `output_format` | State expected JSON shape | Existing Pydantic validation |

These names describe data and formatting conventions, not required classes. Start with lists/dicts and macros with explicit arguments. Pydantic types are useful where their validation has a concrete purpose, especially action or critique outputs. A conversation template does not create API-level message roles or storage. A skill template does not register tools. Template content is text, and model output is data.

For example, a local `parts/context.j2` can contain:

```jinja
{% macro conversation(messages) -%}
{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}
{%- endmacro %}
```

A task template used by `@prompt` imports it normally:

```jinja
{% from "parts/context.j2" import conversation %}
{% include "instructions/answer.j2" %}

{{ conversation(messages) }}

Question: {{ question }}

{{ output_format }}
```

`@prompt` supplies `output_format`. Standalone `render` requires it as an explicit variable when the template references it, or the template can omit that slot.

Render history summaries as explicit working-state fields. Keep a bounded recent history using Python slicing if appropriate. Token budgeting, information-preserving summarization, and message retention are separate application decisions; a formatting macro does not solve them automatically.

## Pattern comparison

Confidence below concerns the documented procedure and its mapping to Slick, not expected quality on the user's workload. Published results are task/model-specific; the suggested recipes are adaptations, not reproductions of the papers' experiments.

| Pattern / source | Key claim or procedure | Template + Python mapping | Evidence / caveat | Confidence |
|---|---|---|---|---|
| Few-shot demonstrations | Supply representative input/output examples | Examples macro followed by the task; typically one call | Ordinary prompting convention; example selection remains application work | High for implementation |
| [Chain-of-thought](https://arxiv.org/abs/2201.11903) | Worked intermediate reasoning examples can elicit multi-step solutions | Worked-solution examples and output instructions | Experimental paper; model/task dependent. Do not make verbose reasoning a universal default or claim access to internal reasoning | High for mechanism |
| [RAG](https://arxiv.org/abs/2005.11401) / retrieval-grounded answering | Combine retrieved external information with generation | Python retrieves; evidence macro renders documents; a function answers | Original paper studies a trained retrieval/generation architecture. A fetch-and-prompt recipe is a practical adaptation | High for broad mapping |
| [Least-to-most](https://arxiv.org/abs/2205.10625) | Decompose a problem and solve subproblems sequentially | Decomposition and solve templates; Python carries earlier answers forward | Experimental paper; a plan in one prompt is not the entire multi-call procedure | High |
| [Self-Refine](https://arxiv.org/abs/2303.17651) | Generate, critique and revise using the same underlying model | Draft, feedback and revision templates; bounded Python loop | Uses prior revisions/feedback as context; self-feedback is not an independent correctness oracle | High |
| [Self-consistency](https://arxiv.org/abs/2203.11171) | Sample diverse reasoning paths and aggregate final answers | Repeat a prompt function; Python normalizes equivalent answers and votes | Needs useful diversity and an answer equivalence rule. Current API adapters do not expose sampling controls; use backend capabilities or a small later extension when needed | High for procedure; workload benefit untested |
| Candidate ranking / best-of-N | Generate candidates and select by a score or judge | Candidate and rubric templates; Python chooses a result | Different from self-consistency; a judge can share the generator's errors | High for implementation |
| [ReAct](https://arxiv.org/abs/2210.03629) | Interleave reasoning, actions and environmental observations | Next-action template plus actual Python execution and accumulated observations | Merely rendering action labels does not execute anything. Typed action-only variants should be called ReAct-inspired | High |
| [Reflexion](https://arxiv.org/abs/2303.11366) | Reflect on feedback from attempts and reuse verbal lessons in later trials | Evaluation/reflection templates; explicit `trial_reflections` passed to future attempts | Distinction from Self-Refine is across-trial reflective memory, not that only Reflexion retains history; no model-weight updates | High |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | Generate/evaluate intermediate candidates and search among them | Candidate/evaluation templates with a Python frontier, pruning and backtracking | More complex execution/cost; branch generation without search is not the complete method | High for mechanism; defer implementation |

The papers broadly agree that behavior can be changed by context and inference-time procedures without necessarily retraining the model. They differ in what is repeated and selected: answers, revisions, actions, trial reflections or partial solution states. They do not establish one universally best prompting pattern.

## ReAct-inspired sketch using the existing core

This is an illustrative single-tool recipe. It uses typed text output; native provider tool calling is not implemented by the current adapters. `backend`, `search`, and `next_action.j2` are supplied by the application.

```python
from typing import Literal
from pydantic import BaseModel
from slick import prompt

class Search(BaseModel):
    kind: Literal["search"]
    query: str

class Finish(BaseModel):
    kind: Literal["finish"]
    answer: str

@prompt(backend=backend, template="next_action.j2")
async def next_action(task: str, observations: list[dict]) -> Search | Finish:
    """Select a search query or return the final answer."""

async def investigate(task: str) -> str:
    observations = []
    for _ in range(8):
        action = await next_action(task, observations)
        if isinstance(action, Finish):
            return action.answer
        result = await search(action.query)
        observations.append({"query": action.query, "result": result})
    raise RuntimeError("Search budget exhausted")
```

The corresponding template describes the available search action, presents previous observations and includes `{{ output_format }}`. Slick validates the selected action; Python invokes the search function. A general multi-tool version would need explicit name allowlisting, per-function argument validation and result serialization, which is a concrete later tool feature. Do not use `eval` or let model-provided names resolve arbitrary application code.

This sketch does not reproduce original ReAct's explicit reasoning traces. It preserves the operational action/observation cycle while requesting only the decision needed by application code.

## Recommended next work

1. Create an optional, inspectable template collection for conversation, evidence, examples, instructions, rubrics and observations. Initially use files under the existing template root; no template registry, loader hierarchy or skill object is needed.
2. Add three ordinary recipe examples: retrieval-grounded answering, critique/revise, and the typed single-tool loop above. Demonstrate each with deterministic fakes and compare live quality only in explicit experiments.
3. Evaluate whether helpers repeat across examples. Generic Python-function tool description/dispatch is a plausible next shared primitive; conversation persistence and a workflow engine are not needed to establish it.
4. Add self-consistency, trial reflection and search only for workloads that justify their extra calls. Measure task success and cost against a single-call baseline.

Existing code was inspected: `slick/prompts.py` provides render/parse/decorated call composition; `slick/backends.py` configures no native tools and rejects tool outputs; `examples/core.py` already demonstrates a history template and application retrieval. Standalone render does not fetch context. A decorated `.render()` can run its application-authored computed-context body, so it is not a guarantee that arbitrary user code performs no I/O.

Researcher checked the method distinctions against primary sources. Verifier checked these claims against the current Slick implementation. No runtime code was changed for this comparison.

## Sources

- ReAct: https://arxiv.org/abs/2210.03629
- Chain-of-thought: https://arxiv.org/abs/2201.11903
- Retrieval-Augmented Generation: https://arxiv.org/abs/2005.11401
- Least-to-most: https://arxiv.org/abs/2205.10625
- Self-Refine: https://arxiv.org/abs/2303.17651
- Self-consistency: https://arxiv.org/abs/2203.11171
- Reflexion: https://arxiv.org/abs/2303.11366
- Tree of Thoughts: https://arxiv.org/abs/2305.10601

"""ReAct-inspired local research loop; not a reproduction of original reasoning traces.

Run offline with ``python -m examples.react``. The application dispatcher executes
only allowlisted Python functions. A selected CLI harness retains its own tools.
"""

import asyncio
import json
import re
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from examples._cli import ScriptedBackend, backend_from_args, parser, positive_int
from slick import Prompt, parse

DEFAULT_GOAL = "When does Atlas launch, and what must happen first?"
DOCUMENTS = [
    {"id": "schedule", "source": "Atlas schedule", "text": "Atlas launches on October 12."},
    {
        "id": "readiness",
        "source": "Atlas readiness checklist",
        "text": "Before the Atlas launch, finish the security review and restore drill.",
    },
    {"id": "borealis", "source": "Borealis schedule", "text": "Borealis launches in November."},
]


class Search(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["search"]
    tool: str
    query: str = Field(min_length=1)


class Finish(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["finish"]
    answer: str = Field(min_length=1)
    citations: list[str]


Action = Annotated[Search | Finish, Field(discriminator="kind")]


class Researcher:
    """Keep a goal, observations, evidence, and a lifetime step budget per instance."""

    def __init__(self, backend, goal: str, *, documents=None, max_steps: int = 4):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.backend = backend
        self.goal = goal
        self.documents = [dict(doc) for doc in (DOCUMENTS if documents is None else documents)]
        self.tools = {"search_documents": self.search_documents}
        self.max_steps = max_steps
        self.steps = 0
        self.observations: list[dict] = []
        self.evidence: dict[str, dict] = {}
        self.result: Finish | None = None
        self.action_prompt = Prompt("react/action.j2")

    def search_documents(self, query: str) -> list[dict]:
        """Return local documents containing any complete query word, in corpus order."""
        words = set(re.findall(r"\w+", query.casefold()))
        return [
            dict(doc)
            for doc in self.documents
            if words.intersection(re.findall(r"\w+", f"{doc['source']} {doc['text']}".casefold()))
        ]

    async def step(self) -> Search | Finish:
        if self.result is not None:
            return self.result
        if self.steps >= self.max_steps:
            raise RuntimeError(f"Research step budget exhausted ({self.max_steps})")
        self.steps += 1
        text = self.action_prompt(
            goal=self.goal,
            tools=list(self.tools),
            records=self.observations,
            documents=list(self.evidence.values()),
            remaining=self.max_steps - self.steps,
            schema=TypeAdapter(Action).json_schema(),
        )
        action = parse(await self.backend.acall(text), Action)
        if isinstance(action, Search):
            if action.tool not in self.tools:
                raise ValueError(f"Tool is not allowlisted: {action.tool}")
            results = self.tools[action.tool](action.query)
            self.observations.append({"action": action.model_dump(), "result": results})
            self.evidence.update((doc["id"], doc) for doc in results)
        else:
            if set(action.citations) - self.evidence.keys():
                raise ValueError("Finish cites evidence that has not been observed")
            self.result = action
        return action

    async def run(self) -> Finish:
        while self.result is None:
            await self.step()
        return self.result


def main(argv=None) -> int:
    argument_parser = parser(__doc__)
    argument_parser.add_argument("--goal", default=DEFAULT_GOAL)
    argument_parser.add_argument("--max-steps", type=positive_int, default=4)
    args = argument_parser.parse_args(argv)
    demo = ScriptedBackend(
        [
            '{"kind":"search","tool":"search_documents","query":"Atlas"}',
            '{"kind":"finish","answer":"Atlas launches on October 12, after the security '
            'review and restore drill.","citations":["schedule","readiness"]}',
        ]
    )
    backend = backend_from_args(args, argument_parser, demo)
    researcher = Researcher(backend, args.goal, max_steps=args.max_steps)
    try:
        result = asyncio.run(researcher.run())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    print(f"Steps: {researcher.steps}; observations:")
    print(json.dumps(researcher.observations, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

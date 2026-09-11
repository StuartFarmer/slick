"""A bounded beam search over partial solution states, inspired by Tree of Thoughts.

This teaching example is not a reproduction of the paper's experiments. Model
scores guide pruning; they do not independently verify a completed answer.
"""

import asyncio
import json
import math
from dataclasses import dataclass, replace

from pydantic import BaseModel, Field, constr

from examples._cli import ScriptedProvider, parser, positive_int, provider_from_args
from slick import prompt

NonemptyText = constr(strip_whitespace=True, min_length=1)


class Candidate(BaseModel):
    step: NonemptyText
    answer: NonemptyText | None = None


class Expansion(BaseModel):
    candidates: list[Candidate]


class Evaluation(BaseModel):
    score: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)


@dataclass(frozen=True)
class ThoughtState:
    steps: tuple[str, ...] = ()
    answer: str | None = None
    score: float | None = None


class ThoughtSearch:
    def __init__(
        self, problem: str, provider, max_depth: int = 2, beam_width: int = 2, breadth: int = 2
    ):
        for name, value in [
            ("max_depth", max_depth),
            ("beam_width", beam_width),
            ("breadth", breadth),
        ]:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.problem = problem
        self.provider = provider
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.breadth = breadth
        self.frontier = [ThoughtState(score=0.0)]

    def _context(self, node: ThoughtState) -> dict:
        return {
            "facts": [self.problem],
            "decisions": node.steps,
            "questions": [] if node.answer else ["Complete the solution."],
        }

    async def expand(self, node: ThoughtState) -> list[ThoughtState]:
        if node.answer is not None or len(node.steps) >= self.max_depth:
            return []
        return await self.generate_children(node, provider=self.provider)

    @prompt(template="tree_of_thoughts/expand.j2", output_type=Expansion)
    async def generate_children(
        self, node: ThoughtState, *, generated: Expansion
    ) -> list[ThoughtState]:
        if len(generated.candidates) > self.breadth:
            raise ValueError("expansion exceeds breadth")
        return [
            ThoughtState(steps=(*node.steps, item.step), answer=item.answer)
            for item in generated.candidates
        ]

    @prompt(template="tree_of_thoughts/evaluate.j2", output_type=Evaluation)
    async def evaluate(self, node: ThoughtState, *, generated: Evaluation) -> ThoughtState:
        return replace(node, score=generated.score)

    def select(self, candidates: list[ThoughtState]) -> list[ThoughtState]:
        """Keep the highest scores; equal scores preserve expansion order."""
        if any(
            node.score is None or not math.isfinite(node.score) or not 0 <= node.score <= 1
            for node in candidates
        ):
            raise ValueError("evaluate every candidate before selecting")
        self.frontier = sorted(candidates, key=lambda node: node.score, reverse=True)[
            : self.beam_width
        ]
        return self.frontier

    async def run(self) -> ThoughtState:
        """Return a completed retained state, or the best partial state at cutoff."""
        self.frontier = [ThoughtState(score=0.0)]
        for _ in range(self.max_depth):
            candidates = []
            for node in self.frontier:
                for child in await self.expand(node):
                    candidates.append(await self.evaluate(child, provider=self.provider))
            if not candidates:
                break
            self.select(candidates)
            for node in self.frontier:
                if node.answer is not None:
                    return node
        return self.frontier[0]


def demo_responses(depth: int, width: int, breadth: int):
    """Script the same bounded traversal, completing the route on the second step."""
    parents = 1
    for level in range(1, min(depth, 2) + 1):
        for parent in range(parents):
            candidates = [
                {"step": f"Take road {index + 1} from A to B."}
                if level == 1
                else {"step": f"Take road {index + 1} from B to C.", "answer": "A -> B -> C"}
                for index in range(breadth)
            ]
            yield json.dumps({"candidates": candidates})
            for index in range(breadth):
                yield json.dumps({"score": 1 / (parent * breadth + index + 1)})
        parents = min(width, parents * breadth)


def main(argv=None):
    argument_parser = parser(__doc__)
    argument_parser.add_argument("--problem", default="Find a two-step route from A to C via B.")
    argument_parser.add_argument("--depth", type=positive_int, default=2)
    argument_parser.add_argument(
        "--width", type=positive_int, default=2, help="retained states per level"
    )
    argument_parser.add_argument(
        "--breadth", type=positive_int, default=2, help="children per state"
    )
    args = argument_parser.parse_args(argv)
    provider = provider_from_args(
        args,
        argument_parser,
        ScriptedProvider(
            demo_responses(args.depth, args.width, args.breadth),
        ),
    )
    search = ThoughtSearch(args.problem, provider, args.depth, args.width, args.breadth)
    result = asyncio.run(search.run())
    if result.answer is None:
        print(f"Search stopped without a complete answer. Best partial state: {result.steps}")
        return 1
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reflexion-inspired retries against a deterministic number-list checker.

Run offline with ``python -m examples.reflexion``. Reflections are retained
as prompt context in this instance; no model weights or cross-process memory change.
The checker always requires three distinct positive even integers summing to 18.
"""

import asyncio
import json
import sys

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from examples._cli import ScriptedBackend, backend_from_args, parser, positive_int
from slick import Prompt, parse

CRITERIA = [
    "Return exactly three integers.",
    "All numbers must be distinct.",
    "All numbers must be positive and even.",
    "The sum must be 18.",
]


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    numbers: list[StrictInt]


class Feedback(BaseModel):
    passed: bool
    errors: list[str]


class Reflection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lesson: str = Field(min_length=1)


class ReflectiveSolver:
    """Own attempts, checker feedback, and accumulated lessons for one toy task."""

    def __init__(self, backend, task: str = "Find a valid list.", *, max_attempts: int = 3):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.backend = backend
        self.task = task
        self.max_attempts = max_attempts
        self.attempt_count = 0
        self.lessons: list[str] = []
        self.history: list[dict] = []
        self.result: Candidate | None = None
        self.attempt_prompt = Prompt("reflexion/attempt.j2")
        self.reflection_prompt = Prompt("reflexion/reflect.j2")

    async def attempt(self) -> Candidate:
        if self.attempt_count >= self.max_attempts:
            raise RuntimeError(f"Attempt budget exhausted ({self.max_attempts})")
        self.attempt_count += 1
        text = self.attempt_prompt(
            task=self.task,
            criteria=CRITERIA,
            history=self.history,
            state={"facts": CRITERIA, "decisions": self.lessons, "questions": []},
            schema=Candidate.model_json_schema(),
        )
        return parse(await self.backend.acall(text), Candidate)

    def evaluate(self, candidate: Candidate) -> Feedback:
        """Compute feedback in Python; the model cannot declare its own success."""
        numbers = candidate.numbers
        checks = [
            len(numbers) == 3,
            len(set(numbers)) == len(numbers),
            all(number > 0 and number % 2 == 0 for number in numbers),
            sum(numbers) == 18,
        ]
        errors = [
            criterion for criterion, passed in zip(CRITERIA, checks, strict=True) if not passed
        ]
        feedback = Feedback(passed=not errors, errors=errors)
        self.history.append(
            {"candidate": candidate.model_dump(), "feedback": feedback.model_dump()}
        )
        return feedback

    async def reflect(self, candidate: Candidate, feedback: Feedback) -> Reflection:
        text = self.reflection_prompt(
            task=self.task,
            criteria=CRITERIA,
            candidate=candidate.model_dump(),
            feedback=feedback.model_dump(),
            lessons=self.lessons,
            schema=Reflection.model_json_schema(),
        )
        reflection = parse(await self.backend.acall(text), Reflection)
        self.lessons.append(reflection.lesson)
        return reflection

    async def run(self) -> Candidate:
        while self.result is None:
            candidate = await self.attempt()
            feedback = self.evaluate(candidate)
            if feedback.passed:
                self.result = candidate
            elif self.attempt_count < self.max_attempts:
                await self.reflect(candidate, feedback)
            else:
                raise RuntimeError(
                    f"Attempt budget exhausted ({self.max_attempts}): {feedback.errors}"
                )
        return self.result


def main(argv=None) -> int:
    argument_parser = parser(__doc__)
    argument_parser.add_argument(
        "--task",
        default="Find a valid list.",
        help="guidance for the number-list task; the four checker criteria stay fixed",
    )
    argument_parser.add_argument("--attempts", type=positive_int, default=3)
    args = argument_parser.parse_args(argv)
    demo = ScriptedBackend(
        [
            '{"numbers":[2,2,14]}',
            '{"lesson":"Check distinctness; replace the repeated 2 and rebalance the sum."}',
            '{"numbers":[2,4,12]}',
        ]
    )
    backend = backend_from_args(args, argument_parser, demo)
    solver = ReflectiveSolver(backend, args.task, max_attempts=args.attempts)
    try:
        result = asyncio.run(solver.run())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    print(f"Attempts: {solver.attempt_count}; checked history and retained lessons:")
    print(json.dumps({"history": solver.history, "lessons": solver.lessons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

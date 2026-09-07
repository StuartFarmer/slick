"""Decompose a problem, solve its subproblems in order, then answer the original."""

import asyncio
import json

from pydantic import BaseModel, Field, constr

from examples._cli import ScriptedBackend, backend_from_args, parser, positive_int
from slick import Prompt, parse

NonemptyText = constr(strip_whitespace=True, min_length=1)


class Decomposition(BaseModel):
    subproblems: list[NonemptyText] = Field(min_length=1)


class Answer(BaseModel):
    answer: NonemptyText


class LeastToMost:
    def __init__(self, problem: str, backend, max_subproblems: int = 4):
        if type(max_subproblems) is not int or max_subproblems < 1:
            raise ValueError("max_subproblems must be a positive integer")
        self.problem = problem
        self.backend = backend
        self.max_subproblems = max_subproblems
        self.subproblems: list[str] = []
        self.solutions: list[dict[str, str]] = []
        self.answer: str | None = None
        self.decompose_prompt = Prompt("least_to_most/decompose.j2")
        self.solve_prompt = Prompt("least_to_most/solve.j2")

    async def decompose(self) -> list[str]:
        text = self.decompose_prompt(
            problem=self.problem,
            limit=self.max_subproblems,
            schema=Decomposition.model_json_schema(),
        )
        plan = parse(await self.backend.acall(text), Decomposition)
        if len(plan.subproblems) > self.max_subproblems:
            raise ValueError("decomposition exceeds max_subproblems")
        self.subproblems = plan.subproblems
        self.solutions = []
        self.answer = None
        return self.subproblems

    async def solve_next(self) -> str:
        if not self.subproblems:
            raise ValueError("decompose the problem before solving")
        if self.answer is not None:
            raise ValueError("the original problem is already solved")
        index = len(self.solutions)
        question = self.subproblems[index] if index < len(self.subproblems) else self.problem
        text = self.solve_prompt(
            problem=self.problem,
            question=question,
            solutions=self.solutions,
            schema=Answer.model_json_schema(),
        )
        result = parse(await self.backend.acall(text), Answer)
        self.solutions.append({"input": question, "output": result.answer})
        if index == len(self.subproblems):
            self.answer = result.answer
        return result.answer

    async def run(self) -> str:
        await self.decompose()
        for _ in range(len(self.subproblems) + 1):
            answer = await self.solve_next()
        return answer


def main(argv=None):
    argument_parser = parser(__doc__)
    argument_parser.add_argument(
        "--problem", default="Three boxes hold 4 apples each. Eat 2. How many remain?"
    )
    argument_parser.add_argument("--max-subproblems", type=positive_int, default=4)
    args = argument_parser.parse_args(argv)
    backend = backend_from_args(
        args,
        argument_parser,
        ScriptedBackend(
            [
                json.dumps({"subproblems": ["How many apples are in three boxes of 4?"]}),
                json.dumps({"answer": "12 apples"}),
                json.dumps({"answer": "10 apples remain."}),
            ]
        ),
    )
    solver = LeastToMost(args.problem, backend, args.max_subproblems)
    print(asyncio.run(solver.run()))


if __name__ == "__main__":
    main()

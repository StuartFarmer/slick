"""Decompose a problem, solve its subproblems in order, then answer the original."""

import asyncio
import json

from pydantic import BaseModel, Field, constr

from examples._cli import ScriptedProvider, parser, positive_int, provider_from_args
from slick import prompt

NonemptyText = constr(strip_whitespace=True, min_length=1)


class Decomposition(BaseModel):
    subproblems: list[NonemptyText] = Field(min_length=1)


class Answer(BaseModel):
    answer: NonemptyText


class LeastToMost:
    def __init__(self, problem: str, provider, max_subproblems: int = 4):
        if type(max_subproblems) is not int or max_subproblems < 1:
            raise ValueError("max_subproblems must be a positive integer")
        self.problem = problem
        self.provider = provider
        self.max_subproblems = max_subproblems
        self.subproblems: list[str] = []
        self.solutions: list[dict[str, str]] = []
        self.answer: str | None = None

    @prompt(template="least_to_most/decompose.j2", output_type=Decomposition)
    async def decompose(self, *, generated: Decomposition) -> list[str]:
        if len(generated.subproblems) > self.max_subproblems:
            raise ValueError("decomposition exceeds max_subproblems")
        self.subproblems = generated.subproblems
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
        result = await self.solve(question, provider=self.provider)
        self.solutions.append({"input": question, "output": result.answer})
        if index == len(self.subproblems):
            self.answer = result.answer
        return result.answer

    @prompt(template="least_to_most/solve.j2", output_type=Answer)
    async def solve(self, question: str) -> Answer:
        """Answer one question using the earlier solutions."""

    async def run(self) -> str:
        await self.decompose(provider=self.provider)
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
    provider = provider_from_args(
        args,
        argument_parser,
        ScriptedProvider(
            [
                json.dumps({"subproblems": ["How many apples are in three boxes of 4?"]}),
                json.dumps({"answer": "12 apples"}),
                json.dumps({"answer": "10 apples remain."}),
            ]
        ),
    )
    solver = LeastToMost(args.problem, provider, args.max_subproblems)
    print(asyncio.run(solver.run()))


if __name__ == "__main__":
    main()

"""Worked-solution prompting (model-dependent): python -m examples.chain_of_thought."""

import asyncio

from pydantic import BaseModel

from slick import Prompt, parse

from ._cli import ScriptedBackend, backend_from_args, parser


class Solution(BaseModel):
    explanation: str
    answer: str


class ReasoningSolver:
    def __init__(self, backend):
        self.backend = backend
        self.prompt = Prompt("chain_of_thought/solve.j2")
        self.solution = None
        self.demonstrations = [
            {
                "input": "Three bags contain four apples each. How many apples?",
                "output": '{"explanation":"3 times 4 is 12.","answer":"12"}',
            },
        ]

    async def solve(self, question):
        text = self.prompt(
            question=question,
            demonstrations=self.demonstrations,
            schema=Solution.model_json_schema(),
        )
        solution = parse(await self.backend.acall(text), Solution)
        self.solution = solution
        return solution


def main(argv=None):
    p = parser(__doc__)
    p.add_argument(
        "--question", default="Five bags have six apples each. Four are eaten. How many remain?"
    )
    args = p.parse_args(argv)
    backend = backend_from_args(
        args,
        p,
        ScriptedBackend(['{"explanation":"5 times 6 is 30; subtract 4 to get 26.","answer":"26"}']),
    )
    print(asyncio.run(ReasoningSolver(backend).solve(args.question)).model_dump_json(indent=2))


if __name__ == "__main__":
    main()

"""Worked-solution prompting (model-dependent): python -m examples.chain_of_thought."""

import asyncio

from pydantic import BaseModel

from slick import prompt

from ._cli import ScriptedProvider, parser, provider_from_args


class Solution(BaseModel):
    explanation: str
    answer: str


class ReasoningSolver:
    def __init__(self):
        self.solution = None
        self.demonstrations = [
            {
                "input": "Three bags contain four apples each. How many apples?",
                "output": '{"explanation":"3 times 4 is 12.","answer":"12"}',
            },
        ]

    @prompt(template="chain_of_thought/solve.j2", output_type=Solution)
    async def solve(self, question: str, *, generated: Solution) -> Solution:
        self.solution = generated
        return generated


def main(argv=None):
    p = parser(__doc__)
    p.add_argument(
        "--question", default="Five bags have six apples each. Four are eaten. How many remain?"
    )
    args = p.parse_args(argv)
    provider = provider_from_args(
        args,
        p,
        ScriptedProvider(
            ['{"explanation":"5 times 6 is 30; subtract 4 to get 26.","answer":"26"}']
        ),
    )
    solver = ReasoningSolver()
    print(asyncio.run(solver.solve(args.question, provider=provider)).model_dump_json(indent=2))


if __name__ == "__main__":
    main()

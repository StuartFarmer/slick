"""Sample solutions and vote on normalized final answers.

Repeated real calls do not guarantee diversity: Slick's provider interface does
not expose sampling temperature. The offline script supplies varied answers.
"""

import asyncio
import json
from collections import Counter
from itertools import cycle, islice

from pydantic import BaseModel, ConfigDict, Field

from examples._cli import ScriptedProvider, parser, positive_int, provider_from_args
from slick import Prompt, parse


class Solution(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    answer: str = Field(min_length=1)
    approach: str = Field(min_length=1, description="A brief description of the method used")


class SolutionSampler:
    def __init__(self, problem: str, provider):
        self.problem = problem
        self.provider = provider
        self.samples: list[Solution] = []
        self.votes: Counter[str] = Counter()
        self.sample_prompt = Prompt("self_consistency/sample.j2")

    async def sample(self, count: int = 3) -> list[Solution]:
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        for _ in range(count):
            text = self.sample_prompt(problem=self.problem, schema=Solution.model_json_schema())
            response, _ = await self.provider.acall(text)
            solution = parse(response, Solution)
            self.samples.append(solution)
        return self.samples

    def choose(self) -> Solution:
        """Casefold and collapse whitespace; ties go to the first sampled answer."""
        if not self.samples:
            raise ValueError("sample at least one solution before choosing")
        normalized = [" ".join(solution.answer.casefold().split()) for solution in self.samples]
        self.votes = Counter(normalized)
        winner = self.votes.most_common(1)[0][0]
        return self.samples[normalized.index(winner)]

    async def run(self, count: int = 3) -> Solution:
        await self.sample(count)
        return self.choose()


def main(argv=None):
    argument_parser = parser(__doc__)
    argument_parser.add_argument("--problem", default="What is 17 + 25?")
    argument_parser.add_argument("--samples", type=positive_int, default=3)
    args = argument_parser.parse_args(argv)
    responses = cycle(
        [
            json.dumps({"answer": "42", "approach": "Add tens and units."}),
            json.dumps({"answer": "41", "approach": "An intentionally incorrect demo sample."}),
            json.dumps({"answer": "42", "approach": "Add 20, then 5."}),
        ]
    )
    provider = provider_from_args(
        args, argument_parser, ScriptedProvider(islice(responses, args.samples))
    )
    sampler = SolutionSampler(args.problem, provider)
    result = asyncio.run(sampler.run(args.samples))
    print(result.answer)
    print(f"Votes: {dict(sampler.votes)}")


if __name__ == "__main__":
    main()

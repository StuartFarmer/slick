"""Draft, critique, and revise with ordinary methods: python -m examples.self_refine."""

import asyncio

from slick import prompt

from ._cli import ScriptedProvider, parser, positive_int, provider_from_args


class SelfRefiner:
    """One task with explicit answer, feedback, and revision history; use sequentially."""

    def __init__(self, task, provider, criteria):
        self.task = task
        self.provider = provider
        self.criteria = list(criteria)
        self.answer = None
        self.feedback = None
        self.revisions = []
        self.feedback_history = []

    @prompt(template="self_refine/draft.j2")
    async def draft(self, *, generated: str) -> str:
        self.answer, self.feedback = generated, None
        self.revisions, self.feedback_history = [generated], []
        return generated

    async def critique(self):
        if self.answer is None:
            raise ValueError("Create a draft first.")
        return await self.generate_feedback(provider=self.provider)

    @prompt(template="self_refine/critique.j2")
    async def generate_feedback(self, *, generated: str) -> str:
        self.feedback = generated
        return generated

    async def revise(self):
        if self.feedback is None:
            raise ValueError("Critique the current answer first.")
        return await self.generate_revision(provider=self.provider)

    @prompt(template="self_refine/revise.j2")
    async def generate_revision(self, *, generated: str) -> str:
        self.revisions.append(generated)
        self.feedback_history.append(self.feedback)
        self.answer, self.feedback = generated, None
        return generated

    async def run(self, rounds=2):
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        await self.draft(provider=self.provider)
        for _ in range(rounds):
            await self.critique()
            await self.revise()
        return self.answer


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--task", default="Explain how Slick separates prompts from execution.")
    p.add_argument("--rounds", type=positive_int, default=2)
    p.add_argument("--criterion", action="append", help="repeat to supply evaluation criteria")
    args = p.parse_args(argv)
    responses = [f"Demo draft for: {args.task}"]
    for turn in range(args.rounds):
        responses.extend(
            [
                "Explain which part renders text and which part executes it.",
                f"Demo revision {turn + 1}: @prompt renders Jinja, generates a response, "
                "and runs the method body. Ordinary Python owns state and sequencing.",
            ]
        )
    provider = provider_from_args(args, p, ScriptedProvider(responses))
    writer = SelfRefiner(args.task, provider, args.criterion or ["Accuracy", "Clarity"])
    answer = asyncio.run(writer.run(args.rounds))
    print(answer)
    print(f"Revisions: {len(writer.revisions) - 1}")


if __name__ == "__main__":
    main()

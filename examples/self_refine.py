"""Draft, critique, and revise with ordinary methods: python -m examples.self_refine."""

import asyncio

from slick import Prompt

from ._cli import ScriptedBackend, backend_from_args, parser, positive_int


class SelfRefiner:
    """One task with explicit answer, feedback, and revision history; use sequentially."""

    def __init__(self, task, backend, criteria):
        self.task = task
        self.backend = backend
        self.criteria = list(criteria)
        self.answer = None
        self.feedback = None
        self.revisions = []
        self.feedback_history = []
        self.draft_prompt = Prompt("self_refine/draft.j2")
        self.critique_prompt = Prompt("self_refine/critique.j2")
        self.revise_prompt = Prompt("self_refine/revise.j2")

    async def draft(self):
        text = self.draft_prompt(task=self.task)
        answer = await self.backend.acall(text)
        self.answer, self.feedback = answer, None
        self.revisions, self.feedback_history = [answer], []
        return answer

    async def critique(self):
        if self.answer is None:
            raise ValueError("Create a draft first.")
        text = self.critique_prompt(
            task=self.task,
            answer=self.answer,
            criteria=self.criteria,
            revisions=self.revisions,
            feedback_history=self.feedback_history,
        )
        self.feedback = await self.backend.acall(text)
        return self.feedback

    async def revise(self):
        if self.feedback is None:
            raise ValueError("Critique the current answer first.")
        text = self.revise_prompt(
            task=self.task,
            answer=self.answer,
            feedback=self.feedback,
            revisions=self.revisions,
            feedback_history=self.feedback_history,
        )
        answer = await self.backend.acall(text)
        self.revisions.append(answer)
        self.feedback_history.append(self.feedback)
        self.answer, self.feedback = answer, None
        return answer

    async def run(self, rounds=2):
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        await self.draft()
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
                f"Demo revision {turn + 1}: Prompt renders Jinja arguments into text; "
                "the backend executes that text. Ordinary Python owns state and sequencing.",
            ]
        )
    backend = backend_from_args(args, p, ScriptedBackend(responses))
    writer = SelfRefiner(args.task, backend, args.criterion or ["Accuracy", "Clarity"])
    answer = asyncio.run(writer.run(args.rounds))
    print(answer)
    print(f"Revisions: {len(writer.revisions) - 1}")


if __name__ == "__main__":
    main()

"""Ordinary application state and methods. Run offline: python -m examples.question_answerer."""

import asyncio

from slick import prompt

from ._cli import ScriptedProvider, parser, provider_from_args


class QuestionAnswerer:
    """Own history; supply a provider or session to each decorated call."""

    def __init__(self):
        self.history: list[dict[str, str]] = []

    @prompt(template="question_answerer/answer.j2")
    async def ask(self, question: str, *, generated: str) -> str:
        self.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": generated},
            ]
        )
        return generated

    @prompt(template="critique.j2")
    async def critique(self, answer: str) -> str:
        """Critique without changing the conversation history."""


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--question", action="append", help="repeat to supply conversation turns")
    p.add_argument("--critique", action="store_true", help="critique the final answer")
    args = p.parse_args(argv)
    questions = args.question or ["My name is Ada.", "What is my name?"]
    responses = [f"Demo answer to: {question}" for question in questions]
    responses.append("Demo critique: check the answer against the conversation.")
    provider = provider_from_args(args, p, ScriptedProvider(responses))
    qa = QuestionAnswerer()

    async def run():
        for question in questions:
            answer = await qa.ask(question, provider=provider)
            print(answer)
        if args.critique:
            print(await qa.critique(answer, provider=provider))

    asyncio.run(run())


if __name__ == "__main__":
    main()

"""Ordinary application state and methods. Run offline: python -m examples.question_answerer."""

import asyncio

from slick import Prompt

from ._cli import ScriptedProvider, parser, provider_from_args


class QuestionAnswerer:
    """Own a provider and history; call methods sequentially on each instance."""

    def __init__(self, provider):
        self.provider = provider
        self.history: list[dict[str, str]] = []
        self.answer_prompt = Prompt("conversation.j2")
        self.critique_prompt = Prompt("critique.j2")

    async def ask(self, question: str) -> str:
        text = self.answer_prompt(question=question, messages=self.history)
        answer, _ = await self.provider.acall(text)
        self.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer

    async def critique(self, answer: str) -> str:
        text = self.critique_prompt(answer=answer)
        response, _ = await self.provider.acall(text)
        return response


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--question", action="append", help="repeat to supply conversation turns")
    p.add_argument("--critique", action="store_true", help="critique the final answer")
    args = p.parse_args(argv)
    questions = args.question or ["My name is Ada.", "What is my name?"]
    responses = [f"Demo answer to: {question}" for question in questions]
    responses.append("Demo critique: check the answer against the conversation.")
    provider = provider_from_args(args, p, ScriptedProvider(responses))
    qa = QuestionAnswerer(provider)

    async def run():
        for question in questions:
            answer = await qa.ask(question)
            print(answer)
        if args.critique:
            print(await qa.critique(answer))

    asyncio.run(run())


if __name__ == "__main__":
    main()

"""Ordinary application state and methods. Run offline: python -m examples.question_answerer."""

import asyncio
from pathlib import Path

from slick import Prompt, prompts


class QuestionAnswerer:
    """Own a backend and history; call methods sequentially on each instance."""

    def __init__(self, backend):
        self.backend = backend
        self.history: list[dict[str, str]] = []
        self.answer_prompt = Prompt("conversation.j2")
        self.critique_prompt = Prompt("critique.j2")

    async def ask(self, question: str) -> str:
        text = self.answer_prompt(question=question, messages=self.history)
        answer = await self.backend.acall(text)
        self.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer

    async def critique(self, answer: str) -> str:
        text = self.critique_prompt(answer=answer)
        return await self.backend.acall(text)


class DemoBackend:
    """Canned responses so the example needs no credentials or network."""

    def __init__(self):
        self.responses = iter(
            [
                "Hello, Ada.",
                "Your name is Ada.",
                "The answer is clear; verify it against the original conversation.",
            ]
        )

    async def acall(self, text: str) -> str:
        return next(self.responses)


async def main():
    qa = QuestionAnswerer(DemoBackend())
    print(await qa.ask("My name is Ada."))
    answer = await qa.ask("What is my name?")
    print(answer)
    print(await qa.critique(answer))


if __name__ == "__main__":
    prompts.TEMPLATE_ROOT = Path(__file__).with_name("prompts")
    asyncio.run(main())

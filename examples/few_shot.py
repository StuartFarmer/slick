"""Classify with input/output demonstrations: python -m examples.few_shot."""

import asyncio
from typing import Literal

from pydantic import TypeAdapter

from slick import Prompt, parse

from ._cli import ScriptedBackend, backend_from_args, parser

Sentiment = Literal["positive", "neutral", "negative"]


class FewShotAnswerer:
    def __init__(self, backend, demonstrations):
        self.backend = backend
        self.demonstrations = demonstrations
        self.prompt = Prompt("few_shot/classify.j2")
        self.answer = None

    async def ask(self, question):
        text = self.prompt(
            question=question,
            demonstrations=self.demonstrations,
            schema=TypeAdapter(Sentiment).json_schema(),
        )
        answer = parse(await self.backend.acall(text), Sentiment)
        self.answer = answer
        return answer


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--question", default="This was a wonderful experience!")
    args = p.parse_args(argv)
    backend = backend_from_args(args, p, ScriptedBackend(['"positive"']))
    qa = FewShotAnswerer(
        backend,
        [
            {"input": "I loved it.", "output": '"positive"'},
            {"input": "It arrived on Tuesday.", "output": '"neutral"'},
            {"input": "It broke immediately.", "output": '"negative"'},
        ],
    )
    print(asyncio.run(qa.ask(args.question)))


if __name__ == "__main__":
    main()

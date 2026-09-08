"""Classify with input/output demonstrations: python -m examples.few_shot."""

import asyncio
from typing import Literal

from pydantic import TypeAdapter

from slick import Prompt, parse

from ._cli import ScriptedProvider, parser, provider_from_args

Sentiment = Literal["positive", "neutral", "negative"]


class FewShotAnswerer:
    def __init__(self, provider, demonstrations):
        self.provider = provider
        self.demonstrations = demonstrations
        self.prompt = Prompt("few_shot/classify.j2")
        self.answer = None

    async def ask(self, question):
        text = self.prompt(
            question=question,
            demonstrations=self.demonstrations,
            schema=TypeAdapter(Sentiment).json_schema(),
        )
        answer = parse(await self.provider.acall(text), Sentiment)
        self.answer = answer
        return answer


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--question", default="This was a wonderful experience!")
    args = p.parse_args(argv)
    provider = provider_from_args(args, p, ScriptedProvider(['"positive"']))
    qa = FewShotAnswerer(
        provider,
        [
            {"input": "I loved it.", "output": '"positive"'},
            {"input": "It arrived on Tuesday.", "output": '"neutral"'},
            {"input": "It broke immediately.", "output": '"negative"'},
        ],
    )
    print(asyncio.run(qa.ask(args.question)))


if __name__ == "__main__":
    main()

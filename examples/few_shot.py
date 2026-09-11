"""Classify with input/output demonstrations: python -m examples.few_shot."""

import asyncio
from typing import Literal

from slick import prompt

from ._cli import ScriptedProvider, parser, provider_from_args

Sentiment = Literal["positive", "neutral", "negative"]


class FewShotAnswerer:
    def __init__(self, demonstrations):
        self.demonstrations = demonstrations
        self.answer = None

    @prompt(template="few_shot/classify.j2", output_type=Sentiment)
    async def ask(self, question: str, *, generated: Sentiment) -> Sentiment:
        self.answer = generated
        return generated


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--question", default="This was a wonderful experience!")
    args = p.parse_args(argv)
    provider = provider_from_args(args, p, ScriptedProvider(['"positive"']))
    qa = FewShotAnswerer(
        [
            {"input": "I loved it.", "output": '"positive"'},
            {"input": "It arrived on Tuesday.", "output": '"neutral"'},
            {"input": "It broke immediately.", "output": '"negative"'},
        ],
    )
    print(asyncio.run(qa.ask(args.question, provider=provider)))


if __name__ == "__main__":
    main()

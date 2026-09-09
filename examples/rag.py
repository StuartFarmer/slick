"""Application retrieval and cited answers: python -m examples.rag."""

import asyncio
import json
import re

from pydantic import BaseModel

from slick import Prompt, parse

from ._cli import ScriptedProvider, parser, provider_from_args


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[str]


class GroundedAnswerer:
    def __init__(self, provider, documents):
        self.provider = provider
        self.documents = documents
        self.prompt = Prompt("rag/answer.j2")
        self.evidence = []
        self.answer = None

    def retrieve(self, question):
        """Local lexical retrieval; replace this method with your chosen store."""
        terms = set(re.findall(r"\w+", question.lower())) - {"a", "the", "is", "how", "can", "i"}
        ranked = sorted(
            self.documents,
            key=lambda d: len(terms & set(re.findall(r"\w+", d["text"].lower()))),
            reverse=True,
        )
        return [d for d in ranked if terms & set(re.findall(r"\w+", d["text"].lower()))][:3]

    async def ask(self, question):
        documents = self.retrieve(question)
        text = self.prompt(
            question=question, documents=documents, schema=GroundedAnswer.model_json_schema()
        )
        response, _ = await self.provider.acall(text)
        answer = parse(response, GroundedAnswer)
        if not set(answer.citations) <= {document["id"] for document in documents}:
            raise ValueError("Answer contains a citation outside the retrieved evidence")
        self.evidence, self.answer = documents, answer
        return answer


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--question", default="How can I supply history?")
    args = p.parse_args(argv)
    documents = [
        {
            "id": "history",
            "source": "Application guide",
            "text": "History can be supplied as a list of messages.",
        },
        {
            "id": "templates",
            "source": "Template guide",
            "text": "Jinja templates can include and import other templates.",
        },
    ]
    # The canned answer uses identifiers from the actual retrieval for this input.
    demo_documents = GroundedAnswerer(None, documents).retrieve(args.question)
    demo_answer = json.dumps(
        {
            "answer": "Demo answer based on the retrieved documents.",
            "citations": [d["id"] for d in demo_documents],
        }
    )
    provider = provider_from_args(args, p, ScriptedProvider([demo_answer]))
    answer = asyncio.run(GroundedAnswerer(provider, documents).ask(args.question))
    print(answer.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

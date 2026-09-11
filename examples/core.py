"""Run offline: python -m examples.core. Pass an API/CLI provider to run() for real calls."""

import asyncio
from pathlib import Path

from pydantic import BaseModel

from slick import prompt, prompts


class Summary(BaseModel):
    headline: str
    points: list[str]


async def search_documents(query: str) -> list[str]:
    """Application retrieval; replace this local lookup with your chosen store."""
    documents = [
        "History can be supplied as a list of messages.",
        "Templates can include other templates.",
    ]
    terms = query.lower().split()
    return [document for document in documents if any(t in document.lower() for t in terms)]


async def run(provider) -> dict:
    @prompt(template="summary.j2", output_type=Summary)
    async def summarize(document: str) -> Summary:
        """Summarize a document into a headline and points."""

    @prompt(template="conversation.j2")
    async def reply(question: str, messages: list[dict]) -> str:
        """Answer using the supplied history."""

    @prompt(template="answer.j2")
    async def answer(question: str, documents: list[str]) -> str:
        """Answer using retrieved documents."""

    summary = await summarize(
        "Slick renders Jinja, calls a provider, and parses the result.", provider=provider
    )
    history = [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Hello, Ada."},
    ]
    conversation = await reply("Do you remember my name?", history, provider=provider)

    question = "How can I supply history?"
    documents = await search_documents(question)
    retrieval = await answer(question, documents, provider=provider)
    return {"summary": summary, "conversation": conversation, "retrieval": retrieval}


class DemoProvider:
    """Fixed responses to demonstrate the complete pipeline without credentials."""

    def __init__(self):
        self.responses = iter(
            [
                '{"headline":"Jinja to LLM", "points":["Explicit context"]}',
                "Your name is Ada.",
                "Supply history as a list of messages.",
            ]
        )

    async def acall(self, text: str):
        return next(self.responses), []


if __name__ == "__main__":
    prompts.TEMPLATE_ROOT = Path(__file__).with_name("prompts")
    for name, result in asyncio.run(run(DemoProvider())).items():
        print(f"{name}: {result}")

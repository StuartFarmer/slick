"""Automatic tool bookkeeping. Run offline: python -m examples.session."""

import asyncio

from slick import Prompt, Session

from ._cli import parser, provider_from_args


class Documents:
    def __init__(self):
        self.executions = 0
        self.documents = {"pricing.md": "Pricing: the team plan costs $20 per seat."}

    async def search(self, query: str) -> str:
        """Find documents containing the query and return their names and text."""
        self.executions += 1
        return (
            "\n".join(
                f"{name}: {text}"
                for name, text in self.documents.items()
                if query.casefold() in f"{name} {text}".casefold()
            )
            or "No documents found."
        )


class DemoProvider:
    """Request a real tool, then answer using its actual result."""

    async def acall(self, context, *, tools=None, tool_results=None):
        if not tool_results:
            return "I'll search the documents.", [
                {"id": "search-1", "name": "search", "arguments": {"query": "pricing"}}
            ]
        return "Search result: " + tool_results[0]["content"], []


async def run(provider, documents, task):
    session = Session(provider=provider, tools=[documents.search])
    prompt = Prompt("session.j2")
    for _ in range(4):
        context = prompt(task=task, observations=[item["text"] for item in session.history])
        text, requests = await session.acall(context)
        if not requests:
            return text
        await session.resolve_pending()
    raise RuntimeError("Example reached its four-call limit")


def main(argv=None):
    argument_parser = parser(__doc__)
    argument_parser.add_argument("--task", default="Find the pricing document.")
    args = argument_parser.parse_args(argv)
    if args.provider in {"codex", "claude"}:
        argument_parser.error("This example requires Python tools; use demo, openai, or anthropic")
    provider = provider_from_args(args, argument_parser, DemoProvider())
    documents = Documents()
    print(asyncio.run(run(provider, documents, args.task)))
    print(f"Search executions: {documents.executions}")


if __name__ == "__main__":
    main()

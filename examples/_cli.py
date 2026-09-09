"""Command-line plumbing shared only by the examples."""

import argparse
from pathlib import Path

from slick import prompts, providers


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument(
        "--provider",
        choices=["demo", "openai", "anthropic", "codex", "claude"],
        default="demo",
        help="demo uses canned responses; other choices make real calls",
    )
    result.add_argument("--model", help="required model ID for OpenAI/Anthropic")
    result.add_argument(
        "--timeout", type=positive_int, default=60, help="seconds per provider call"
    )
    return result


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def provider_from_args(args, argument_parser, demo):
    prompts.TEMPLATE_ROOT = Path(__file__).with_name("prompts")
    if args.provider == "demo":
        if args.model:
            argument_parser.error("--model requires a real provider")
        return demo
    if args.provider in {"openai", "anthropic"}:
        if not args.model:
            argument_parser.error("--model is required for API providers")
        provider = providers.OpenAIAPI if args.provider == "openai" else providers.AnthropicAPI
        return provider(model=args.model, timeout=args.timeout)
    provider = providers.CodexCLI if args.provider == "codex" else providers.ClaudeCLI
    return provider(model=args.model, timeout=args.timeout)


class ScriptedProvider:
    """Offline responses; loops still render, parse, execute tools, and update state."""

    def __init__(self, responses):
        self.responses = iter(responses)

    async def acall(self, text: str, *, tools=None, tool_results=None):
        try:
            return next(self.responses), []
        except StopIteration as exc:
            raise RuntimeError("Demo response script exhausted") from exc

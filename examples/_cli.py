"""Command-line plumbing shared only by the examples."""

import argparse
from pathlib import Path

from slick import get_model, prompts
from slick.backends import Anthropic, OpenAI


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument(
        "--backend",
        choices=["demo", "openai", "anthropic", "codex", "claude"],
        default="demo",
        help="demo uses canned responses; other choices make real calls",
    )
    result.add_argument("--model", help="required model ID for OpenAI/Anthropic")
    result.add_argument("--timeout", type=positive_int, default=60, help="seconds per backend call")
    return result


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def backend_from_args(args, argument_parser, demo):
    prompts.TEMPLATE_ROOT = Path(__file__).with_name("prompts")
    if args.backend == "demo":
        if args.model:
            argument_parser.error("--model requires a real backend")
        return demo
    if args.backend in {"openai", "anthropic"}:
        if not args.model:
            argument_parser.error("--model is required for API backends")
        provider = OpenAI if args.backend == "openai" else Anthropic
        return provider(model=args.model, timeout=args.timeout)
    backend = get_model(args.backend, args.model)
    backend.timeout = args.timeout
    return backend


class ScriptedBackend:
    """Offline responses; loops still render, parse, execute tools, and update state."""

    def __init__(self, responses):
        self.responses = iter(responses)

    async def acall(self, text: str) -> str:
        try:
            return next(self.responses)
        except StopIteration as exc:
            raise RuntimeError("Demo response script exhausted") from exc

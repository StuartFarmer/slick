"""Slick CLI: send one prompt to an explicitly selected provider.

slick call "summarize this" --provider codex  # prompt as an argument
cat document.md | slick call --provider codex  # or on stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, providers
from .providers import ProviderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slick", description="Slick CLI")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    call = sub.add_parser("call", help="Send one prompt to a model")
    call.add_argument("prompt", nargs="?", help="Prompt text; read from stdin if omitted")
    call.add_argument(
        "--provider",
        choices=["anthropic", "claude", "codex", "litellm", "openai", "openrouter"],
        required=True,
        help="Provider to use",
    )
    call.add_argument("--model", help="Model id to pass to the provider")
    call.add_argument("--api-base", help="API endpoint override (LiteLLM only)")
    call.add_argument("--output", type=Path, help="Write the response here as well")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        print("No prompt given.", file=sys.stderr)
        return 2

    try:
        if args.api_base is not None and args.provider != "litellm":
            raise ValueError("--api-base requires --provider litellm.")
        api_providers = {
            "litellm": "LiteLLMAPI",
            "openai": "OpenAIAPI",
            "anthropic": "AnthropicAPI",
            "openrouter": "OpenRouterAPI",
        }
        if args.provider in api_providers:
            if not args.model:
                raise ValueError("API providers require --model.")
            kwargs = {"model": args.model}
            if args.api_base is not None:
                kwargs["api_base"] = args.api_base
            provider = getattr(providers, api_providers[args.provider])(**kwargs)
        else:
            cli_provider = providers.CodexCLI if args.provider == "codex" else providers.ClaudeCLI
            provider = cli_provider(model=args.model)
        text, requests = provider.call(prompt)
        if requests:
            raise ValueError("The CLI expects final text; handle tool requests in Python.")
    except (ProviderError, ImportError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

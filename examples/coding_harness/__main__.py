"""Run with a model provider, or use --dry-run for the offline scripted demo."""

import argparse
import asyncio
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from slick import prompts, providers

from .agent import create_agent
from .config import load_config
from .demo import DemoProvider, create_demo
from .session import load


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--provider", choices=["openai", "anthropic", "litellm"], help="default: openai"
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="run the scripted demo in a temporary repository without API calls",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="project directory; create it and initialize Git if needed",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--resume", type=Path)
    return parser


async def _headless(provider_instance, root, config, task, saved):
    agent = await create_agent(provider_instance, root, config, saved=saved)
    result = await agent.run(task)
    return {"verified": 0, "unverified": 2, "blocked": 2, "failed": 1, "cancelled": 130}[
        result["status"]
    ]


def _validate_args(parser, args):
    if os.name != "posix" or sys.platform not in {"darwin", "linux"}:
        parser.error("Command execution requires macOS or Linux")
    if args.headless and not args.task:
        parser.error("--headless requires --task")
    if args.task and not args.headless:
        parser.error("--task requires --headless; use the input box interactively")
    if args.resume:
        if any([args.dry_run, args.workspace, args.config]):
            parser.error("--resume cannot be combined with --dry-run or workspace/config overrides")
        return
    args.provider = "demo" if args.dry_run else (args.provider or "openai")
    if args.dry_run:
        if args.workspace or args.model or args.config:
            parser.error("--dry-run owns its temporary workspace, model, and check configuration")
    elif not args.model or not args.workspace:
        parser.error("Real providers require --model and --workspace; use --dry-run for the demo")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "prompts"
    try:
        saved = load(args.resume) if args.resume else None
        provider = (args.provider or saved.provider) if saved else args.provider
        if saved and provider != saved.provider and not args.model:
            parser.error("Changing the resumed provider requires --model")
        model = (args.model or saved.model) if saved else args.model
        context = (
            TemporaryDirectory(prefix="slick-coding-demo-")
            if provider == "demo" and not saved
            else nullcontext(None)
        )
        with context as temporary:
            root = Path(saved.root) if saved else Path(temporary or args.workspace).resolve()
            config = (
                saved.config
                if saved
                else (create_demo(root) if provider == "demo" else load_config(args.config))
            )
            if provider == "demo":
                provider_instance = DemoProvider(root)
            else:
                provider_name = {
                    "openai": "OpenAIAPI",
                    "anthropic": "AnthropicAPI",
                    "litellm": "LiteLLMAPI",
                }[provider]
                provider_instance = getattr(providers, provider_name)(model=model)
            if args.headless:
                return asyncio.run(_headless(provider_instance, root, config, args.task, saved))
            try:
                from .tui import HarnessApp
            except ImportError as error:
                raise RuntimeError(
                    "Install the TUI with: python -m pip install -r "
                    "examples/coding_harness/requirements.txt"
                ) from error
            agent = asyncio.run(create_agent(provider_instance, root, config, saved=saved))
            return HarnessApp(agent).run() or 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    except Exception as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

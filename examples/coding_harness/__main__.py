"""Run with a model provider, or use --dry-run for the offline scripted demo."""

import argparse
import asyncio
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from slick import prompts
from slick.providers import AnthropicAPI, LiteLLMAPI, OpenAIAPI

from .agent import CodingAgent
from .demo import DemoProvider, create_demo
from .session import load_session, restore_session
from .state import load_config
from .workspace import Workspace


async def deny(request):
    return "deny"


async def create_agent(provider_instance, root, config, *, decide, emit, saved=None):
    if saved is not None:
        if Path(root).resolve() != Path(saved.root) or config != saved.config:
            raise ValueError("Resume configuration does not match saved session")
        return await restore_session(saved, provider_instance, decide=decide, emit=emit)
    workspace = Workspace(
        Path(root),
        checks=config.checks,
        decide=decide,
        command_timeout=config.limits.command_timeout,
    )
    await workspace.initialize(create=True)
    agent = CodingAgent(provider_instance, workspace, config, emit=emit)
    agent.state.fingerprint = await workspace.fingerprint()
    return agent


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


def print_event(event):
    data = event.data
    if event.kind in {"user", "assistant", "status"}:
        print(f"{event.kind.capitalize()}: {data['text']}")
    elif event.kind == "tool_finished":
        print(f"{'Error' if data['is_error'] else 'Done'}: {data['name']}")
    elif event.kind == "verification":
        for result in data["results"]:
            print(f"Check {result['name']}: exit {result['command']['exit_code']}")
    elif event.kind == "completed":
        result = data["result"]
        print(
            f"{result['status'].capitalize()} · {result['turns']} model turns · "
            f"{result['tool_calls']} tools · {result['repairs']} repairs"
        )
        print(result["answer"])


async def _headless(provider_instance, root, config, task, saved):
    agent = await create_agent(
        provider_instance, root, config, decide=deny, emit=print_event, saved=saved
    )
    result = await agent.run(task)
    return {"verified": 0, "unverified": 2, "blocked": 2, "failed": 1, "cancelled": 130}[
        result.status
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
            parser.error(
                "--resume cannot be combined with --dry-run or "
                "workspace/config overrides"
            )
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
        saved = load_session(args.resume) if args.resume else None
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
            provider_instance = (
                DemoProvider(root)
                if provider == "demo"
                else {
                    "openai": OpenAIAPI,
                    "anthropic": AnthropicAPI,
                    "litellm": LiteLLMAPI,
                }[provider](model=model)
            )
            if args.headless:
                return asyncio.run(_headless(provider_instance, root, config, args.task, saved))
            try:
                from .tui import HarnessApp
            except ImportError as error:
                raise RuntimeError(
                    "Install the TUI with: python -m pip install -r "
                    "examples/coding_harness/requirements.txt"
                ) from error
            return HarnessApp(provider_instance, root, config, saved=saved).run() or 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    except Exception as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

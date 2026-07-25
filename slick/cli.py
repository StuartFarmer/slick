"""Slick CLI: check what a model resolves to, and send it one prompt.

    slick model                        # the backend and model id a bare call uses
    slick call "summarize this"        # prompt as an argument
    cat document.md | slick call       # or on stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .models import BACKENDS, ModelError, get_default, get_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slick", description="Slick CLI")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("model", help="Show the resolved default backend and model")

    call = sub.add_parser("call", help="Send one prompt to a model")
    call.add_argument("prompt", nargs="?", help="Prompt text; read from stdin if omitted")
    call.add_argument("--backend", choices=sorted(BACKENDS), help="Backend to use")
    call.add_argument("--model", help="Model id to pass to the backend")
    call.add_argument("--output", type=Path, help="Write the response here as well")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "model":
        backend, model = get_default()
        print(f"backend={backend} model={model or '(backend default)'}")
        return 0

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        print("No prompt given.", file=sys.stderr)
        return 2

    try:
        text = get_model(args.backend, args.model).call(prompt)
    except (ModelError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

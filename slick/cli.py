from __future__ import annotations

import argparse

from . import __version__
from . import models as _models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slick", description="Slick CLI")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    subparsers = parser.add_subparsers(dest="command")

    # models subcommands (skeleton wiring; logic to be added later)
    models = subparsers.add_parser("models", help="Model management")
    models_sub = models.add_subparsers(dest="models_cmd")

    models_list = models_sub.add_parser("list", help="List models")
    models_list.add_argument("--provider", help="Filter by provider")

    models_sub.add_parser("providers", help="List providers")

    models_set = models_sub.add_parser("set-default", help="Set default model")
    models_set.add_argument("model_or_alias")
    models_set.add_argument("--provider")

    models_sub.add_parser("show-default", help="Show default model")

    models_test = models_sub.add_parser("test", help="Test a model with a simple prompt")
    models_test.add_argument("--model")
    models_test.add_argument("--provider")
    models_test.add_argument("--prompt", default="Hello!")

    return parser


def _handle_models(args: argparse.Namespace) -> None:
    """Dispatch for 'slick models ...' commands (placeholder outputs)."""

    cmd = args.models_cmd
    if cmd == "list":
        provider = args.provider or "openai"
        try:
            names = _models.list_models(provider)
        except Exception as e:
            print(f"Error listing models for {provider}: {e}")
            return
        if not names:
            print(f"No models visible for provider '{provider}' (missing API key or none available).")
            return
        for n in names:
            print(n)
    elif cmd == "providers":
        for p in _models.list_providers():
            print(p)
    elif cmd == "set-default":
        _models.set_default_model(args.model_or_alias, provider=args.provider)
        print("Default updated in-memory (not persisted).")
    elif cmd == "show-default":
        m, p = _models.get_default_model()
        print(f"model={m} provider={p}")
    elif cmd == "test":
        try:
            chat = _models.create_chat_model(model=args.model, provider=args.provider)
        except Exception as e:
            print(f"Could not create chat model: {e}")
            return
        try:
            from langchain_core.messages import HumanMessage

            resp = chat.invoke([HumanMessage(content=args.prompt or "Hello!")])
            text = getattr(resp, "content", str(resp))
            print(text)
        except Exception as e:
            print(f"Invocation failed: {e}")
    else:
        print("Unknown models command; use --help")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "version", False):
        print(__version__)
        return

    if getattr(args, "command", None) == "models":
        _handle_models(args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

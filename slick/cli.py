from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slick", description="Slick CLI")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(__version__)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

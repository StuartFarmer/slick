"""Render all shared Jinja primitives: python -m examples.primitives."""

import argparse
from pathlib import Path

from slick import Prompt, prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="How does Slick work?")
    args = parser.parse_args()
    prompts.TEMPLATE_ROOT = Path(__file__).with_name("prompts")
    print(
        Prompt("primitives.j2")(
            task=args.question,
            audience="engineers",
            constraints=["Use supplied evidence"],
            messages=[{"role": "user", "content": args.question}],
            documents=[{"id": "guide", "source": "Slick guide", "text": "Prompts render text."}],
            demonstrations=[
                {"input": "What is a prompt?", "output": "A template rendered with arguments."}
            ],
            criteria=["Accuracy", "Clarity"],
            state={
                "facts": ["Templates use Jinja"],
                "decisions": ["Keep state in Python"],
                "questions": ["Which provider should execute the text?"],
            },
            records=[{"action": "read guide", "result": "Found the rendering contract"}],
        )
    )


if __name__ == "__main__":
    main()

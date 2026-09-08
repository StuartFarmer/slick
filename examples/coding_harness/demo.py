"""Scripted model turns over a real, disposable broken Python repository."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from slick.turns import ModelTurn, ToolCall, ToolResult, UserMessage, validate_history

from .state import Check, HarnessConfig

BROKEN = "def total(values):\n    return sum(values) + 1\n"
PARTIAL = "def total(values):\n    return sum(values) if values else 1\n"
FIXED = "def total(values):\n    return sum(values)\n"
TESTS = """import unittest
from pricing import total

class TotalTests(unittest.TestCase):
    def test_nonempty(self):
        self.assertEqual(total([2, 3]), 5)

    def test_empty(self):
        self.assertEqual(total([]), 0)
"""


def create_demo(root: Path) -> HarnessConfig:
    root = Path(root)
    if any(root.iterdir()):
        raise ValueError("Demo requires an empty temporary directory")
    (root / "pricing.py").write_text(BROKEN, encoding="utf-8")
    (root / "test_pricing.py").write_text(TESTS, encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Slick demo",
            "-c",
            "user.email=demo@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Demo fixture",
        ],
    ):
        subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=15,
        )
    return HarnessConfig(
        checks=[Check(name="unit", argv=[sys.executable, "-m", "unittest", "-q"])],
        skills=["python"],
    )


class DemoProvider:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.model = "scripted"
        self._number = 0
        self._read = None
        self._await_check = False

    def identity(self):
        return {"provider": "demo", "model": self.model}

    def _turn(self, text, calls=()):
        return ModelTurn(
            "demo", self.model, text, list(calls), [], "tool_calls" if calls else "end_turn"
        )

    def _call(self, name, arguments):
        self._number += 1
        return ToolCall(f"demo_{self._number}", name, arguments)

    async def aturn(self, history, *, tools, instructions=""):
        validate_history(history, provider="demo", model=self.model)
        if not tools:
            return self._turn(
                json.dumps(
                    {
                        "facts": ["The demo uses real unittest checks."],
                        "decisions": [],
                        "open_questions": [],
                        "modified_files": ["pricing.py"],
                        "next_steps": ["Read current pricing.py and run the configured checks."],
                    }
                )
            )
        source = (self.root / "pricing.py").read_text(encoding="utf-8")
        if self._await_check:
            self._await_check = False
            return self._turn("The edit is ready for the configured checks.")
        if source == FIXED:
            return self._turn(
                "Updated total() to handle both ordinary and empty lists. "
                "The configured checks determine verification."
            )
        if source not in {BROKEN, PARTIAL}:
            raise RuntimeError("Demo fixture changed unexpectedly; start a fresh demo")
        previous = next((item for item in reversed(history) if isinstance(item, ToolResult)), None)
        if self._read is None or previous is None or previous.call_id != self._read.id:
            if source == PARTIAL:
                feedback = [item.text for item in history if isinstance(item, UserMessage)]
                if not any(
                    "Verification: failed" in text and "exit_code: 1" in text for text in feedback
                ):
                    raise RuntimeError("Demo expected a real failed verification before repair")
            self._read = self._call("read_file", {"path": "pricing.py"})
            return self._turn("I'll inspect the current implementation.", [self._read])
        if previous.is_error:
            raise RuntimeError("Demo read failed: " + previous.content)
        observation = json.loads(previous.content)
        digest = hashlib.sha256(source.encode()).hexdigest()
        if observation["sha256"] != digest:
            raise RuntimeError("Demo observation is stale")
        replacement = PARTIAL if source == BROKEN else FIXED
        call = self._call(
            "edit_file",
            {"path": "pricing.py", "old": source, "new": replacement, "expected_sha256": digest},
        )
        self._read = None
        self._await_check = True
        return self._turn("I'll apply a focused edit.", [call])

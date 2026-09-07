"""Executable example contracts, including the text rendered by shared parts."""

import asyncio
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from slick import prompts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def template_root(monkeypatch):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", ROOT / "examples" / "prompts")


def test_shared_primitives_render_as_one_standalone_example():
    result = subprocess.run(
        [sys.executable, "-m", "examples.primitives"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    for content in (
        "user: How does Slick work?",
        "[guide]",
        "Input:",
        "Output:",
        "Accuracy",
        "Decisions",
        "Unresolved questions",
        "Observation:",
        "engineers",
    ):
        assert content in result.stdout
    assert "{{" not in result.stdout


def test_self_refiner_methods_and_failed_draft_preserve_state(template_root):
    module = importlib.import_module("examples.self_refine")

    class Backend:
        def __init__(self):
            self.inputs = []
            self.responses = iter(["draft", "add evidence", "revision"])

        async def acall(self, text):
            self.inputs.append(text)
            if len(self.inputs) == 4:
                raise RuntimeError("unavailable")
            return next(self.responses)

    async def run():
        backend = Backend()
        writer = module.SelfRefiner("Explain Slick", backend, ["Accuracy"])
        with pytest.raises(ValueError):
            await writer.critique()
        assert await writer.draft() == "draft"
        assert await writer.critique() == "add evidence"
        assert await writer.revise() == "revision"
        assert writer.feedback is None
        assert "Accuracy" in backend.inputs[1]
        assert "draft" in backend.inputs[2] and "add evidence" in backend.inputs[2]
        with pytest.raises(RuntimeError):
            await writer.draft()
        assert writer.answer == "revision"

    asyncio.run(run())


@pytest.mark.parametrize(
    "module",
    [
        "self_refine",
        "question_answerer",
        "few_shot",
        "chain_of_thought",
        "rag",
        "least_to_most",
        "self_consistency",
        "react",
        "reflexion",
        "tree_of_thoughts",
    ],
)
def test_pattern_command_runs_without_credentials(module):
    result = subprocess.run(
        [sys.executable, "-m", f"examples.{module}", "--backend", "demo"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    invalid = subprocess.run(
        [sys.executable, "-m", f"examples.{module}", "--backend", "openai"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert invalid.returncode == 2
    assert "--model" in invalid.stderr


def test_grounded_answer_checks_citations_against_retrieved_evidence(template_root):
    module = importlib.import_module("examples.rag")

    class Backend:
        async def acall(self, text):
            assert "[history]" in text
            return '{"answer":"Use messages", "citations":["invented"]}'

    qa = module.GroundedAnswerer(
        Backend(), [{"id": "history", "source": "guide", "text": "History is a list of messages."}]
    )
    with pytest.raises(ValueError, match="citation"):
        asyncio.run(qa.ask("history"))
    assert qa.answer is None

"""Run the documented examples through the real core, without API calls."""

import asyncio
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from slick import prompts


def test_session_example_runs_actual_tool_offline():
    result = subprocess.run(
        [sys.executable, "-m", "examples.session"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "pricing.md" in result.stdout
    assert "Search executions: 1" in result.stdout


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_session_example_rejects_command_providers(provider):
    result = subprocess.run(
        [sys.executable, "-m", "examples.session", "--provider", provider],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "Python tools" in result.stderr


def test_summary_history_and_retrieval_examples(monkeypatch, tmp_path):
    example = importlib.import_module("examples.core")
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))
    monkeypatch.chdir(tmp_path)

    class FakeProvider:
        def __init__(self):
            self.inputs = []

        async def acall(self, text):
            self.inputs.append(text)
            return (
                [
                    '{"headline":"Jinja to LLM", "points":["Explicit context"]}',
                    "Yes, the history is included.",
                    "Use a list of messages.",
                ][len(self.inputs) - 1],
                [],
            )

    provider = FakeProvider()
    results = asyncio.run(example.run(provider))
    assert results["summary"].headline == "Jinja to LLM"
    assert results["conversation"] == "Yes, the history is included."
    assert results["retrieval"] == "Use a list of messages."
    assert len(provider.inputs) == 3
    assert "# Output Format" in provider.inputs[0]
    assert "user: My name is Ada." in provider.inputs[1]
    assert "assistant: Hello, Ada." in provider.inputs[1]
    assert "History can be supplied as a list of messages." in provider.inputs[2]
    assert all("Write clearly" in text for text in provider.inputs)
    assert not (tmp_path / "logs").exists()


def test_question_answerer_owns_history_and_keeps_critique_separate(monkeypatch):
    example = importlib.import_module("examples.question_answerer")
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))

    class FakeProvider:
        def __init__(self):
            self.inputs = []

        async def acall(self, text):
            self.inputs.append(text)
            return (
                ["Hello, Ada.", "Your name is Ada.", "The answer is supported."][
                    len(self.inputs) - 1
                ],
                [],
            )

    async def run():
        provider = FakeProvider()
        qa = example.QuestionAnswerer(provider)
        other = example.QuestionAnswerer(provider)
        assert await qa.ask("My name is Ada.") == "Hello, Ada."
        answer = await qa.ask("What is my name?")
        history = list(qa.history)
        assert await qa.critique(answer) == "The answer is supported."
        assert (
            qa.history
            == history
            == [
                {"role": "user", "content": "My name is Ada."},
                {"role": "assistant", "content": "Hello, Ada."},
                {"role": "user", "content": "What is my name?"},
                {"role": "assistant", "content": "Your name is Ada."},
            ]
        )
        assert other.history == []
        assert "user: My name is Ada." in provider.inputs[1]
        assert "assistant: Hello, Ada." in provider.inputs[1]
        assert answer in provider.inputs[2]
        assert "What is my name?" not in provider.inputs[2]

    asyncio.run(run())


def test_question_answerer_does_not_record_failed_exchange(monkeypatch):
    example = importlib.import_module("examples.question_answerer")
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))

    class FakeProvider:
        async def acall(self, text):
            raise RuntimeError("provider unavailable")

    qa = example.QuestionAnswerer(FakeProvider())
    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(qa.ask("Hello"))
    assert qa.history == []

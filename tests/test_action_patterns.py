"""Offline checks of action execution, external feedback, and loop budgets."""

import asyncio
import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from examples._cli import ScriptedProvider
from slick import prompts


class RecordingProvider(ScriptedProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.inputs = []

    async def acall(self, text):
        self.inputs.append(text)
        return await super().acall(text)


@pytest.fixture(autouse=True)
def template_root(monkeypatch):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(__file__).parents[1] / "examples/prompts")


def test_researcher_executes_search_and_supplies_observations():
    react = importlib.import_module("examples.react")
    provider = RecordingProvider(
        [
            '{"kind":"search","tool":"search_documents","query":"launch"}',
            '{"kind":"finish","answer":"Launch is Tuesday.","citations":["schedule"]}',
        ]
    )
    researcher = react.Researcher(
        provider,
        "When is launch?",
        max_steps=2,
        documents=[{"id": "schedule", "source": "Calendar", "text": "Launch is Tuesday."}],
    )
    result = asyncio.run(researcher.run())
    assert result.answer == "Launch is Tuesday."
    assert researcher.observations[0]["result"][0]["text"] == "Launch is Tuesday."
    assert "Launch is Tuesday." in provider.inputs[1]
    assert researcher.steps == 2


def test_researcher_rejects_unregistered_actions_and_unseen_citations():
    react = importlib.import_module("examples.react")
    for response, error in [
        ('{"kind":"search","tool":"shell","query":"touch /tmp/no"}', ValueError),
        ('{"kind":"finish","answer":"Unknown","citations":["invented"]}', ValueError),
        ('{"kind":"execute","code":"print(1)"}', ValidationError),
    ]:
        researcher = react.Researcher(ScriptedProvider([response]), "Find facts")
        with pytest.raises(error):
            asyncio.run(researcher.step())
        assert researcher.observations == []


def test_researcher_budget_applies_to_step_and_run():
    react = importlib.import_module("examples.react")
    provider = RecordingProvider(
        [
            '{"kind":"search","tool":"search_documents","query":"missing"}',
        ]
    )
    researcher = react.Researcher(provider, "Find facts", max_steps=1)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        asyncio.run(researcher.run())
    with pytest.raises(RuntimeError, match="budget exhausted"):
        asyncio.run(researcher.step())
    assert len(provider.inputs) == 1
    assert researcher.observations[0]["result"] == []


def test_reflections_follow_checker_feedback_and_reach_next_attempt():
    reflexion = importlib.import_module("examples.reflexion")
    provider = RecordingProvider(
        [
            '{"numbers":[2,2,14]}',
            '{"lesson":"Check distinctness before submitting."}',
            '{"numbers":[2,4,12]}',
        ]
    )
    solver = reflexion.ReflectiveSolver(provider, max_attempts=2)
    result = asyncio.run(solver.run())
    assert result.numbers == [2, 4, 12]
    assert solver.lessons == ["Check distinctness before submitting."]
    assert "distinct" in provider.inputs[1]
    assert solver.lessons[0] in provider.inputs[2]
    assert solver.history[0]["feedback"]["passed"] is False
    assert solver.history[1]["feedback"]["passed"] is True


@pytest.mark.parametrize("numbers", [[2, 2, 14], [2, 4, 10], [1, 4, 13], [-2, 4, 16], [18]])
def test_reflective_checker_rejects_wrong_answers(numbers):
    reflexion = importlib.import_module("examples.reflexion")
    solver = reflexion.ReflectiveSolver(ScriptedProvider([]))
    feedback = solver.evaluate(reflexion.Candidate(numbers=numbers))
    assert not feedback.passed
    assert feedback.errors


def test_reflective_solver_exhaustion_and_strict_integer_parsing():
    reflexion = importlib.import_module("examples.reflexion")
    provider = RecordingProvider(['{"numbers":[2,2,14]}'])
    solver = reflexion.ReflectiveSolver(provider, max_attempts=1)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        asyncio.run(solver.run())
    with pytest.raises(RuntimeError, match="budget exhausted"):
        asyncio.run(solver.attempt())
    assert len(provider.inputs) == 1
    invalid = reflexion.ReflectiveSolver(ScriptedProvider(['{"numbers":[true,4,13]}']))
    with pytest.raises(ValidationError):
        asyncio.run(invalid.attempt())


@pytest.mark.parametrize("module,budget", [("react", "--max-steps"), ("reflexion", "--attempts")])
def test_action_clis_demo_success_and_exhaustion(module, budget, capsys):
    main = importlib.import_module(f"examples.{module}").main
    assert main([]) == 0
    assert capsys.readouterr().out
    assert main([budget, "1"]) == 1
    assert "budget exhausted" in capsys.readouterr().err

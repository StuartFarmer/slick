"""Behavioral contracts for the bounded search and decomposition examples."""

import asyncio
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from slick import prompts

ROOT = Path(__file__).resolve().parents[1]


class Backend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    async def acall(self, text):
        self.inputs.append(text)
        return json.dumps(next(self.responses))


@pytest.fixture(autouse=True)
def template_root(monkeypatch):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", ROOT / "examples" / "prompts")


def test_self_consistency_counts_answers_without_a_judge():
    module = importlib.import_module("examples.self_consistency")
    backend = Backend(
        [
            {"answer": " Paris ", "approach": "Recall"},
            {"answer": "Lyon", "approach": "Guess"},
            {"answer": "PARIS", "approach": "Geography"},
        ]
    )
    sampler = module.SolutionSampler("Capital of France?", backend)
    with pytest.raises(ValueError, match="sample"):
        sampler.choose()
    asyncio.run(sampler.sample(3))
    assert sampler.choose().answer == "Paris"
    assert sampler.votes == {"paris": 2, "lyon": 1}
    assert len(backend.inputs) == 3
    assert all("answer" in text and "properties" in text for text in backend.inputs)


def test_sampler_rejects_blank_answer():
    module = importlib.import_module("examples.self_consistency")
    sampler = module.SolutionSampler("Question", Backend([{"answer": "  ", "approach": "a"}]))
    with pytest.raises(ValidationError):
        asyncio.run(sampler.sample(1))
    assert sampler.samples == []


def test_tree_search_prunes_and_expands_partial_state():
    module = importlib.import_module("examples.tree_of_thoughts")
    backend = Backend(
        [
            {"candidates": [{"step": "promising"}, {"step": "discard me"}]},
            {"score": 0.9},
            {"score": 0.1},
            {"candidates": [{"step": "finish", "answer": "done"}]},
            {"score": 1.0},
        ]
    )
    search = module.ThoughtSearch("Task", backend, max_depth=2, beam_width=1, breadth=2)
    result = asyncio.run(search.run())
    assert result.answer == "done"
    assert result.steps == ("promising", "finish")
    assert search.frontier == [result]
    assert "promising" in backend.inputs[3]
    assert "discard me" not in backend.inputs[3]
    assert "properties" in backend.inputs[0]


def test_tree_depth_exhaustion_and_nonfinite_scores():
    module = importlib.import_module("examples.tree_of_thoughts")
    backend = Backend([{"candidates": [{"step": "partial"}]}, {"score": 0.5}])
    search = module.ThoughtSearch("Task", backend, max_depth=1, beam_width=1, breadth=1)
    result = asyncio.run(search.run())
    assert result.answer is None
    assert len(backend.inputs) == 2
    invalid = module.ThoughtSearch("Task", Backend([{"score": float("nan")}]))
    with pytest.raises(ValidationError):
        asyncio.run(invalid.evaluate(module.ThoughtState(steps=("a",))))


def test_tree_rejects_excessive_expansion():
    module = importlib.import_module("examples.tree_of_thoughts")
    backend = Backend([{"candidates": [{"step": "a"}, {"step": "b"}]}])
    search = module.ThoughtSearch("Task", backend, breadth=1)
    with pytest.raises(ValueError, match="breadth"):
        asyncio.run(search.expand(module.ThoughtState()))


def test_least_to_most_uses_decomposition_and_prior_answers():
    module = importlib.import_module("examples.least_to_most")
    backend = Backend(
        [
            {"subproblems": ["Find unit cost", "Find item count"]},
            {"answer": "$3"},
            {"answer": "4"},
            {"answer": "$12"},
        ]
    )
    solver = module.LeastToMost("Find total cost", backend, max_subproblems=2)
    assert asyncio.run(solver.run()) == "$12"
    assert len(solver.solutions) == 3
    assert "Find unit cost" in backend.inputs[1]
    assert "$3" in backend.inputs[2]
    assert "$3" in backend.inputs[3] and "4" in backend.inputs[3]
    assert "Find total cost" in backend.inputs[3]
    assert len(backend.inputs) == 4


def test_least_to_most_rejects_plan_over_bound():
    module = importlib.import_module("examples.least_to_most")
    solver = module.LeastToMost("Task", Backend([{"subproblems": ["a", "b"]}]), max_subproblems=1)
    with pytest.raises(ValueError, match="max_subproblems"):
        asyncio.run(solver.decompose())
    assert solver.subproblems == []


@pytest.mark.parametrize("name", ["self_consistency", "tree_of_thoughts", "least_to_most"])
def test_cli_runs_demo_and_rejects_invalid_limits(name):
    result = subprocess.run(
        [sys.executable, "-m", f"examples.{name}", "--backend", "demo"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    limit = {
        "self_consistency": "--samples",
        "tree_of_thoughts": "--depth",
        "least_to_most": "--max-subproblems",
    }[name]
    invalid = subprocess.run(
        [sys.executable, "-m", f"examples.{name}", limit, "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert invalid.returncode == 2


def test_tree_cli_signals_incomplete_search():
    result = subprocess.run(
        [sys.executable, "-m", "examples.tree_of_thoughts", "--depth", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "partial" in result.stdout.lower()


@pytest.mark.parametrize("score", [-0.1, 1.1, float("inf"), "high"])
def test_tree_rejects_invalid_numeric_scores(score):
    module = importlib.import_module("examples.tree_of_thoughts")
    search = module.ThoughtSearch("Task", Backend([{"score": score}]))
    with pytest.raises(ValidationError):
        asyncio.run(search.evaluate(module.ThoughtState(steps=("a",))))


@pytest.mark.parametrize("limit", ["max_depth", "beam_width", "breadth"])
@pytest.mark.parametrize("value", [0, -1, 1.5, float("inf"), True])
def test_tree_constructor_validates_all_bounds(limit, value):
    module = importlib.import_module("examples.tree_of_thoughts")
    with pytest.raises(ValueError, match=limit):
        module.ThoughtSearch("Task", Backend([]), **{limit: value})


def test_sampler_normalizes_whitespace_and_breaks_ties_by_first_sample():
    module = importlib.import_module("examples.self_consistency")
    backend = Backend(
        [
            {"answer": "New  York", "approach": "a"},
            {"answer": "Boston", "approach": "b"},
            {"answer": " new york ", "approach": "c"},
            {"answer": "BOSTON", "approach": "d"},
        ]
    )
    sampler = module.SolutionSampler("Task", backend)
    assert asyncio.run(sampler.run(4)).answer == "New  York"
    assert sampler.votes == {"new york": 2, "boston": 2}


def test_tree_dead_end_stops_without_more_calls():
    module = importlib.import_module("examples.tree_of_thoughts")
    backend = Backend([{"candidates": []}])
    search = module.ThoughtSearch("Task", backend, max_depth=5)
    assert asyncio.run(search.run()).answer is None
    assert len(backend.inputs) == 1


@pytest.mark.parametrize(
    ("module_name", "class_name", "responses"),
    [
        (
            "self_consistency",
            "SolutionSampler",
            [
                {"answer": "done", "approach": "a"},
                {"answer": "done", "approach": "b"},
                {"answer": "other", "approach": "c"},
            ],
        ),
        (
            "tree_of_thoughts",
            "ThoughtSearch",
            [
                {"candidates": [{"step": "finish", "answer": "done"}]},
                {"score": 1.0},
            ],
        ),
        (
            "least_to_most",
            "LeastToMost",
            [
                {"subproblems": ["First step"]},
                {"answer": "intermediate"},
                {"answer": "done"},
            ],
        ),
    ],
)
def test_patterns_accept_fenced_json(module_name, class_name, responses):
    class FencedBackend(Backend):
        async def acall(self, text):
            response = await super().acall(text)
            return f"```json\n{response}\n```"

    module = importlib.import_module(f"examples.{module_name}")
    pattern = getattr(module, class_name)("Task", FencedBackend(responses))
    result = asyncio.run(pattern.run())
    assert (result if isinstance(result, str) else result.answer) == "done"

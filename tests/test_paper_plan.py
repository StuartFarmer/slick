"""Exercise the paper example with offline provider responses."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from examples import paper_plan as example
from examples._cli import ScriptedProvider
from slick import Inbox, Provider, Workflow, prompts, workflow


def test_assessments_run_concurrently_before_generation(monkeypatch):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))

    async def scenario():
        method, evaluation, plan = list(example.demo_provider().responses)
        started = []
        ready = asyncio.Event()

        class OfflineProvider(Provider):
            async def acall(self, context, **kwargs):
                started.append(context)
                if len(started) == 2:
                    ready.set()
                if context.startswith("Assess"):
                    await ready.wait()
                    answer = method if context.startswith("Assess the method") else evaluation
                else:
                    assert len(started) == 3
                    assert json.loads(method)["summary"] in context
                    assert json.loads(evaluation)["summary"] in context
                    answer = plan
                return answer, [{"id": "unused", "name": "unused", "arguments": {}}]

        planner = example.PaperPlanner(example.DEMO_PAPER, OfflineProvider())
        result = await asyncio.wait_for(planner.run(), 1)
        assert started[0].startswith("Assess the method")
        assert started[1].startswith("Assess the evaluation")
        assert result.plan == example.Plan.model_validate_json(plan)
        assert result.plan.tasks[0].description in example.format_report(result)

    asyncio.run(scenario())


def test_revision_uses_feedback_without_reassessment_or_mutating_the_draft(monkeypatch):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))

    async def scenario():
        planner = example.PaperPlanner(example.DEMO_PAPER, example.demo_provider())
        original = await planner.run()
        snapshot = original.model_dump()
        revised = original.plan.model_copy(update={"objective": "Compare both classifiers"})
        contexts = []

        class OfflineProvider(ScriptedProvider):
            async def acall(self, context, **kwargs):
                contexts.append(context)
                return await super().acall(context, **kwargs)

        planner.provider = OfflineProvider([revised.model_dump_json(), "invalid JSON"])
        result = await planner.revise(
            original, "Make the comparison explicit.", provider=planner.provider
        )
        assert result.plan == revised
        assert result.method == original.method and result.evaluation == original.evaluation
        assert contexts[0].startswith("Revise the plan")
        assert "Make the comparison explicit." in contexts[0]
        assert original.plan.objective in contexts[0]
        assert len(contexts) == 1
        with pytest.raises(ValidationError):
            await planner.revise(original, "Try again.", provider=planner.provider)
        assert len(contexts) == 2  # No implicit retries.
        assert original.model_dump() == snapshot
        with pytest.raises(ValidationError):
            await planner.revise(original, "   ", provider=planner.provider)
        assert len(contexts) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["assess_method", "assess_evaluation"])
def test_assessment_rejects_invented_evidence(monkeypatch, method):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))
    response = json.loads(next(example.demo_provider().responses))
    response["evidence"] = ["An invented quotation."]
    planner = example.PaperPlanner(example.DEMO_PAPER, ScriptedProvider([json.dumps(response)]))
    with pytest.raises(ValueError, match="quote"):
        asyncio.run(getattr(planner, method)(provider=planner.provider))


def test_generation_errors_propagate_and_unknowns_remain_visible(monkeypatch):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))
    method, evaluation, plan = list(example.demo_provider().responses)
    method = example.Assessment.model_validate_json(method)
    evaluation = example.Assessment.model_validate_json(evaluation)
    method.questions = ["Which random seed should be used?"]
    raw_plan = json.loads(plan)
    raw_plan["questions"] = method.questions
    contexts = []

    class OfflineProvider(ScriptedProvider):
        async def acall(self, context, **kwargs):
            contexts.append(context)
            return await super().acall(context, **kwargs)

    planner = example.PaperPlanner(
        example.DEMO_PAPER, OfflineProvider(["invalid JSON", json.dumps(raw_plan)])
    )
    with pytest.raises(ValidationError):
        asyncio.run(planner.generate_plan(method, evaluation, provider=planner.provider))
    assert len(contexts) == 1
    result = asyncio.run(planner.generate_plan(method, evaluation, provider=planner.provider))
    assert method.questions[0] in contexts[1]
    assert method.questions[0] in example.format_report(result)


def test_cli_demo_and_input_validation(tmp_path, capsys):
    assert example.main([]) == 0
    assert "DEMO" in capsys.readouterr().out
    paper = tmp_path / "paper.md"
    paper.write_text(example.DEMO_PAPER)
    with pytest.raises(SystemExit) as error:
        example.main([str(paper)])
    assert error.value.code == 2
    assert "demo" in capsys.readouterr().err.lower()
    with pytest.raises(ValueError, match="paper"):
        example.PaperPlanner("  ", None)


def test_cli_reads_markdown_with_offline_provider(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))
    paper = tmp_path / "paper.md"
    paper.write_text(example.DEMO_PAPER + "\nOCR note: café.", encoding="utf-8")
    contexts = []

    class OfflineProvider(ScriptedProvider):
        async def acall(self, context, **kwargs):
            contexts.append(context)
            return await super().acall(context, **kwargs)

    provider = OfflineProvider(example.demo_provider().responses)
    monkeypatch.setattr(example, "provider_from_args", lambda *args: provider)
    assert example.main([str(paper), "--provider", "openai", "--model", "offline"]) == 0
    assert len(contexts) == 3 and all("OCR note: café." in text for text in contexts)
    assert "DEMO" not in capsys.readouterr().out


def test_review_channels_revise_then_approve_or_reject(monkeypatch, tmp_path):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))

    async def scenario():
        inbox = Inbox(tmp_path / "reviews.db", poll_interval=0.001)
        reviewer = Inbox(inbox.path, poll_interval=0.001)
        planner = example.PaperPlanner(example.DEMO_PAPER, example.demo_provider())
        draft = await planner.run()
        revised = draft.plan.model_copy(update={"objective": "A revised objective"})
        planner.provider = ScriptedProvider([revised.model_dump_json()])

        async def next_request():
            while not (pending := reviewer.pending()):
                await asyncio.sleep(0.001)
            return pending[0]

        async def review():
            first, message = await next_request()
            assert message == draft.model_dump(mode="json")
            reviewer.send(first, {"action": "revise", "feedback": "Clarify the objective."})
            second, new_draft = await next_request()
            assert second != first
            with pytest.raises(ValueError, match="already"):
                reviewer.send(first, {"action": "approve"})
            assert new_draft["plan"] == revised.model_dump(mode="json")
            reviewer.send(second, {"action": "approve"})

        result, _ = await asyncio.gather(planner.review(draft, inbox), review())
        assert result.plan == revised
        assert result.method == draft.method
        assert draft.plan != revised

        async def reject():
            channel, _ = await next_request()
            reviewer.send(channel, {"action": "reject"})

        result, _ = await asyncio.gather(planner.review(draft, inbox), reject())
        assert result is None

    asyncio.run(asyncio.wait_for(scenario(), 2))
    for message in (
        {"action": "revise"},
        {"action": "revise", "feedback": " "},
        {"action": "unknown"},
    ):
        with pytest.raises(ValidationError):
            example.ReviewDecision.model_validate(message)


def test_cli_review_from_another_process(tmp_path):
    async def scenario():
        path = tmp_path / "reviews.db"
        inbox = Inbox(path)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "examples.paper_plan",
            "--inbox",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            while not (pending := inbox.pending()):
                await asyncio.sleep(0.01)
            channel, message = pending[0]
            draft = example.PaperPlan.model_validate(message)
            inbox.send(channel, example.ReviewDecision(action="approve"))
            stdout, stderr = await process.communicate()
            assert process.returncode == 0, stderr.decode()
            assert draft.plan.objective in stdout.decode()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_cli_recovers_after_process_death_without_new_model_calls(tmp_path):
    async def scenario():
        path = tmp_path / "reviews.db"
        inbox = Inbox(path)
        arguments = ["--inbox", str(path), "--run-id", "paper-42"]

        async def launch(*command):
            return await asyncio.create_subprocess_exec(
                sys.executable,
                *command,
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        first = await launch("-m", "examples.paper_plan")
        try:
            while not (pending := inbox.pending()):
                await asyncio.sleep(0.01)
            channel, message = pending[0]
            competing = await launch("-m", "examples.paper_plan")
            try:
                _, error = await competing.communicate()
                assert competing.returncode != 0
                assert b"already active" in error
            finally:
                if competing.returncode is None:
                    competing.kill()
                    await competing.wait()
        finally:
            if first.returncode is None:
                first.kill()
            await first.wait()

        inbox.send(channel, example.ReviewDecision(action="approve"))
        # A new interpreter supplies an empty provider: any repeated model call fails.
        second = await launch(
            "-c",
            "from examples import paper_plan; from examples._cli import ScriptedProvider; "
            "paper_plan.demo_provider = lambda: ScriptedProvider([]); "
            "raise SystemExit(paper_plan.main())",
        )
        try:
            stdout, stderr = await second.communicate()
            assert second.returncode == 0, stderr.decode()
            assert message["plan"]["objective"] in stdout.decode()
            assert inbox.pending() == []
        finally:
            if second.returncode is None:
                second.kill()
                await second.wait()

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_recovery_restores_a_completed_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", Path(example.__file__).with_name("prompts"))

    @workflow
    async def generate_and_review(planner, inbox):
        draft = await planner.run()
        return await planner.review(draft, inbox)

    async def scenario():
        inbox = Inbox(tmp_path / "review.db", poll_interval=0.001)
        responses = list(example.demo_provider().responses)
        revised = example.Plan.model_validate_json(responses[-1])
        revised.objective = "A saved revision"
        planner = example.PaperPlanner(
            example.DEMO_PAPER, ScriptedProvider([*responses, revised.model_dump_json()])
        )

        async def execute():
            async with Workflow(inbox.path, run_id="one"):
                return await generate_and_review(planner, inbox)

        waiting = asyncio.create_task(execute())
        try:
            while not (pending := inbox.pending()):
                await asyncio.sleep(0.001)
            inbox.send(pending[0][0], {"action": "revise", "feedback": "Clarify the objective."})
            while not (pending := inbox.pending()):
                await asyncio.sleep(0.001)
            channel, message = pending[0]
            assert message["plan"]["objective"] == revised.objective
        finally:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting

        planner.provider = ScriptedProvider([])
        inbox.send(channel, {"action": "approve"})
        result = await execute()
        assert isinstance(result, example.PaperPlan)
        assert result.plan == revised

    asyncio.run(asyncio.wait_for(scenario(), 3))

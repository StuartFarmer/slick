"""Assess a paper, draft an ordered task plan, and revise it with explicit feedback."""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator, validate_call

from examples._cli import ScriptedProvider, parser, provider_from_args
from slick import Inbox, Prompt, Session, Workflow, prompt, workflow

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Assessment(BaseModel, extra="forbid"):
    summary: Text
    requirements: list[Text] = Field(min_length=1)
    evidence: list[Text] = Field(min_length=1)
    questions: list[Text]


class Task(BaseModel, extra="forbid"):
    description: Text
    acceptance: list[Text] = Field(min_length=1)


class Plan(BaseModel, extra="forbid"):
    objective: Text
    assumptions: list[Text]
    questions: list[Text]
    tasks: list[Task] = Field(min_length=1)


class PaperPlan(BaseModel, extra="forbid"):
    method: Assessment
    evaluation: Assessment
    plan: Plan


class ReviewDecision(BaseModel, extra="forbid"):
    action: Literal["approve", "revise", "reject"]
    feedback: Text | None = None

    @model_validator(mode="after")
    def require_revision_feedback(self):
        if self.action == "revise" and self.feedback is None:
            raise ValueError("Revision requires feedback")
        return self


def check_evidence(assessment: Assessment, paper: str) -> None:
    """Check quotation provenance; a matching quote does not prove the claim."""
    for quote in assessment.evidence:
        if quote not in paper:
            raise ValueError(f"Evidence quote is not present in the supplied paper: {quote!r}")


class PaperPlanner:
    """Explicit assessment, generation, and revision; the caller owns retries."""

    def __init__(self, paper: str, provider):
        if not isinstance(paper, str) or not paper.strip():
            raise ValueError("paper must contain Markdown text")
        self.paper = paper
        self.provider = provider

    @prompt(template="paper_plan/method.j2", output_type=Assessment)
    async def assess_method(self, *, generated: Assessment) -> Assessment:
        check_evidence(generated, self.paper)
        return generated

    @prompt(template="paper_plan/evaluation.j2", output_type=Assessment)
    async def assess_evaluation(self, *, generated: Assessment) -> Assessment:
        check_evidence(generated, self.paper)
        return generated

    @prompt(template="paper_plan/plan.j2", output_type=Plan, max_turns=20)
    async def generate_plan(
        self, method: Assessment, evaluation: Assessment, *, generated: Plan
    ) -> PaperPlan:
        return PaperPlan(method=method, evaluation=evaluation, plan=generated)

    @validate_call
    @prompt(template="paper_plan/revise.j2", output_type=Plan, max_turns=20)
    async def revise(self, draft: PaperPlan, feedback: Text, *, generated: Plan) -> PaperPlan:
        """Validate inputs before generation, then wrap the new plan without changing the draft."""
        return PaperPlan(method=draft.method, evaluation=draft.evaluation, plan=generated)

    async def run(self, *, session: Session | None = None) -> PaperPlan:
        # Let both assessments finish before propagating either failure.
        assessments = await asyncio.gather(
            self.assess_method(provider=self.provider),
            self.assess_evaluation(provider=self.provider),
            return_exceptions=True,
        )
        for assessment in assessments:
            if isinstance(assessment, BaseException):
                raise assessment
        method, evaluation = assessments
        return await self.generate_plan(
            method, evaluation, session=session or Session(provider=self.provider)
        )

    @workflow
    async def review(
        self, draft: PaperPlan, inbox: Inbox, *, session: Session | None = None
    ) -> PaperPlan | None:
        """Wait for review; publish each revision on a fresh channel."""
        while True:
            decision = await inbox.request(draft, output_type=ReviewDecision)
            if decision.action == "approve":
                return draft
            if decision.action == "reject":
                return None
            draft = await self.revise(
                draft, decision.feedback, session=session or Session(provider=self.provider)
            )


def format_report(result: PaperPlan) -> str:
    return Prompt("paper_plan/report.j2")(**result.model_dump())


DEMO_PAPER = """# Majority baseline on synthetic labels (fictional teaching paper)

Generate 1,000 binary labels using random.Random(7), with probability 0.7 of label 1.
Use the first 800 labels for training and the final 200 for testing, without shuffling.
Fit a majority-class classifier on training labels; ties predict 0.
Compare test accuracy with an always-zero baseline on the same test split.
Report both accuracies and their difference. This example claims no numeric result.
"""


def demo_provider():
    """Three canned responses for DEMO_PAPER; no model reasoning."""
    method = Assessment(
        summary="A fully specified synthetic baseline experiment.",
        requirements=["Use Python random.Random(7) and preserve sample order."],
        evidence=["Fit a majority-class classifier on training labels; ties predict 0."],
        questions=[],
    )
    evaluation = Assessment(
        summary="Compare classifiers on held-out labels; no target score is claimed.",
        requirements=["Report both test accuracies and their difference."],
        evidence=["This example claims no numeric result."],
        questions=[],
    )
    plan = Plan(
        objective="Implement the fictional synthetic-label experiment.",
        assumptions=["Record the Python version with the results."],
        questions=[],
        tasks=[
            Task(
                description="Generate labels and split 800/200 in original order.",
                acceptance=["Exactly 1,000 binary labels; reruns with seed 7 are identical."],
            ),
            Task(
                description="Fit the majority classifier using only training labels.",
                acceptance=["Ties predict zero; held-out labels never enter fitting."],
            ),
            Task(
                description="Measure majority and always-zero accuracy on the same test labels.",
                acceptance=["Save both accuracies, their difference, seed and Python version."],
            ),
        ],
    )
    return ScriptedProvider(item.model_dump_json() for item in (method, evaluation, plan))


def main(argv=None):
    argument_parser = parser(__doc__)
    argument_parser.add_argument("paper", nargs="?", type=Path, help="UTF-8 OCR Markdown file")
    argument_parser.add_argument("--inbox", type=Path, help="SQLite file for external review")
    argument_parser.add_argument("--run-id", help="Persist or resume this workflow in --inbox")
    args = argument_parser.parse_args(argv)
    if args.run_id is not None and (not args.run_id.strip() or args.inbox is None):
        argument_parser.error("--run-id requires a nonblank ID and --inbox")
    if args.provider == "demo" and args.paper:
        argument_parser.error(
            "demo uses its bundled fictional paper; choose a real provider for yours"
        )
    if args.provider != "demo" and args.paper is None:
        argument_parser.error("supply a paper Markdown file for a real provider")
    try:
        paper = args.paper.read_text(encoding="utf-8") if args.paper else DEMO_PAPER
    except (OSError, UnicodeError) as error:
        argument_parser.error(str(error))
    if not paper.strip():
        argument_parser.error("paper must contain Markdown text")
    provider = provider_from_args(args, argument_parser, demo_provider())
    planner = PaperPlanner(paper, provider)
    inbox = Inbox(args.inbox) if args.inbox else None

    @workflow
    async def run():
        draft = await planner.run()
        return await planner.review(draft, inbox) if inbox is not None else draft

    async def execute():
        if args.run_id is None:
            return await run()
        async with Workflow(args.inbox, run_id=args.run_id, inputs={"paper": paper}):
            return await run()

    result = asyncio.run(execute())
    if result is None:
        print("Plan rejected.", file=sys.stderr)
        return 1
    if args.provider == "demo":
        print("> DEMO: canned responses for a fictional paper; no model assessment.\n")
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

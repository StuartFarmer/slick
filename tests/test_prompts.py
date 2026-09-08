"""Prompt-as-function tests: templates, rendering, returns, repair, cache."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import Literal

from jinja2 import UndefinedError
from pydantic import BaseModel

from slick import prompts as prompts_module
from slick.prompts import PromptError, prompt


class Verdict(BaseModel):
    approved: bool
    reasons: list[str]


class StubProvider:
    """Hands back canned responses in order, repeating the last one."""

    provider = "stub"
    model = None

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses) or [""]
        self.prompts: list[str] = []

    def call(self, prompt_text: str) -> str:
        self.prompts.append(prompt_text)
        return self.responses[min(len(self.prompts), len(self.responses)) - 1]


class Logged(unittest.TestCase):
    """Base for tests that let a prompt reach a model."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        # Cache hits announce themselves on stderr; keep the runner clean.
        redirect = redirect_stderr(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def runs(self) -> list[Path]:
        return sorted(self.tmp.iterdir())


class Rooted(unittest.TestCase):
    """Base for tests that keep template files in a temp TEMPLATE_ROOT."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        original = prompts_module.TEMPLATE_ROOT
        prompts_module.TEMPLATE_ROOT = self.root
        self.addCleanup(setattr, prompts_module, "TEMPLATE_ROOT", original)

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


class TemplateFiles(Rooted):
    def test_a_named_file_is_the_prompt_and_the_docstring_is_documentation(self):
        self.write("summarize.md.j2", "Summarize for {{ audience }}.\n\n{{ document }}")

        @prompt(template="summarize.md.j2")
        def summarize(document: str, audience: str = "an engineer") -> str:
            """Summarize a document for one audience."""

        rendered = summarize.render("DOCBODY")
        self.assertIn("Summarize for an engineer.", rendered)
        self.assertIn("DOCBODY", rendered)
        self.assertNotIn("Summarize a document for one audience.", rendered)
        self.assertEqual(summarize.template_name, "summarize.md.j2")
        self.assertIn("{{ document }}", summarize.source())

    def test_a_template_file_needs_no_docstring(self):
        self.write("bare.md.j2", "{{ document }}")

        @prompt(template="bare.md.j2")
        def bare(document: str) -> str: ...

        self.assertEqual(bare.render("DOCBODY"), "DOCBODY")

    def test_include_resolves_against_the_template_root(self):
        self.write("shared/house_style.md", "Write plainly.")
        self.write(
            "summarize.md.j2",
            '{% include "shared/house_style.md" %}\n\n{{ document }}',
        )

        @prompt(template="summarize.md.j2")
        def summarize(document: str) -> str:
            """Summarize a document."""

        rendered = summarize.render("DOCBODY")
        self.assertIn("Write plainly.", rendered)
        self.assertIn("DOCBODY", rendered)

    def test_a_template_can_live_in_a_subdirectory(self):
        self.write("nested/report.md.j2", "Report on {{ document }}")

        @prompt(template="nested/report.md.j2")
        def report(document: str) -> str:
            """Write a report."""

        self.assertEqual(report.render("DOCBODY"), "Report on DOCBODY")

    def test_editing_the_file_takes_effect_on_the_next_call(self):
        self.write("summarize.md.j2", "First: {{ document }}")

        @prompt(template="summarize.md.j2")
        def summarize(document: str) -> str:
            """Summarize a document."""

        self.assertEqual(summarize.render("DOCBODY"), "First: DOCBODY")
        self.write("summarize.md.j2", "Second: {{ document }}")
        self.assertEqual(summarize.render("DOCBODY"), "Second: DOCBODY")

    def test_a_missing_template_names_the_file_and_the_root(self):
        @prompt(template="gone.md.j2")
        def missing(document: str) -> str:
            """Summarize a document."""

        for attempt in (lambda: missing.render("DOCBODY"), missing.source):
            with self.assertRaises(PromptError) as caught:
                attempt()
            self.assertIn("gone.md.j2", str(caught.exception))
            self.assertIn(str(self.root), str(caught.exception))

    def test_the_schema_is_appended_to_a_file_that_omits_it(self):
        self.write("judge.md.j2", "Judge {{ document }}")

        @prompt(template="judge.md.j2")
        def judge(document: str) -> Verdict:
            """Judge a document."""

        rendered = judge.render("DOCBODY")
        self.assertIn("# Output Format", rendered)
        self.assertLess(rendered.index("DOCBODY"), rendered.index("# Output Format"))

    def test_a_file_can_place_the_schema_itself(self):
        self.write("judge.md.j2", "{{ output_format }}\n\nJudge {{ document }}")

        @prompt(template="judge.md.j2")
        def judge(document: str) -> Verdict:
            """Judge a document."""

        rendered = judge.render("DOCBODY")
        self.assertEqual(rendered.count("# Output Format"), 1)
        self.assertLess(rendered.index("# Output Format"), rendered.index("DOCBODY"))


class Render(unittest.TestCase):
    def test_parameters_become_template_variables(self):
        @prompt
        def summarize(document: str, notes: str = "") -> str:
            """
            Summarize the document below.
            {% if notes %}

            # Notes
            {{ notes }}
            {% endif %}

            # Document
            {{ document }}
            """

        fresh = summarize.render("DOCBODY")
        self.assertIn("DOCBODY", fresh)
        self.assertNotIn("# Notes", fresh)

        annotated = summarize.render("DOCBODY", notes="keep it to three lines")
        self.assertIn("keep it to three lines", annotated)
        self.assertLess(annotated.index("# Notes"), annotated.index("DOCBODY"))

    def test_a_docstring_template_has_no_template_name(self):
        @prompt
        def summarize(document: str) -> str:
            """{{ document }}"""

        self.assertIsNone(summarize.template_name)
        self.assertEqual(summarize.source(), "{{ document }}")

    def test_an_unknown_variable_fails_instead_of_rendering_blank(self):
        @prompt
        def typo(document: str) -> str:
            """{{ document }} {{ documnet }}"""

        with self.assertRaises(UndefinedError):
            typo.render("DOCBODY")

    def test_the_body_can_add_computed_variables(self):
        @prompt
        def sized(document: str) -> str:
            """{{ document }} ({{ words }} words)"""
            return {"words": len(document.split())}

        self.assertIn("(2 words)", sized.render("two words"))

    def test_an_ellipsis_body_is_empty_not_a_return_value(self):
        @prompt
        def bare(document: str) -> str:
            """{{ document }}"""
            ...

        self.assertEqual(bare.render("DOCBODY"), "DOCBODY")

    def test_a_body_that_returns_anything_else_is_an_error(self):
        @prompt
        def wrong(document: str) -> str:
            """{{ document }}"""
            return "not a mapping"

        with self.assertRaises(PromptError):
            wrong.render("DOCBODY")


class Decoration(unittest.TestCase):
    def test_a_reserved_parameter_name_is_rejected(self):
        with self.assertRaises(PromptError):

            @prompt
            def clash(model: str) -> str:
                """{{ model }}"""

    def test_a_prompt_with_neither_template_nor_docstring_is_rejected(self):
        with self.assertRaises(PromptError):

            @prompt
            def blank(document: str) -> str:
                return None


class TextReturns(Logged):
    def test_returns_the_response_and_logs_the_exchange(self):
        model = StubProvider("# Summary\n")

        @prompt(model=model, log_dir=self.tmp)
        def summarize(document: str) -> str:
            """{{ document }}"""

        self.assertEqual(summarize("DOCBODY"), "# Summary\n")
        (run,) = self.runs()
        self.assertEqual((run / "prompt.md").read_text(), "DOCBODY")
        self.assertEqual((run / "response.txt").read_text(), "# Summary\n")

    def test_output_also_saves_the_text(self):
        @prompt(model=StubProvider("# Summary\n"), log_dir=self.tmp)
        def summarize(document: str) -> str:
            """{{ document }}"""

        report = self.tmp / "out" / "summary.md"
        summarize("DOCBODY", output=report)
        self.assertEqual(report.read_text(), "# Summary\n")

    def test_an_empty_response_is_not_written(self):
        @prompt(model=StubProvider("  \n"), log_dir=self.tmp)
        def summarize(document: str) -> str:
            """{{ document }}"""

        report = self.tmp / "summary.md"
        with self.assertRaises(PromptError):
            summarize("DOCBODY", output=report)
        self.assertFalse(report.exists())

    def test_an_identical_prompt_is_served_from_the_cache(self):
        model = StubProvider("# Summary\n")

        @prompt(model=model, log_dir=self.tmp)
        def summarize(document: str) -> str:
            """{{ document }}"""

        self.assertEqual(summarize("DOCBODY"), summarize("DOCBODY"))
        self.assertEqual(len(model.prompts), 1)

    def test_a_changed_prompt_is_a_different_call(self):
        model = StubProvider("# Summary\n")

        @prompt(model=model, log_dir=self.tmp)
        def summarize(document: str) -> str:
            """{{ document }}"""

        summarize("DOCBODY")
        summarize("OTHERBODY")
        self.assertEqual(len(model.prompts), 2)
        self.assertEqual(len(self.runs()), 2)

    def test_cache_off_calls_every_time(self):
        model = StubProvider("# Summary\n")

        @prompt(model=model, log_dir=self.tmp, cache=False)
        def summarize(document: str) -> str:
            """{{ document }}"""

        summarize("DOCBODY")
        summarize("DOCBODY")
        self.assertEqual(len(model.prompts), 2)

    def test_a_per_call_model_overrides_the_decorated_one(self):
        decorated, override = StubProvider("from decorated"), StubProvider("from override")

        @prompt(model=decorated, log_dir=self.tmp)
        def summarize(document: str) -> str:
            """{{ document }}"""

        self.assertEqual(summarize("DOCBODY", model=override), "from override")
        self.assertEqual(decorated.prompts, [])


class TypedReturns(Logged):
    def test_the_schema_is_appended_when_the_template_never_asks(self):
        @prompt(model=StubProvider(), log_dir=self.tmp)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        rendered = judge.render("DOCBODY")
        self.assertIn("# Output Format", rendered)
        self.assertIn("reasons", rendered)
        self.assertLess(rendered.index("DOCBODY"), rendered.index("# Output Format"))

    def test_the_schema_lands_where_the_template_puts_it(self):
        @prompt(model=StubProvider(), log_dir=self.tmp)
        def judge(document: str) -> Verdict:
            """
            {{ output_format }}

            # Document
            {{ document }}
            """

        rendered = judge.render("DOCBODY")
        self.assertEqual(rendered.count("# Output Format"), 1)
        self.assertLess(rendered.index("# Output Format"), rendered.index("DOCBODY"))

    def test_a_text_return_gets_no_schema(self):
        @prompt(model=StubProvider(), log_dir=self.tmp)
        def summarize(document: str) -> str:
            """{{ document }}{{ output_format }}"""

        self.assertEqual(summarize.render("DOCBODY"), "DOCBODY")

    def test_a_fenced_json_response_is_validated(self):
        model = StubProvider('```json\n{"approved": true, "reasons": []}\n```')

        @prompt(model=model, log_dir=self.tmp)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        verdict = judge("DOCBODY")
        self.assertIsInstance(verdict, Verdict)
        self.assertTrue(verdict.approved)

    def test_prose_around_the_json_is_tolerated(self):
        model = StubProvider('Sure! {"approved": false, "reasons": ["no data"]} Hope that helps.')

        @prompt(model=model, log_dir=self.tmp)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        self.assertEqual(judge("DOCBODY").reasons, ["no data"])

    def test_a_bare_scalar_answer_is_coerced(self):
        @prompt(model=StubProvider("approve"), log_dir=self.tmp)
        def decide(document: str) -> Literal["approve", "reject"]:
            """{{ document }}"""

        self.assertEqual(decide("DOCBODY"), "approve")

    def test_output_is_refused_for_a_structured_return(self):
        @prompt(model=StubProvider(), log_dir=self.tmp)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        with self.assertRaises(PromptError):
            judge("DOCBODY", output=self.tmp / "verdict.json")

    def test_a_cached_structured_response_is_parsed_without_a_call(self):
        model = StubProvider('{"approved": true, "reasons": []}')

        @prompt(model=model, log_dir=self.tmp)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        self.assertEqual(judge("DOCBODY"), judge("DOCBODY"))
        self.assertEqual(len(model.prompts), 1)


class Repair(Logged):
    def test_an_unparseable_response_is_handed_back_once(self):
        model = StubProvider("I could not do it.", '{"approved": true, "reasons": []}')

        @prompt(model=model, log_dir=self.tmp)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        self.assertTrue(judge("DOCBODY").approved)
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("# Repair", model.prompts[1])
        self.assertIn("I could not do it.", model.prompts[1])

        (run,) = self.runs()
        self.assertEqual((run / "rejected.1.txt").read_text(), "I could not do it.")
        self.assertIn("approved", (run / "response.txt").read_text())

    def test_the_repair_budget_runs_out(self):
        model = StubProvider("nope", "still nope")

        @prompt(model=model, log_dir=self.tmp)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        with self.assertRaises(PromptError):
            judge("DOCBODY")
        self.assertEqual(len(model.prompts), 2)

        # Nothing parsed, so nothing was cached: the next run starts clean.
        (run,) = self.runs()
        self.assertFalse((run / "response.txt").exists())

    def test_no_repair_budget_fails_on_the_first_response(self):
        model = StubProvider("nope", '{"approved": true, "reasons": []}')

        @prompt(model=model, log_dir=self.tmp, max_repairs=0)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        with self.assertRaises(PromptError):
            judge("DOCBODY")
        self.assertEqual(len(model.prompts), 1)


if __name__ == "__main__":
    unittest.main()

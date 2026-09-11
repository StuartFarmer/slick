"""Prompt-as-function tests: templates, rendering, and returns."""

import tempfile
import unittest
from pathlib import Path
from typing import Literal

from jinja2 import TemplateNotFound, UndefinedError
from pydantic import BaseModel, ValidationError

from slick import prompts as prompts_module
from slick.prompts import prompt


class Verdict(BaseModel):
    approved: bool
    reasons: list[str]


class StubProvider:
    """Hands back canned responses in order, repeating the last one."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses) or [""]
        self.prompts: list[str] = []

    def call(self, prompt_text: str):
        self.prompts.append(prompt_text)
        return (self.responses[min(len(self.prompts), len(self.responses)) - 1], [])


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

        with self.assertRaises(TemplateNotFound) as caught:
            missing.render("DOCBODY")
        self.assertIn("gone.md.j2", str(caught.exception))
        self.assertIn(str(self.root), str(caught.exception))

    def test_the_schema_is_appended_to_a_file_that_omits_it(self):
        self.write("judge.md.j2", "Judge {{ document }}")

        @prompt(template="judge.md.j2", output_type=Verdict)
        def judge(document: str) -> Verdict:
            """Judge a document."""

        rendered = judge.render("DOCBODY")
        self.assertIn("# Output Format", rendered)
        self.assertLess(rendered.index("DOCBODY"), rendered.index("# Output Format"))

    def test_a_file_can_place_the_schema_itself(self):
        self.write("judge.md.j2", "{{ output_format }}\n\nJudge {{ document }}")

        @prompt(template="judge.md.j2", output_type=Verdict)
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

    def test_an_unknown_variable_fails_instead_of_rendering_blank(self):
        @prompt
        def typo(document: str) -> str:
            """{{ document }} {{ documnet }}"""

        with self.assertRaises(UndefinedError):
            typo.render("DOCBODY")

    def test_the_body_can_return_a_postprocessed_mapping(self):
        @prompt
        def sized(document: str) -> str:
            """{{ document }}"""
            return {"words": len(document.split())}

        self.assertEqual(sized.render("two words"), "two words")
        self.assertEqual(sized("two words", provider=StubProvider()), {"words": 2})

    def test_an_ellipsis_body_is_empty_not_a_return_value(self):
        @prompt
        def bare(document: str) -> str:
            """{{ document }}"""
            ...

        self.assertEqual(bare.render("DOCBODY"), "DOCBODY")

    def test_a_body_can_replace_the_generated_text(self):
        @prompt
        def wrong(document: str) -> str:
            """{{ document }}"""
            return "not a mapping"

        self.assertEqual(wrong.render("DOCBODY"), "DOCBODY")
        self.assertEqual(wrong("DOCBODY", provider=StubProvider()), "not a mapping")


class Decoration(unittest.TestCase):
    def test_model_is_an_ordinary_template_parameter(self):
        @prompt
        def describe(model: str) -> str:
            """{{ model }}"""

        self.assertEqual(describe.render(model="example"), "example")

    def test_a_missing_template_fails_when_jinja_renders_it(self):
        @prompt
        def blank(document: str) -> str:
            return None

        with self.assertRaises(TypeError):
            blank.render("text")


class TypedReturns(unittest.TestCase):
    def test_the_schema_is_appended_when_the_template_never_asks(self):
        @prompt(output_type=Verdict)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        rendered = judge.render("DOCBODY")
        self.assertIn("# Output Format", rendered)
        self.assertIn("reasons", rendered)
        self.assertLess(rendered.index("DOCBODY"), rendered.index("# Output Format"))

    def test_the_schema_lands_where_the_template_puts_it(self):
        @prompt(output_type=Verdict)
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
        @prompt()
        def summarize(document: str) -> str:
            """{{ document }}{{ output_format }}"""

        self.assertEqual(summarize.render("DOCBODY"), "DOCBODY")

    def test_a_json_response_is_validated(self):
        model = StubProvider('{"approved": true, "reasons": []}')

        @prompt(output_type=Verdict)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        verdict = judge("DOCBODY", provider=model)
        self.assertIsInstance(verdict, Verdict)
        self.assertTrue(verdict.approved)

    def test_prose_around_the_json_is_rejected(self):
        model = StubProvider('Sure! {"approved": false, "reasons": ["no data"]} Hope that helps.')

        @prompt(output_type=Verdict)
        def judge(document: str) -> Verdict:
            """{{ document }}"""

        with self.assertRaises(ValidationError):
            judge("DOCBODY", provider=model)
        self.assertEqual(len(model.prompts), 1)

    def test_a_json_scalar_answer_is_parsed(self):
        @prompt(output_type=Literal["approve", "reject"])
        def decide(document: str) -> Literal["approve", "reject"]:
            """{{ document }}"""

        self.assertEqual(decide("DOCBODY", provider=StubProvider('"approve"')), "approve")


if __name__ == "__main__":
    unittest.main()

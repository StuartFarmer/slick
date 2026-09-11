"""Prompt execution has one API and uses ordinary Python/Jinja/Pydantic errors."""

import asyncio
from typing import Literal

import pytest
from jinja2 import TemplateNotFound
from pydantic import ValidationError

from slick import parse, prompt, prompts, render


@pytest.mark.parametrize("text", ["```json\n[1]\n```", "Here is [1].", "approve"])
def test_structured_parsing_requires_json(text):
    with pytest.raises(ValidationError):
        parse(text, Literal["approve"] if text == "approve" else list[int])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_model_is_template_data_and_calls_have_no_implicit_side_effects(
    tmp_path, monkeypatch, asynchronous
):
    monkeypatch.chdir(tmp_path)

    class Provider:
        calls = 0

        def call(self, text):
            self.calls += 1
            return text, []

        async def acall(self, text):
            return self.call(text)

    def declaration(model: str) -> str:
        """{{ model }}"""

    async def async_declaration(model: str) -> str:
        """{{ model }}"""

    provider = Provider()
    fn = prompt(async_declaration if asynchronous else declaration)
    for _ in range(2):
        assert (
            asyncio.run(fn(model="data", provider=provider))
            if asynchronous
            else fn(model="data", provider=provider)
        ) == "data"
    assert provider.calls == 2
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    "option, value",
    [("model", "codex"), ("max_repairs", 1), ("cache", True), ("log_dir", "logs")],
)
def test_removed_decorator_options(option, value):
    with pytest.raises(TypeError, match=option):
        prompt(**{option: value})


def test_missing_template_raises_jinjas_error(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", tmp_path)
    with pytest.raises(TemplateNotFound):
        render("missing.j2")

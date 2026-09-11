"""Provider prompting is stateless and preserves tool requests in both call modes."""

import asyncio
import json

import pytest
from jinja2 import UndefinedError
from pydantic import ValidationError

from slick import Prompt, Provider, prompts
from tests.test_session_calls import Script
from tests.test_session_prompts import Assessment, SyncScript


class AsyncProvider(Script, Provider):
    pass


class SyncProvider(SyncScript, Provider):
    pass


def invoke(provider, template, asynchronous, **variables):
    if asynchronous:
        return asyncio.run(provider.aprompt(template, **variables))
    return provider.prompt(template, **variables)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("output_type", [Assessment, list[int]])
def test_typed_calls_supply_schema_and_return_unoffered_requests_without_pending_state(
    tmp_path, monkeypatch, asynchronous, output_type
):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", tmp_path)
    (tmp_path / "test.j2").write_text("{{ paper }}\n{{ schema | tojson }}")
    request = {"id": "a", "name": "search", "arguments": {"query": "x"}}
    text = '{"summary":"checked"}' if output_type is Assessment else "[1, 2]"
    provider = (AsyncProvider if asynchronous else SyncProvider)((text, [request]), (text, []))
    for requests in ([request], []):
        result = invoke(
            provider, Prompt("test.j2"), asynchronous, output_type=output_type, paper="Paper"
        )
        expected = Assessment(summary="checked") if output_type is Assessment else [1, 2]
        assert result == (expected, requests)
    context, results = provider.inputs[0]
    paper, schema = context.split("\n")
    assert paper == "Paper"
    assert json.loads(schema)["type"] == ("object" if output_type is Assessment else "array")
    assert results is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_raw_calls_leave_text_and_schema_alone_and_forward_tool_results(
    tmp_path, monkeypatch, asynchronous
):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", tmp_path)
    template = tmp_path / "test.j2"
    template.write_text("{{ schema | default('no schema') }}")
    provider = (AsyncProvider if asynchronous else SyncProvider)(("  raw text\n", []), ("next", []))
    assert invoke(provider, Prompt("test.j2"), asynchronous) == ("  raw text\n", [])
    results = [
        {
            "request": {"id": "a", "name": "search", "arguments": {"query": "x"}},
            "content": "Found",
            "is_error": False,
        }
    ]
    assert invoke(
        provider,
        Prompt("test.j2"),
        asynchronous,
        output_type=None,
        schema="manual",
        tool_results=results,
    ) == ("next", [])
    assert provider.inputs == [("no schema", None), ("manual", results)]
    template.write_text("{{ schema }}")
    with pytest.raises(UndefinedError):
        invoke(provider, Prompt("test.j2"), asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_provider_and_validation_errors_propagate_without_retries(
    tmp_path, monkeypatch, asynchronous
):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", tmp_path)
    (tmp_path / "test.j2").write_text("{{ schema | tojson }}")
    provider = (AsyncProvider if asynchronous else SyncProvider)(
        ("invalid JSON", []), RuntimeError("offline")
    )
    with pytest.raises(ValidationError):
        invoke(provider, Prompt("test.j2"), asynchronous, output_type=Assessment)
    with pytest.raises(RuntimeError, match="offline"):
        invoke(provider, Prompt("test.j2"), asynchronous, output_type=Assessment)
    assert len(provider.inputs) == 2

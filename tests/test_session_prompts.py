"""Rendered session calls preserve text, typed results, and tool bookkeeping."""

import asyncio
import json
from copy import deepcopy

import pytest
from jinja2 import UndefinedError
from pydantic import BaseModel, ValidationError

from slick import Prompt, Session, prompts
from tests.test_session_calls import Script, search


class Assessment(BaseModel):
    summary: str


class SyncScript:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.inputs = []

    def call(self, context, *, tools=None, tool_results=None):
        self.inputs.append((context, deepcopy(tool_results)))
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def invoke(session, template, asynchronous, **variables):
    if asynchronous:
        return asyncio.run(session.aprompt(template, **variables))
    return session.prompt(template, **variables)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("output_type", [Assessment, list[int]])
def test_typed_prompt_renders_schema_and_records_raw_response(
    tmp_path, monkeypatch, asynchronous, output_type
):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", tmp_path)
    (tmp_path / "test.j2").write_text("{{ paper }}\n{{ schema | tojson }}")
    text = '{"summary":"checked"}' if output_type is Assessment else "[1, 2]"
    provider = (Script if asynchronous else SyncScript)((text, []))
    session = Session()
    result, requests = invoke(
        session,
        Prompt("test.j2"),
        asynchronous,
        provider=provider,
        output_type=output_type,
        paper="Paper text",
    )
    assert result == (Assessment(summary="checked") if output_type is Assessment else [1, 2])
    assert requests == []
    context = session.history[0]["context"]
    paper, schema = context.split("\n")
    assert paper == "Paper text"
    assert json.loads(schema)["type"] == ("object" if output_type is Assessment else "array")
    assert session.history[0]["text"] == text
    assert provider.inputs == [(context, [])]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_raw_prompt_has_no_implicit_schema_or_parsing(tmp_path, monkeypatch, asynchronous):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", tmp_path)
    template = tmp_path / "test.j2"
    template.write_text("{{ paper }}{% if schema is defined %}: {{ schema }}{% endif %}")
    provider = (Script if asynchronous else SyncScript)(("  raw text\n", []), ("next", []))
    session = Session(provider=provider)
    assert invoke(session, Prompt("test.j2"), asynchronous, paper="First") == ("  raw text\n", [])
    assert invoke(
        session, Prompt("test.j2"), asynchronous, output_type=None, paper="Second", schema="manual"
    ) == ("next", [])
    assert [exchange["context"] for exchange in session.history] == ["First", "Second: manual"]
    template.write_text("{{ schema }}")
    with pytest.raises(UndefinedError):
        invoke(session, Prompt("test.j2"), asynchronous)
    assert len(session.history) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_prompt_keeps_tools_pending_and_submits_cancelled_results(
    tmp_path, monkeypatch, asynchronous
):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", tmp_path)
    (tmp_path / "test.j2").write_text("{{ paper }}")
    request = {"id": "a", "name": "search", "arguments": {"query": "x"}}
    provider = (Script if asynchronous else SyncScript)(("Searching", [request]), ("Done", []))
    session = Session(provider=provider, tools=[search])
    assert invoke(session, Prompt("test.j2"), asynchronous, paper="First") == (
        "Searching",
        [request],
    )
    assert session.pending_requests == [request]
    with pytest.raises(ValueError, match="pending"):
        invoke(session, Prompt("test.j2"), asynchronous, paper="Blocked")
    results = session.cancel_pending("Not needed")
    assert invoke(session, Prompt("test.j2"), asynchronous, paper="Second") == ("Done", [])
    assert provider.inputs[-1] == ("Second", results)
    assert session.ready_results == []
    assert session.history[0]["work"][0]["submitted"] is True


@pytest.mark.parametrize("asynchronous", [False, True])
def test_parse_failure_keeps_exchange_and_provider_failure_does_not(
    tmp_path, monkeypatch, asynchronous
):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", tmp_path)
    (tmp_path / "test.j2").write_text("{{ schema | tojson }}")
    request = {"id": "a", "name": "search", "arguments": {"query": "x"}}
    provider = (Script if asynchronous else SyncScript)(
        ("invalid JSON", [request]), RuntimeError("offline"), ('{"summary":"ok"}', [])
    )
    session = Session(provider=provider, tools=[search])
    with pytest.raises(ValidationError):
        invoke(session, Prompt("test.j2"), asynchronous, output_type=Assessment)
    assert session.history[0]["text"] == "invalid JSON"
    assert session.pending_requests == [request]
    results = session.cancel_pending("Not needed")
    with pytest.raises(RuntimeError, match="offline"):
        invoke(session, Prompt("test.j2"), asynchronous, output_type=Assessment)
    assert len(session.history) == 1
    assert session.ready_results == results
    assert invoke(session, Prompt("test.j2"), asynchronous, output_type=Assessment) == (
        Assessment(summary="ok"),
        [],
    )
    assert session.ready_results == []

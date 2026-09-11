import asyncio
import json
from copy import deepcopy

import pytest
from sdk_fakes import sdk_response
from test_tool_providers import provider, reply

from slick import Session, Workflow, workflow


@workflow
async def inspect_code(session) -> str:
    return await session.arun("Inspect the code")


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_native_conversation_preserves_reasoning_and_all_tool_turns(name):
    responses = [reply(name), reply(name, arguments={"path": "b.py"}), reply(name, calls=False)]
    for index, response in enumerate(responses[:2]):
        if name == "openai":
            response["output"].insert(
                0, {"type": "reasoning", "encrypted_content": f"opaque-{index}"}
            )
            response["output"][-1]["call_id"] = str(index)
        elif name == "anthropic":
            response["content"].insert(
                0, {"type": "thinking", "thinking": "", "signature": f"opaque-{index}"}
            )
            response["content"][-1]["id"] = str(index)
        else:
            message = response["choices"][0]["message"]
            message["reasoning_details"] = [
                {"type": "reasoning.encrypted", "data": f"opaque-{index}"}
            ]
            message["tool_calls"][0]["id"] = str(index)
    iterator = iter(responses)
    sent = []
    model = provider(name)

    async def send(payload):
        sent.append(deepcopy(payload))
        return sdk_response(next(iterator))

    model._asend = send

    def read(path: str) -> str:
        """Read a file."""
        return f"contents of {path}"

    session = Session(provider=model, tools=[read])
    assert asyncio.run(session.arun("Inspect the code")) == "  answer\n"
    final = json.dumps(sent[-1])
    assert "opaque-0" in final and "opaque-1" in final
    assert final.count("contents of a.py") == 1
    assert final.count("contents of b.py") == 1
    assert final.count("Inspect the code") == 1
    snapshot = json.loads(json.dumps(session.to_dict()))
    assert "opaque-0" in json.dumps(snapshot["continuation"])
    if name == "openai":
        assert sent[0]["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_incomplete_native_responses_are_not_final_answers(name):
    response = reply(name, calls=False, text="partial")
    if name == "openai":
        response["status"] = "incomplete"
    elif name == "anthropic":
        response["stop_reason"] = "max_tokens"
    else:
        response["choices"][0]["finish_reason"] = "length"
    model = provider(name)
    model._send = lambda payload: sdk_response(response)
    session = Session(provider=model)
    with pytest.raises(RuntimeError, match="before completing"):
        session.run("answer")
    assert session.history[-1]["text"] == "partial"


@pytest.mark.parametrize("name", ["openai", "anthropic", "chat_completions"])
def test_restart_preserves_native_turn_and_completed_tool(name, tmp_path):
    calls = []

    def read(path: str) -> str:
        """Read a file."""
        calls.append(path)
        return "saved contents"

    async def scenario():
        first = provider(name)
        responses = iter([reply(name)])

        async def fail_after_tool(payload):
            response = next(responses, None)
            if response is None:
                raise ConnectionError("offline")
            return sdk_response(response)

        first._asend = fail_after_tool
        path = tmp_path / "run.db"
        from slick.providers import ProviderError

        with pytest.raises(ProviderError):
            async with Workflow(path, run_id="native"):
                await inspect_code(Session(provider=first, tools=[read]))

        resumed = provider(name)
        sent = []

        async def finish(payload):
            sent.append(deepcopy(payload))
            return sdk_response(reply(name, calls=False))

        resumed._asend = finish
        async with Workflow(path, run_id="native"):
            assert await inspect_code(Session(provider=resumed, tools=[read])) == "  answer\n"
        assert calls == ["a.py"]
        assert len(sent) == 1
        assert json.dumps(sent[0]).count("saved contents") == 1
        assert json.dumps(sent[0]).count("Inspect the code") == 1

    asyncio.run(scenario())


def test_anthropic_pause_continues_before_returning_text():
    paused = reply("anthropic", calls=False, text="Still working")
    paused["stop_reason"] = "pause_turn"
    responses = iter([paused, reply("anthropic", calls=False, text="Finished")])
    sent = []
    model = provider("anthropic")

    def send(payload):
        sent.append(payload)
        return sdk_response(next(responses))

    model._send = send
    assert Session(provider=model).run("Inspect") == "Finished"
    assert sent[1]["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Still working"}],
    }

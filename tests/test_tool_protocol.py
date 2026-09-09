"""Tool dictionaries use ordinary Python and JSON behavior."""

import pytest

from slick.tools import ToolRequest, ToolResult, make_request


def test_protocol_types_are_plain_dictionaries():
    request = ToolRequest(id="", name="", arguments={})
    result = ToolResult(request=request, content="ok")
    assert result == {"request": request, "content": "ok"}


def test_decode_uses_standard_json_semantics():
    assert make_request("a", "read", '{"x":1,"x":2}')["arguments"] == {"x": 2}
    assert make_request("a", "read", "[]")["arguments"] == []
    request = make_request("a", "read", "bad")
    assert request["arguments"] == "bad" and request["argument_error"]


def test_prompt_does_not_accept_text_with_pending_requests(tmp_path):
    from slick import PromptError, prompt

    class Provider:
        calls = 0

        def identity(self):
            return {"provider": "test", "model": "scripted"}

        def call(self, context):
            self.calls += 1
            return "answer", [{"id": "a", "name": "read", "arguments": {}}]

    provider = Provider()

    @prompt(provider=provider, cache=True, log_dir=tmp_path)
    def answer() -> str:
        """Answer the question."""

    with pytest.raises(PromptError, match="tool requests"):
        answer()
    assert provider.calls == 1
    assert not list(tmp_path.rglob("response.txt"))

"""Tool dictionaries use ordinary Python and JSON behavior."""

import asyncio

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


@pytest.mark.parametrize("asynchronous", [False, True])
def test_prompt_does_not_accept_text_with_pending_requests(asynchronous):
    from slick import PromptError, prompt

    class Provider:
        calls = 0

        def call(self, context):
            self.calls += 1
            return "answer", [{"id": "a", "name": "read", "arguments": {}}]

        async def acall(self, context):
            return self.call(context)

    provider = Provider()

    def declaration() -> str:
        """Answer the question."""

    async def async_declaration() -> str:
        """Answer the question."""

    answer = prompt(provider=provider)(async_declaration if asynchronous else declaration)
    with pytest.raises(PromptError, match="tool requests"):
        asyncio.run(answer()) if asynchronous else answer()
    assert provider.calls == 1

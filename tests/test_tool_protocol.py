"""Stateless tool dictionaries: validation, correlation, and JSON portability."""

import json
from copy import deepcopy

import pytest


def result(**changes):
    return {
        "request": {"id": "a", "name": "read", "arguments": {"path": "a.py"}},
        "content": "print(1)",
        **changes,
    }


def test_results_are_self_contained_and_copied():
    from slick.tools._protocol import validate_results

    payload = [result()]
    original = deepcopy(payload)
    normalized = validate_results(json.loads(json.dumps(payload)))
    assert normalized[0]["is_error"] is False
    normalized[0]["request"]["arguments"]["path"] = "b.py"
    assert payload == original


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [None],
        [{"request_id": "a", "content": "x"}],
        [result(content=1)],
        [result(is_error=1)],
        [result(extra=True)],
        [result(), result()],
        [result(request={"id": "", "name": "read", "arguments": {}})],
        [result(request={"id": "a", "name": "", "arguments": {}})],
        [result(request={"id": "a", "name": "read", "arguments": {"x": float("nan")}})],
        [
            result(
                request={
                    "id": "a",
                    "name": "read",
                    "arguments": "bad",
                    "argument_error": "Invalid JSON",
                }
            )
        ],
    ],
)
def test_results_reject_invalid_payloads(payload):
    from slick.tools._protocol import validate_results

    with pytest.raises(ValueError):
        validate_results(payload)


def test_bad_arguments_can_be_answered_with_an_error():
    from slick.tools._protocol import make_request, validate_results

    request = make_request("a", "read", '{"x":1,"x":2}')
    assert request["arguments"] == '{"x":1,"x":2}'
    assert "duplicate" in request["argument_error"]
    assert validate_results([result(request=request, is_error=True)])[0]["request"] == request


@pytest.mark.parametrize("raw", ["[]", "null", '{"x":NaN}', "bad"])
def test_bad_json_is_preserved_without_inventing_arguments(raw):
    from slick.tools._protocol import make_request

    request = make_request("a", "read", raw)
    assert request["arguments"] == raw
    assert request["argument_error"]


def test_empty_batches_and_repeated_ids_across_independent_requests():
    from slick.tools._protocol import validate_results

    assert validate_results(None) == []
    assert validate_results([]) == []
    assert validate_results([result()]) == validate_results([result()])


def test_context_validation():
    from slick.tools._protocol import prepare_call

    assert prepare_call("", None, [result()])[1][0]["content"] == "print(1)"
    for context in [None, 1, ""]:
        with pytest.raises(ValueError):
            prepare_call(context, None, None)


def test_response_contract_does_not_accidentally_unpack_a_string():
    from slick.tools._protocol import validate_response

    for response in ["ok", ["ok", []], ("ok", None), (1, [])]:
        with pytest.raises(ValueError):
            validate_response(response)
    assert validate_response(("  ok\n", [])) == ("  ok\n", [])


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

    @prompt(provider=provider, cache=True, log_dir=tmp_path, max_repairs=2)
    def answer() -> str:
        """Answer the question."""

    with pytest.raises(PromptError, match="tool requests"):
        answer()
    assert provider.calls == 1
    assert not list(tmp_path.rglob("response.txt"))

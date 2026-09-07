"""The functional API: rendering, text execution, and typed results."""

import asyncio
import inspect

import pytest
from jinja2 import UndefinedError
from pydantic import BaseModel, ValidationError

import slick
from slick import prompts


class Summary(BaseModel):
    headline: str
    points: list[str]


class Backend:
    def __init__(self, response="answer"):
        self.response = response
        self.calls = []

    def call(self, text):
        self.calls.append(text)
        return self.response

    async def acall(self, text):
        await asyncio.sleep(0)
        return self.call(text)

    def identity(self):
        return {"backend": "test", "response": self.response}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "TEMPLATE_ROOT", tmp_path)
    monkeypatch.setattr(prompts, "LOG_DIR", tmp_path / "logs")
    return tmp_path


def test_render_composes_history_macros_and_includes_without_execution(root):
    (root / "style.j2").write_text("Be concise.")
    (root / "skills.j2").write_text("{% macro greet(name) %}Hello {{ name }}{% endmacro %}")
    (root / "chat.j2").write_text(
        '{% from "skills.j2" import greet %}{% include "style.j2" %}\n\n'
        "{{ greet(name) }}\n{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    )
    assert (
        slick.render("chat.j2", name="Ada", messages=[{"role": "user", "content": "Why?"}])
        == "Be concise.\nHello Ada\nuser: Why?"
    )
    (root / "style.j2").write_text("Be precise.")
    assert slick.render("chat.j2", name="Ada", messages=[]).startswith("Be precise.")
    with pytest.raises(UndefinedError):
        slick.render("chat.j2", messages=[])
    with pytest.raises(slick.PromptError, match=r"missing\.j2"):
        slick.render("missing.j2")
    assert not (root / "logs").exists()


def test_parse_is_shared_and_does_not_call_a_backend():
    assert slick.parse("  exact text\n") == "  exact text\n"
    assert slick.parse("```json\n[1, 2]\n```", list[int]) == [1, 2]
    assert slick.parse('{"headline":"H", "points":[]}', Summary).headline == "H"
    with pytest.raises(ValidationError):
        slick.parse('{"headline": []}', Summary)


def test_prompt_object_renders_lazily_with_execution_options_as_template_data(root, capsys):
    reply = slick.Prompt("reply.j2")
    (root / "reply.j2").write_text(
        "{{ question }} / {{ backend }} / {{ model }} / {{ output }} / {{ template }}"
    )
    assert (
        reply(question="Why?", backend="context", model="example", output="text", template="data")
        == "Why? / context / example / text / data"
    )
    (root / "reply.j2").write_text("Updated: {{ question }}")
    assert reply(question="How?") == "Updated: How?"
    with pytest.raises(UndefinedError):
        reply()
    assert not (root / "logs").exists()
    assert capsys.readouterr() == ("", "")


def test_backend_calls_do_not_implicitly_cache_log_or_rewrite(root, capsys):
    backend = Backend("  answer\n")

    @slick.prompt(backend=backend)
    def answer(question: str, backend: str = "context") -> str:
        """{{ question }} / {{ backend }}"""

    assert answer("why") == "  answer\n"
    assert answer("why") == "  answer\n"
    assert backend.calls == ["why / context", "why / context"]
    assert not (root / "logs").exists()
    assert capsys.readouterr() == ("", "")
    assert str(inspect.signature(answer)) == "(question: str, backend: str = 'context') -> str"


def test_modern_invalid_output_is_not_repaired_and_preserves_response(root):
    backend = Backend("invalid")

    @slick.prompt(backend=backend)
    def summarize(document: str) -> Summary:
        """{{ document }}\n{{ output_format }}"""

    with pytest.raises(slick.PromptError) as caught:
        summarize("data")
    assert caught.value.response == "invalid"
    assert isinstance(caught.value.__cause__, ValidationError)
    assert len(backend.calls) == 1
    assert backend.calls[0].count("# Output Format") == 1
    assert not (root / "logs").exists()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_cache_logs_only_accepted_typed_output(root, asynchronous):
    backend = Backend('{"headline":"H", "points":[]}')

    def declaration(document: str) -> Summary:
        """{{ document }}"""

    async def async_declaration(document: str) -> Summary:
        """{{ document }}"""

    fn = slick.prompt(backend=backend, cache=True, log_dir=root / "chosen")(
        async_declaration if asynchronous else declaration
    )
    if asynchronous:

        async def run():
            return await fn("x"), await fn("x")

        first, second = asyncio.run(run())
    else:
        first, second = fn("x"), fn("x")
    assert first == second == Summary(headline="H", points=[])
    assert len(backend.calls) == 1
    assert len(list((root / "chosen").glob("*/response.txt"))) == 1


def test_async_functions_render_and_execute_without_hidden_sync_calls(root):
    class AsyncOnly:
        async def acall(self, text):
            await asyncio.sleep(0)
            return '{"headline":"' + text.splitlines()[0] + '", "points":[]}'

    @slick.prompt(backend=AsyncOnly())
    async def summarize(document: str) -> Summary:
        """{{ document }} {{ count }}"""
        await asyncio.sleep(0)
        return {"count": len(document)}

    async def run():
        assert (await summarize.render("abc")).startswith("abc 3")
        return await asyncio.gather(summarize("abc"), summarize("xy"))

    results = asyncio.run(run())
    assert [r.headline for r in results] == ["abc 3", "xy 2"]
    assert inspect.iscoroutinefunction(summarize)
    assert not (root / "logs").exists()


def test_async_cancellation_reaches_backend(root):
    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        class Waiting:
            async def acall(self, text):
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()

        @slick.prompt(backend=Waiting())
        async def answer(question: str) -> str:
            """{{ question }}"""

        task = asyncio.create_task(answer("why"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()

    asyncio.run(run())


def test_backend_selection_is_explicit_and_unsupported_async_fails(root):
    with pytest.raises(slick.PromptError, match=r"model.*backend|backend.*model"):
        slick.prompt(model=Backend(), backend=Backend())(lambda: None)

    @slick.prompt(backend=Backend())
    def answer(question: str) -> str:
        """{{ question }}"""

    with pytest.raises(slick.PromptError, match="model"):
        answer("x", model=Backend())

    class SyncOnly:
        def call(self, text):
            raise AssertionError("sync call must not be attempted")

    @slick.prompt(backend=SyncOnly())
    async def async_answer(question: str) -> str:
        """{{ question }}"""

    with pytest.raises(slick.PromptError, match="acall"):
        asyncio.run(async_answer("x"))


def test_custom_backend_needs_no_identity_unless_persistence_is_enabled(root):
    class Minimal:
        def call(self, text):
            return text

    @slick.prompt(backend=Minimal())
    def echo(value: str) -> str:
        """{{ value }}"""

    assert echo("hello") == "hello"

    @slick.prompt(backend=Minimal(), cache=True)
    def cached(value: str) -> str:
        """{{ value }}"""

    with pytest.raises(slick.PromptError, match="identity"):
        cached("hello")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_repairs_do_not_require_disk_persistence(root, asynchronous):
    class Repairing(Backend):
        def call(self, text):
            self.calls.append(text)
            return "bad" if len(self.calls) == 1 else "[1,2]"

    backend = Repairing()

    def declaration() -> list[int]:
        """Numbers"""

    async def async_declaration() -> list[int]:
        """Numbers"""

    fn = slick.prompt(backend=backend, max_repairs=1)(
        async_declaration if asynchronous else declaration
    )
    assert (asyncio.run(fn()) if asynchronous else fn()) == [1, 2]
    assert len(backend.calls) == 2
    assert "# Repair" in backend.calls[1]
    assert not (root / "logs").exists()


def test_output_save_remains_explicit_on_modern_path(root):
    @slick.prompt(backend=Backend("answer"))
    def answer() -> str:
        """Answer"""

    assert answer(output=root / "result.txt") == "answer"
    assert (root / "result.txt").read_text() == "answer"
    assert not (root / "logs").exists()

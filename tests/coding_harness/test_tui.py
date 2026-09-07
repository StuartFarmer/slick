"""Optional Textual job; all effects occur in disposable workspaces."""

import asyncio
import hashlib
import json
import os
import sys
from dataclasses import replace

import pytest

pytest.importorskip("textual")

from textual.app import App
from textual.widgets import Input, ListView, RichLog, Static

from examples.coding_harness.tui import DetailModal, plain


def test_external_output_is_plain_and_terminal_controls_are_removed():
    output = plain(
        "[bold]literal[/bold]\x1b[31m red\x1b[0m "
        "\x1b]8;;https://evil.invalid\x07link\x1b]8;;\x07"
        "\x1b]52;c;clipboard\x07\x00\r\nend"
    )
    assert output.plain == "[bold]literal[/bold] red link\nend"
    assert not output.spans
    assert plain("hello\x1b]52;c;unterminated").plain == "hello"


def test_result_details_show_actual_output_lines_and_truncation():
    from examples.coding_harness.tui import result_detail

    detail = result_detail(
        json.dumps(
            {
                "argv": ["python", "-m", "unittest"],
                "exit_code": 1,
                "timed_out": False,
                "stdout": "first\nsecond\n",
                "stderr": "FAILED\n",
                "truncated": True,
            }
        )
    )
    assert "first\nsecond\n" in detail
    assert "FAILED\n" in detail
    assert '"exit_code": 1' in detail
    assert '"truncated": true' in detail


@pytest.mark.parametrize(
    "key,expected", [("1", "once"), ("2", "session"), ("3", "deny"), ("escape", "deny")]
)
def test_keyboard_command_decision(key, expected):
    async def scenario():
        app = App()
        decisions = []
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(
                DetailModal("Run command", "argv: ['python']\ncwd: /tmp", decision=True),
                decisions.append,
            )
            await pilot.press(key)
            await pilot.pause()
            assert decisions == [expected]

    asyncio.run(scenario())


@pytest.fixture
def harness(tmp_path):
    from examples.coding_harness.demo import DemoBackend, create_demo
    from examples.coding_harness.tui import HarnessApp

    root = tmp_path / "demo"
    root.mkdir()
    config = create_demo(root)
    return HarnessApp(DemoBackend(root), root, config)


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
def test_enter_repairs_real_demo_and_opens_result_and_diff(harness, size):
    async def scenario():
        async with harness.run_test(size=size) as pilot:
            field = harness.query_one("#task", Input)
            field.value = "Fix the total calculation"
            await pilot.press("enter")
            await asyncio.wait_for(harness.workers.wait_for_complete(), 20)
            await pilot.pause()
            result = harness.agent.state.last_result
            assert result.status == "verified"
            assert result.repairs == 1
            assert all(check.command.exit_code == 0 for check in result.checks)
            assert (harness.workspace_root / "pricing.py").read_text() == (
                "def total(values):\n    return sum(values)\n"
            )
            assert not field.disabled
            results = harness.query_one("#results", ListView)
            results.focus()
            await pilot.press("home", "enter")
            await pilot.pause()
            assert isinstance(harness.screen, DetailModal)
            await pilot.press("escape", "ctrl+d")
            await asyncio.wait_for(harness.workers.wait_for_complete(), 5)
            await pilot.pause()
            assert isinstance(harness.screen, DetailModal)
            assert "pricing.py" in harness.screen.content
            assert "pre-existing" in harness.screen.content.lower()
            await pilot.press("escape")

    asyncio.run(scenario())


def test_help_new_and_narrow_hint(harness):
    async def scenario():
        async with harness.run_test(size=(65, 20)) as pilot:
            hint = harness.query_one("#resize-hint", Static)
            assert hint.display
            task = harness.query_one("#task", Input)
            task.value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            assert "/save PATH" in "".join(
                line.text for line in harness.query_one("#transcript", RichLog).lines
            )
            source = harness.workspace_root / "pricing.py"
            before = source.read_bytes()
            task.value = "/new"
            await pilot.press("enter")
            await asyncio.wait_for(harness.workers.wait_for_complete(), 10)
            assert source.read_bytes() == before
            assert harness.agent.state.last_result is None
            assert not task.disabled

    asyncio.run(scenario())


def test_detail_modal_preserves_markup_as_literal_and_scrolls():
    async def scenario():
        app = App()
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(DetailModal("Output", "[bold]literal[/bold]\n" * 80))
            await pilot.pause()
            log = app.screen.query_one(RichLog)
            assert "[bold]literal[/bold]" in "".join(line.text for line in log.lines)
            await pilot.press("end", "escape")
            assert not isinstance(app.screen, DetailModal)

    asyncio.run(scenario())


def test_command_decision_does_not_hide_tail_arguments():
    async def scenario():
        app = App()
        async with app.run_test() as pilot:
            await app.push_screen(
                DetailModal("Command", "argument " * 2600 + "TAIL_ARGUMENT", decision=True)
            )
            await pilot.pause()
            assert "TAIL_ARGUMENT" in "".join(
                line.text for line in app.screen.query_one(RichLog).lines
            )
            await pilot.press("escape")

    asyncio.run(scenario())


class CommandBackend:
    """Scripted native turns; command execution still uses the real workspace."""

    def __init__(self, commands):
        self.commands = commands
        self.calls = 0

    def identity(self):
        return {"provider": "demo", "model": "command-test"}

    async def aturn(self, history, *, tools, instructions=None):
        from slick.turns import ModelTurn, ToolCall

        self.calls += 1
        calls = (
            [
                ToolCall(f"command-{index}", "run_command", {"argv": argv})
                for index, argv in enumerate(self.commands)
            ]
            if self.calls == 1
            else []
        )
        return ModelTurn(
            "demo",
            "command-test",
            "Run commands" if calls else "Done",
            calls,
            [],
            "tool_calls" if calls else "end_turn",
        )


async def wait_until(pilot, predicate):
    async def wait():
        while not predicate():
            await pilot.pause(0.01)

    await asyncio.wait_for(wait(), 10)


def command_app(harness, commands):
    from examples.coding_harness.state import HarnessConfig
    from examples.coding_harness.tui import HarnessApp

    return HarnessApp(CommandBackend(commands), harness.workspace_root, HarnessConfig())


@pytest.mark.parametrize("choice,count", [("1", 1), ("2", 2)])
def test_decisions_control_real_effects_and_exact_session_scope(harness, choice, count):
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; "
        "p=Path('allowed.txt'); p.write_text(p.read_text()+'x' if p.exists() else 'x')",
    ]
    denied = [sys.executable, "-c", "from pathlib import Path; Path('denied.txt').touch()"]
    app = command_app(harness, [command, command, denied])

    async def scenario():
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#task", Input).value = "Run the commands"
            await pilot.press("enter")
            await wait_until(pilot, lambda: isinstance(app.screen, DetailModal))
            assert json.dumps(command) in app.screen.content
            assert str(harness.workspace_root) in app.screen.content
            await pilot.press(choice)
            if choice == "1":
                await wait_until(pilot, lambda: isinstance(app.screen, DetailModal))
                await pilot.press("3")
            await wait_until(pilot, lambda: isinstance(app.screen, DetailModal))
            assert "denied.txt" in app.screen.content
            await pilot.press("escape")
            await asyncio.wait_for(app.workers.wait_for_complete(), 10)
            assert (harness.workspace_root / "allowed.txt").read_text() == "x" * count
            assert not (harness.workspace_root / "denied.txt").exists()
            assert not app.query_one("#task", Input).disabled

    asyncio.run(scenario())


@pytest.mark.parametrize("quit_app", [False, True])
def test_cancel_or_quit_reaps_child_and_allows_followup(harness, quit_app):
    command = [
        sys.executable,
        "-c",
        "import os,time; from pathlib import Path; "
        "Path('child.pid').write_text(str(os.getpid())); time.sleep(60)",
    ]
    app = command_app(harness, [command])

    async def scenario():
        child_pid = None
        try:
            async with app.run_test(size=(80, 24)) as pilot:
                field = app.query_one("#task", Input)
                field.value = "Start command"
                await pilot.press("enter")
                await wait_until(pilot, lambda: isinstance(app.screen, DetailModal))
                await pilot.press("1")
                pidfile = harness.workspace_root / "child.pid"
                await wait_until(pilot, pidfile.exists)
                child_pid = int(pidfile.read_text())
                assert field.disabled
                # Bypassing the disabled widget must still not enqueue a second run.
                app.post_message(Input.Submitted(field, "Overlapping task"))
                await pilot.pause()
                assert app.backend.calls == 1
                if quit_app:
                    await pilot.press("ctrl+d")
                    await wait_until(pilot, lambda: isinstance(app.screen, DetailModal))
                    assert not app.screen.decision
                await pilot.press("ctrl+q" if quit_app else "escape")
                await asyncio.wait_for(app.workers.wait_for_complete(), 10)
                with pytest.raises(ProcessLookupError):
                    os.kill(child_pid, 0)
                child_pid = None
                assert app.agent.state.last_result.status == "cancelled"
                if not quit_app:
                    assert not field.disabled
                    field.value = "Follow up"
                    await pilot.press("enter")
                    await asyncio.wait_for(app.workers.wait_for_complete(), 10)
                    assert app.agent.state.last_result.status == "unverified"
        finally:
            if child_pid is not None:
                try:
                    os.kill(child_pid, 9)
                except ProcessLookupError:
                    pass

    asyncio.run(scenario())


def test_save_and_resume_restore_inert_history(harness, tmp_path):
    from examples.coding_harness.session import load_session
    from examples.coding_harness.tui import HarnessApp

    path = tmp_path / "saved session.json"

    async def scenario():
        async with harness.run_test() as pilot:
            field = harness.query_one("#task", Input)
            field.value = "Fix the total calculation"
            await pilot.press("enter")
            await asyncio.wait_for(harness.workers.wait_for_complete(), 20)
            field.value = f"/save {path}"
            await pilot.press("enter")
            await asyncio.wait_for(harness.workers.wait_for_complete(), 10)
            assert path.is_file()
            history_size = len(harness.agent.state.history)
        saved = load_session(path)
        restored = HarnessApp(harness.backend, harness.workspace_root, harness.config, saved=saved)
        before = (harness.workspace_root / "pricing.py").read_bytes()
        async with restored.run_test() as pilot:
            assert len(restored.agent.state.history) == history_size
            assert restored.agent.state.last_result.status == "verified"
            assert not restored.query_one("#task", Input).disabled
            assert (harness.workspace_root / "pricing.py").read_bytes() == before

    asyncio.run(scenario())


def test_backend_failure_exposes_reason_and_returns_to_idle(harness):
    class FailingBackend(CommandBackend):
        async def aturn(self, history, *, tools, instructions=None):
            raise RuntimeError("Backend unavailable: local test failure")

    app = command_app(harness, [])
    app.backend = FailingBackend([])

    async def scenario():
        async with app.run_test() as pilot:
            field = app.query_one("#task", Input)
            field.value = "Try the backend"
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), 10)
            await pilot.pause()
            assert app.agent.state.last_result.status == "failed"
            assert "Backend unavailable" in "".join(
                line.text for line in app.query_one("#transcript", RichLog).lines
            )
            assert not field.disabled

    asyncio.run(scenario())


def test_compact_command_replaces_history_without_changing_files(harness):
    from slick.turns import ModelTurn

    class SummaryBackend(CommandBackend):
        async def aturn(self, history, *, tools, instructions=None):
            if not tools:
                text = json.dumps(
                    {
                        "facts": ["No files changed"],
                        "decisions": [],
                        "open_questions": [],
                        "modified_files": [],
                        "next_steps": [],
                    }
                )
                return ModelTurn("demo", "command-test", text, [], [], "end_turn")
            return await super().aturn(history, tools=tools, instructions=instructions)

    app = command_app(harness, [])
    app.backend = SummaryBackend([])

    async def scenario():
        async with app.run_test() as pilot:
            field = app.query_one("#task", Input)
            field.value = "Inspect the task"
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), 10)
            history = list(app.agent.state.history)
            before = (app.workspace_root / "pricing.py").read_bytes()
            field.value = "/compact"
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), 10)
            assert app.agent.state.archived_histories, "".join(
                line.text for line in app.query_one("#transcript", RichLog).lines
            )
            assert app.agent.state.archived_histories[-1] == history
            assert len(app.agent.state.history) == 1
            assert (app.workspace_root / "pricing.py").read_bytes() == before
            assert not field.disabled

    asyncio.run(scenario())


def test_reported_token_counts_are_visible(harness):
    class UsageBackend(CommandBackend):
        async def aturn(self, history, *, tools, instructions=None):
            turn = await super().aturn(history, tools=tools, instructions=instructions)
            return replace(turn, input_tokens=12, output_tokens=3)

    app = command_app(harness, [])
    app.backend = UsageBackend([])

    async def scenario():
        async with app.run_test() as pilot:
            app.query_one("#task", Input).value = "Return reported usage"
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), 10)
            await pilot.pause()
            output = "".join(line.text for line in app.query_one("#transcript", RichLog).lines)
            assert '"input_tokens": 12' in output
            assert '"output_tokens": 3' in output

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["verified", "unverified"])
def test_completion_reports_real_verification_file_edits(harness, status):
    from examples.coding_harness.state import HarnessConfig
    from examples.coding_harness.tui import HarnessApp
    from slick.turns import ModelTurn, ToolCall

    source = harness.workspace_root / "pricing.py"
    source.write_text("def total(values):\n    return sum(values)\n")
    tests = harness.workspace_root / "test_pricing.py"
    digest = hashlib.sha256(tests.read_bytes()).hexdigest()

    class EditTestsBackend(CommandBackend):
        async def aturn(self, history, *, tools, instructions=None):
            self.calls += 1
            calls = []
            if self.calls == 1:
                calls = [
                    ToolCall(
                        "edit-test",
                        "edit_file",
                        {
                            "path": "test_pricing.py",
                            "old": "import unittest",
                            "new": "import unittest\n# Reviewed by the harness",
                            "expected_sha256": digest,
                        },
                    )
                ]
            return ModelTurn(
                "demo",
                "command-test",
                "Reviewed the tests",
                calls,
                [],
                "tool_calls" if calls else "end_turn",
            )

    config = harness.config if status == "verified" else HarnessConfig()
    app = HarnessApp(EditTestsBackend([]), harness.workspace_root, config)

    async def scenario():
        async with app.run_test() as pilot:
            app.query_one("#task", Input).value = "Review the verification file"
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), 10)
            await pilot.pause()
            assert "# Reviewed by the harness" in tests.read_text()
            assert app.agent.state.last_result.status == status
            output = "".join(line.text for line in app.query_one("#transcript", RichLog).lines)
            assert "Verification-related files changed: test_pricing.py" in output

    asyncio.run(scenario())


def test_failed_startup_quits_with_nonzero_result(tmp_path):
    from examples.coding_harness.state import HarnessConfig
    from examples.coding_harness.tui import HarnessApp

    app = HarnessApp(CommandBackend([]), tmp_path / "missing-workspace", HarnessConfig())

    async def scenario():
        async with app.run_test() as pilot:
            assert app.agent is None
            assert app.query_one("#task", Input).disabled
            output = "".join(line.text for line in app.query_one("#transcript", RichLog).lines)
            assert "Failed to initialize" in output
            await pilot.press("ctrl+q")
        assert app.return_value == 1

    asyncio.run(scenario())

"""Behavioral coverage for the direct console and Textual interfaces."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("textual")
from textual.app import App
from textual.widgets import Input, RichLog, Static

from examples.coding_harness.tui import ApprovalModal, HarnessApp
from examples.coding_harness.ui import ConsoleUI, safe_text


def transcript(app):
    return "\n".join(line.text for line in app.query_one("#transcript", RichLog).lines)


async def wait_until(pilot, predicate):
    async def wait():
        while not predicate():
            await pilot.pause(0.01)

    await asyncio.wait_for(wait(), 5)


class FakeWorkspace:
    def __init__(self, root):
        self.root = Path(root)
        self.decide = self._deny

    async def _deny(self, request):
        return "deny"

    async def git_diff(self):
        return {
            "staged": "",
            "unstaged": "diff --git a/example.py b/example.py\n+[bold]literal[/bold]\x1b[31m",
            "untracked": ["notes.txt"],
            "truncated": False,
        }


class FakeAgent:
    def __init__(self, root):
        self.provider = type(
            "FakeProvider",
            (),
            {"identity": lambda _: {"provider": "demo", "model": "offline"}},
        )()
        self.workspace = FakeWorkspace(root)
        self.config = object()
        self.messages = [{"role": "assistant", "text": "Restored context"}]
        self.last_result = None
        self.running = False
        self.ui = ConsoleUI(write=lambda text: None)
        self.compactions = 0
        self.resets = 0
        self.started = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def run(self, task):
        self.running = True
        self.ui.user(task)
        try:
            if task == "slow":
                self.started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    self.last_result = {
                        "status": "cancelled",
                        "answer": "Cancelled",
                        "checks": [],
                        "turns": 0,
                        "tool_calls": 0,
                        "repairs": 0,
                        "changed_paths": [],
                    }
                    self.ui.completed(self.last_result)
                    raise
                finally:
                    await asyncio.sleep(0)
                    self.cleaned.set()
                return None
            if task.startswith("approve "):
                name = task.removeprefix("approve ")
                decision = await self.workspace.decide(
                    {
                        "argv": ["python", "-c", f"Path({name!r}).touch()"],
                        "cwd": str(self.workspace.root),
                        "timeout": 17,
                    }
                )
                if decision in {"once", "session"}:
                    (self.workspace.root / name).touch()
            self.ui.tool("read_file", "[bold]literal[/bold]\x1b]52;c;clipboard\x07")
            self.last_result = {
                "status": "unverified",
                "answer": f"Finished {task}",
                "checks": [],
                "turns": 1,
                "tool_calls": 1,
                "repairs": 0,
                "changed_paths": [],
            }
            self.ui.completed(self.last_result)
            return self.last_result
        finally:
            self.running = False

    async def compact(self):
        self.compactions += 1

    async def new_conversation(self):
        self.resets += 1
        self.messages.clear()
        self.last_result = None

    def save(self, path):
        Path(path).write_text(json.dumps({"messages": self.messages}))


def test_console_ui_is_safe_readable_and_denies_commands():
    lines = []
    ui = ConsoleUI(write=lines.append)
    ui.user("fix [total]\x1b[31m")
    ui.assistant("done")
    ui.status("checking")
    ui.tool("run_command", "line one\nline two", is_error=True)
    ui.checks(
        [
            {
                "name": "tests",
                "command": {"exit_code": 1, "timed_out": False, "truncated": True},
            }
        ],
        baseline=True,
    )
    ui.completed(
        {
            "status": "blocked",
            "answer": "needs work",
            "turns": 2,
            "tool_calls": 1,
            "repairs": 1,
        }
    )

    assert lines == [
        "You: fix [total]",
        "Agent: done",
        "Status: checking",
        "Tool error (run_command): line one\nline two",
        "Baseline tests: exit 1\n[output truncated]",
        "Blocked · 2 turns · 1 tools · 1 repairs\nneeds work",
    ]
    assert (
        safe_text(
            "link\x1b]8;;https://evil.invalid\x07text\x1b]8;;\x07"
            "\x1bXDCS\x1b\\\x90C1 DCS\x9c\x00"
        )
        == "linktext"
    )
    assert asyncio.run(ui.approve({"argv": ["python"], "cwd": "/tmp", "timeout": 5})) == "deny"


@pytest.mark.parametrize(
    ("key", "expected"),
    [("1", "once"), ("2", "session"), ("3", "deny"), ("escape", "deny")],
)
def test_approval_modal_returns_keyboard_choice_and_shows_exact_request(key, expected):
    request = {
        "argv": ["python", "-c", "print('[literal]')", "TAIL_ARGUMENT"],
        "cwd": "/tmp/work space",
        "timeout": 23,
    }

    async def scenario():
        app = App()
        decisions = []
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(ApprovalModal(request), decisions.append)
            await pilot.pause()
            screen = app.screen
            output = "\n".join(line.text for line in screen.query_one(RichLog).lines)
            assert json.dumps(request["argv"], ensure_ascii=False) in output
            assert request["cwd"] in output
            assert "timeout: 23s" in output
            await pilot.press(key)
            await pilot.pause()
        assert decisions == [expected]

    asyncio.run(scenario())


def test_mount_wires_direct_ui_and_approval_controls_real_effects(tmp_path):
    agent = FakeAgent(tmp_path)
    app = HarnessApp(agent)

    async def scenario():
        async with app.run_test(size=(80, 24)) as pilot:
            assert agent.ui is app
            assert agent.workspace.decide.__self__ is app
            assert "Restored context" in transcript(app)
            field = app.query_one("#task", Input)

            field.value = "approve allowed.txt"
            await pilot.press("enter")
            await wait_until(pilot, lambda: isinstance(app.screen, ApprovalModal))
            assert "allowed.txt" in app.screen.content
            await pilot.press("1")
            await wait_until(pilot, lambda: app.operation is None)
            assert (tmp_path / "allowed.txt").exists()

            field.value = "approve denied.txt"
            await pilot.press("enter")
            await wait_until(pilot, lambda: isinstance(app.screen, ApprovalModal))
            await pilot.press("escape")
            await wait_until(pilot, lambda: app.operation is None)
            assert not (tmp_path / "denied.txt").exists()
            output = transcript(app)
            assert "[bold]literal[/bold]" in output
            assert "clipboard" not in output
            assert not field.disabled

    asyncio.run(scenario())


def test_cancel_waits_for_cleanup_and_allows_followup(tmp_path):
    agent = FakeAgent(tmp_path)
    app = HarnessApp(agent)

    async def scenario():
        async with app.run_test() as pilot:
            field = app.query_one("#task", Input)
            field.value = "slow"
            await pilot.press("enter")
            await wait_until(pilot, agent.started.is_set)
            await pilot.press("escape")
            await wait_until(pilot, lambda: app.operation is None)
            assert agent.cleaned.is_set()
            assert not field.disabled

            field.value = "follow up"
            await pilot.press("enter")
            await wait_until(pilot, lambda: app.operation is None)
            assert agent.last_result["answer"] == "Finished follow up"

    asyncio.run(scenario())


def test_unmount_cancels_active_task_without_rendering_into_removed_widgets(tmp_path):
    agent = FakeAgent(tmp_path)
    app = HarnessApp(agent)

    async def scenario():
        async with app.run_test() as pilot:
            field = app.query_one("#task", Input)
            field.value = "slow"
            await pilot.press("enter")
            await wait_until(pilot, agent.started.is_set)
        assert agent.cleaned.is_set()
        assert app.operation is None

    asyncio.run(scenario())


def test_commands_and_diff_use_agent_methods(tmp_path):
    agent = FakeAgent(tmp_path)
    app = HarnessApp(agent)
    saved = tmp_path / "saved session.json"

    async def submit(pilot, text):
        app.query_one("#task", Input).value = text
        await pilot.press("enter")
        await wait_until(pilot, lambda: app.operation is None)

    async def scenario():
        async with app.run_test() as pilot:
            assert "demo / offline" in app.query_one("#heading", Static).render().plain
            assert str(tmp_path) in app.query_one("#heading", Static).render().plain
            await submit(pilot, "/help")
            assert "/save PATH" in transcript(app)
            await submit(pilot, "/compact")
            assert app.query_one("#status", Static).render().plain == "Ready"
            await submit(pilot, f"/save {saved}")
            assert agent.compactions == 1
            assert saved.is_file()
            assert app.query_one("#status", Static).render().plain == "Ready"

            await pilot.press("ctrl+d")
            await wait_until(pilot, lambda: "notes.txt" in transcript(app))
            output = transcript(app)
            assert "diff --git" in output
            assert "[bold]literal[/bold]" in output

            await submit(pilot, "/new")
            assert agent.resets == 1
            assert "New conversation" in transcript(app)
            assert app.query_one("#status", Static).render().plain == "Ready"

    asyncio.run(scenario())


def test_real_offline_demo_repairs_and_verifies_through_tui(tmp_path):
    from examples.coding_harness.agent import create_agent
    from examples.coding_harness.demo import FIXED, DemoProvider, create_demo

    root = tmp_path / "demo"
    root.mkdir()
    config = create_demo(root)

    async def scenario():
        agent = await create_agent(DemoProvider(root), root, config)
        app = HarnessApp(agent)
        async with app.run_test(size=(80, 24)) as pilot:
            field = app.query_one("#task", Input)
            field.value = "Fix the total calculation"
            await pilot.press("enter")
            await wait_until(pilot, lambda: app.operation is None)
            assert agent.last_result["status"] == "verified"
            assert agent.last_result["repairs"] == 1
            assert (root / "pricing.py").read_text() == FIXED
            assert "Verified" in transcript(app)

            await pilot.press("ctrl+d")
            await wait_until(pilot, lambda: "diff --git" in transcript(app))
            assert "pricing.py" in transcript(app)

    asyncio.run(scenario())

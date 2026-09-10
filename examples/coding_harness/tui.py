"""Optional Textual frontend for a pre-initialized coding agent."""

import asyncio
import json
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, RichLog, Static

from .ui import check_text, completion_text, content_text, safe_text

INLINE_LIMIT = 4000


def plain(value):
    return Text(safe_text(value))


class ApprovalModal(ModalScreen[str]):
    """Show an entire command request and require an explicit decision."""

    BINDINGS: ClassVar = [
        ("escape", "choose('deny')", "Deny"),
        ("1", "choose('once')", "Allow once"),
        ("2", "choose('session')", "Allow for session"),
        ("3", "choose('deny')", "Deny"),
    ]
    DEFAULT_CSS = """
    ApprovalModal { align: center middle; background: $background 80%; }
    ApprovalModal > Vertical { width: 95%; height: 90%; border: solid $accent; padding: 1; }
    ApprovalModal RichLog { height: 1fr; }
    ApprovalModal Button { width: 100%; height: 3; }
    """

    def __init__(self, request):
        super().__init__()
        self.request = request
        self.content = (
            f"argv: {json.dumps(request['argv'], ensure_ascii=False)}\n"
            f"cwd: {request['cwd']}\n"
            f"timeout: {request['timeout']}s"
        )

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Run command?")
            yield RichLog(wrap=True, markup=False, highlight=False)
            yield Button("1  Allow once", id="once")
            yield Button("2  Allow for session", id="session")
            yield Button("3  Deny", id="deny")

    def on_mount(self):
        self.query_one(RichLog).write(plain(self.content))
        self.query_one("#deny", Button).focus()

    def action_choose(self, choice):
        self.dismiss(choice)

    def on_button_pressed(self, event: Button.Pressed):
        self.action_choose(event.button.id)


class HarnessApp(App):
    """A thin view over a CodingAgent; all execution remains in the agent."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+d", "diff", "Diff", priority=True),
        Binding("ctrl+q", "quit_cleanly", "Quit", priority=True),
        Binding("ctrl+c", "quit_cleanly", "Quit", show=False, priority=True),
    ]
    CSS = """
    Screen { layout: vertical; }
    #heading { height: 2; text-style: bold; }
    #transcript { height: 1fr; min-height: 3; }
    #status { height: 1; color: $text-muted; }
    #task { height: 3; }
    """

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.operation = None
        self._closing = False

    def compose(self) -> ComposeResult:
        identity = self.agent.provider.identity()
        yield Static(
            plain(
                f"Slick coding harness · {identity['provider']} / {identity['model']}\n"
                f"{self.agent.workspace.root}"
            ),
            id="heading",
        )
        yield RichLog(
            id="transcript", wrap=True, markup=False, highlight=False, max_lines=2000
        )
        yield Static("Ready", id="status")
        yield Input(placeholder="Enter a task or /help", id="task")
        yield Footer()

    def on_mount(self):
        for message in self.agent.messages:
            role = message["role"]
            if role == "user":
                self.user(message["text"])
            elif role == "assistant":
                self.assistant(message["text"])
            else:
                self.tool("tool", message["text"], message.get("error", False))
        self.agent.ui = self
        self.agent.workspace.decide = self.approve
        field = self.query_one("#task", Input)
        field.disabled = self.agent.running
        if not field.disabled:
            field.focus()

    def write(self, text):
        if not self._closing:
            self.query_one("#transcript", RichLog).write(plain(text))

    def user(self, text):
        self.write(f"You: {text}")

    def assistant(self, text):
        self.write(f"Agent: {text}")

    def status(self, text):
        if not self._closing:
            self.query_one("#status", Static).update(plain(text))

    def tool(self, name, content, is_error=False):
        output = content_text(content)
        if len(output) > INLINE_LIMIT:
            output = output[:INLINE_LIMIT] + "\n[display truncated]"
        label = "Tool error" if is_error else "Tool"
        self.write(f"{label} ({name}):\n{output}")

    def checks(self, results, baseline=False):
        for check in results:
            self.write(check_text(check, baseline))

    def completed(self, result):
        self.status(result["status"].capitalize())
        self.write(completion_text(result))

    async def approve(self, request):
        modal = ApprovalModal(request)
        loop = asyncio.get_running_loop()
        answer = loop.create_future()

        def decide(choice):
            if not answer.done():
                answer.set_result(choice or "deny")

        try:
            await self.push_screen(modal, decide)
            return await answer
        finally:
            if self.screen is modal:
                modal.dismiss("deny")

    def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if not text:
            return
        if self.operation is not None:
            self.write("A task is already running; press Escape to cancel it.")
            return
        event.input.value = ""
        if text == "/help":
            self.write(
                "Enter: send task · Esc: cancel · Ctrl+D: diff · Ctrl+Q: quit\n"
                "/help · /new · /compact · /save PATH"
            )
            return
        if text.startswith("/") and text not in {"/new", "/compact"} and not text.startswith(
            "/save "
        ):
            self.write("Unknown command or missing path. Use /help.")
            return
        event.input.disabled = True
        self.status("Working")
        self.operation = asyncio.create_task(self._run(text))

    async def _run(self, text):
        try:
            if text == "/new":
                await self.agent.new_conversation()
                if not self._closing:
                    self.query_one("#transcript", RichLog).clear()
                self.write("New conversation; workspace files preserved.")
                self.status("Ready")
            elif text == "/compact":
                await self.agent.compact()
                self.write("Conversation compacted.")
                self.status("Ready")
            elif text.startswith("/save "):
                path = Path(text[6:].strip()).expanduser()
                self.agent.save(path)
                self.write(f"Session saved: {path}")
                self.status("Ready")
            else:
                await self.agent.run(text)
        except asyncio.CancelledError:
            self.status("Cancelled")
            self.write("Cancelled; active operation cleanup finished.")
        except Exception as error:
            self.status("Failed")
            self.write(f"Failed: {type(error).__name__}: {error}")
        finally:
            self.operation = None
            if not self._closing:
                field = self.query_one("#task", Input)
                field.disabled = False
                field.focus()

    async def action_cancel(self):
        task = self.operation
        if task is not None and not task.done():
            self.status("Cancelling; waiting for cleanup")
            task.cancel()
            await task

    async def action_diff(self):
        if isinstance(self.screen, ApprovalModal):
            return
        try:
            diff = await self.agent.workspace.git_diff()
            sections = ["Workspace diff (may include pre-existing changes):"]
            for name in ("staged", "unstaged"):
                if diff.get(name):
                    sections.append(f"{name.capitalize()}:\n{diff[name]}")
            if diff.get("untracked"):
                sections.append("Untracked:\n" + "\n".join(diff["untracked"]))
            if diff.get("truncated"):
                sections.append("[diff truncated]")
            if len(sections) == 1:
                sections.append("No changes.")
            self.write("\n\n".join(sections))
        except Exception as error:
            self.write(f"Diff failed: {type(error).__name__}: {error}")

    async def action_quit_cleanly(self):
        await self.action_cancel()
        self.exit(0)

    async def on_unmount(self):
        self._closing = True
        task = self.operation
        if task is not None and not task.done():
            task.cancel()
            await task

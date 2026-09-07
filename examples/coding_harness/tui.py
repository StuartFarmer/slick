"""Optional, keyboard-accessible terminal frontend for the coding example."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, ListItem, ListView, RichLog, Static

if TYPE_CHECKING:
    from .state import CommandDecision, CommandRequest, HarnessConfig, HarnessEvent


def plain(value: object) -> Text:
    """Make untrusted observations inert, including OSC clipboard/link escapes."""
    text = str(value)
    text = re.sub(r"(?:\x1b\]|\x9d).*?(?:\x07|\x1b\\|\x9c|$)", "", text, flags=re.S)
    text = re.sub(r"\x1b[P^_].*?(?:\x1b\\|$)", "", text, flags=re.S)
    text = re.sub(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"\x1b[ -/]*[@-~]", "", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", text)
    return Text(text)


def result_detail(content: str) -> str:
    """Keep observation metadata, with actual lines for output, file text and diffs."""
    try:
        data = json.loads(content)
    except ValueError:
        return content
    if not isinstance(data, dict):
        return content
    sections = []
    for name in ("text", "diff", "stdout", "stderr"):
        if isinstance(data.get(name), str):
            sections.append(f"{name}:\n{data.pop(name)}")
    return "\n\n".join([json.dumps(data, ensure_ascii=False, indent=2), *sections])


class DetailModal(ModalScreen):
    """Scrollable observations and exact command decisions share one screen."""

    BINDINGS: ClassVar = [
        ("escape", "close", "Close / Deny"),
        ("1", "choose('once')", "Allow once"),
        ("2", "choose('session')", "Allow for session"),
        ("3", "choose('deny')", "Deny"),
    ]
    DEFAULT_CSS = """
    DetailModal { align: center middle; background: $background 80%; }
    DetailModal > Vertical { width: 95%; height: 90%; border: solid $accent; padding: 1; }
    DetailModal RichLog { height: 1fr; }
    DetailModal Button { width: 100%; min-height: 1; height: 3; }
    """

    def __init__(self, title: str, content: str, *, decision: bool = False):
        super().__init__()
        self.heading = title
        self.content = content
        self.decision = decision

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(plain(self.heading))
            yield RichLog(wrap=True, markup=False, highlight=False)
            if self.decision:
                yield Button("1  Allow once", id="once")
                yield Button("2  Allow this command for session", id="session")
                yield Button("3  Deny", id="deny")
            else:
                yield Button("Close (Esc)", id="close")

    def on_mount(self) -> None:
        output = self.content if self.decision else self.content[:20000]
        if not self.decision and len(self.content) > 20000:
            output += "\n[display truncated]"
        log = self.query_one(RichLog)
        log.write(plain(output))
        if self.decision:
            self.query_one("#deny", Button).focus()
        else:
            log.focus()

    def action_close(self) -> None:
        self.dismiss("deny" if self.decision else None)

    def action_choose(self, choice: str) -> None:
        if self.decision:
            self.dismiss(choice)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.action_close()
        else:
            self.action_choose(event.button.id)


class AgentEvent(Message):
    def __init__(self, event: HarnessEvent):
        super().__init__()
        self.event = event


class ResultItem(ListItem):
    def __init__(self, title: str, detail: str):
        super().__init__(Label(plain(title)))
        self.heading = title
        self.detail = detail


class HarnessApp(App):
    """UI events display observations; the agent alone advances execution."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+d", "diff", "Diff", priority=True),
        Binding("ctrl+q", "quit_cleanly", "Quit", priority=True),
        Binding("ctrl+c", "quit_cleanly", "Quit", show=False, priority=True),
    ]
    CSS = """
    Screen { layout: vertical; }
    #heading { height: auto; max-height: 2; text-style: bold; }
    #resize-hint { height: 1; color: $warning; }
    #transcript { height: 1fr; min-height: 3; }
    #results { height: 5; border-top: solid $primary; }
    #results Label { padding: 0 1; }
    #status { height: 1; }
    #task { height: 3; }
    """

    def __init__(self, backend, workspace_root: Path, config: HarnessConfig, *, saved=None):
        super().__init__()
        self.backend = backend
        self.workspace_root = Path(workspace_root).resolve()
        self.config = config
        self.saved = saved
        self.agent = None
        self.busy = False
        self._operation_task = None
        self._started = 0.0
        self._status = "Initializing workspace"

    def compose(self) -> ComposeResult:
        identity = self.backend.identity()
        yield Static(
            plain(
                f"Slick coding example · {identity['provider']} / {identity['model']}\n"
                f"{self.workspace_root}"
            ),
            id="heading",
        )
        yield Static("Resize terminal to at least 80 columns x 24 rows", id="resize-hint")
        yield RichLog(id="transcript", wrap=True, markup=False, highlight=False, max_lines=2000)
        yield ListView(id="results")
        yield Static("Initializing workspace", id="status")
        yield Input(placeholder="Enter a task or /help", id="task", disabled=True)
        yield Footer()

    async def on_mount(self) -> None:
        from .__main__ import create_agent

        self.set_interval(0.25, lambda: self._refresh_status() if self.busy else None)
        self.on_resize()
        try:
            self.agent = await create_agent(
                self.backend,
                self.workspace_root,
                self.config,
                decide=self.request_command,
                emit=lambda event: self.post_message(AgentEvent(event)),
                saved=self.saved,
            )
        except Exception as error:
            self._set_status(f"Failed to initialize: {type(error).__name__}: {error}")
            self._write(self._status)
            return
        self._write(f"Workspace: {self.workspace_root}. Existing changes are preserved.")
        if self.saved is not None:
            self._write("Restored session history; no saved actions were executed.")
        self.query_one("#task", Input).disabled = False
        self.query_one("#task", Input).focus()
        self._set_status("Ready")

    def on_resize(self) -> None:
        if self.is_mounted:
            self.query_one("#resize-hint", Static).display = (
                self.size.width < 80 or self.size.height < 24
            )

    def _write(self, text: str) -> None:
        self.query_one("#transcript", RichLog).write(plain(text))

    def _set_status(self, text: str) -> None:
        self._status = text
        self._refresh_status()

    def _refresh_status(self) -> None:
        elapsed = f" · {time.monotonic() - self._started:.0f}s" if self.busy else ""
        self.query_one("#status", Static).update(plain(self._status + elapsed))

    async def _add_result(self, title: str, detail: str) -> None:
        results = self.query_one("#results", ListView)
        await results.append(ResultItem(title, detail))
        if results.index is None:
            results.index = 0

    async def on_agent_event(self, message: AgentEvent) -> None:
        event = message.event
        data = event.data
        if event.kind in {"user", "assistant"}:
            self._write(f"{'You' if event.kind == 'user' else 'Agent'}: {data['text']}")
            if data.get("usage"):
                self._write(f"Reported tokens: {json.dumps(data['usage'])}")
        elif event.kind == "status":
            self._set_status(data["text"])
        elif event.kind == "tool_started":
            self._set_status(f"Running {data['name']}")
            self._write(f"Started: {data['name']}")
        elif event.kind == "tool_finished":
            title = f"{'Error' if data['is_error'] else 'Done'}: {data['name']}"
            self._write(title)
            await self._add_result(title, result_detail(data["content"]))
        elif event.kind == "verification":
            phase = "Baseline" if data["baseline"] else "Verification"
            for check in data["results"]:
                command = check["command"]
                state = "timed out" if command["timed_out"] else f"exit {command['exit_code']}"
                title = f"{phase}: {check['name']} · {state}"
                self._write(title)
                await self._add_result(
                    title, result_detail(json.dumps(command, ensure_ascii=False))
                )
            if not data["stable"]:
                self._write(f"{phase}: workspace changed during checks; results are stale.")
        elif event.kind == "compacted":
            self._write(f"Compacted context: {data['before']} → {data['after']} characters")
        elif event.kind == "completed":
            result = data["result"]
            self._set_status(
                f"{result['status'].capitalize()} · {result['turns']} model turns · "
                f"{result['tool_calls']} tools · {result['repairs']} repairs"
            )
            self._write(self._status)
            self._write(result["answer"])

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ResultItem):
            await self.push_screen(DetailModal(item.heading, item.detail))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if self.busy or self.agent is None:
            self._write("A task is already running; cancel it before submitting another.")
            return
        event.input.value = ""
        if text == "/help":
            self._write(
                "Enter: send task · Esc: cancel · Ctrl+D: diff · Ctrl+Q: quit\n"
                "/help · /new · /compact · /save PATH\n"
                "Tab to results, then arrows and Enter to inspect output."
            )
        elif (
            text.startswith("/")
            and text not in {"/new", "/compact"}
            and not text.startswith("/save ")
        ):
            self._write("Unknown command or missing path. Use /help.")
        else:
            self.busy = True
            self._started = time.monotonic()
            event.input.disabled = True
            self._set_status("Working")
            self.run_worker(self._run(text), name="agent-run", exit_on_error=False)

    async def _run(self, text: str) -> None:
        async def operation():
            if text == "/new":
                await self.agent.new_conversation()
                self.query_one("#transcript", RichLog).clear()
                await self.query_one("#results", ListView).clear()
                self._write("New conversation; fresh workspace baseline captured. Files preserved.")
                self._set_status("Ready")
            elif text == "/compact":
                await self.agent.compact()
                self._set_status("Ready")
            elif text.startswith("/save "):
                from .session import save_session

                path = Path(text[len("/save ") :].strip()).expanduser()
                save_session(path, self.agent)
                self._write(f"Session saved: {path}")
                self._set_status("Ready")
            else:
                await self.agent.run(text)

        self._operation_task = asyncio.create_task(operation())
        try:
            await self._operation_task
        except asyncio.CancelledError:
            self._set_status("Cancelled")
            self._write("Cancelled; active operation cleanup finished.")
        except Exception as error:
            self._set_status(f"Failed: {type(error).__name__}: {error}")
            self._write(self._status)
        finally:
            self._operation_task = None
            self.busy = False
            self._refresh_status()
            field = self.query_one("#task", Input)
            field.disabled = False
            field.focus()

    async def request_command(self, request: CommandRequest) -> CommandDecision:
        future = asyncio.get_running_loop().create_future()
        content = (
            f"argv: {json.dumps(request.argv, ensure_ascii=False)}\n"
            f"cwd: {request.cwd}\ntimeout: {request.timeout}s\n\n"
            "This command runs with your permissions, including network access."
        )
        modal = DetailModal("Command decision", content, decision=True)

        def decided(value):
            if not future.done():
                future.set_result(value or "deny")

        await self.push_screen(modal, decided)
        try:
            return await future
        finally:
            if self.screen is modal:
                modal.dismiss("deny")

    async def action_cancel(self) -> None:
        if self._operation_task is not None and not self._operation_task.done():
            self._set_status("Cancelling; waiting for cleanup")
            self._operation_task.cancel()
            try:
                await self._operation_task
            except asyncio.CancelledError:
                pass

    def action_diff(self) -> None:
        if self.agent is None or isinstance(self.screen, DetailModal):
            return
        self.run_worker(self._show_diff(), name="diff", exit_on_error=False)

    async def _show_diff(self) -> None:
        try:
            diff = await self.agent.workspace.git_diff()
            text = (
                "Current workspace diff; may include pre-existing user changes.\n\n"
                f"Staged:\n{diff.staged}\nUnstaged:\n{diff.unstaged}\n"
                f"Untracked:\n{chr(10).join(diff.untracked)}"
            )
            if diff.truncated:
                text += "\n[diff truncated]"
            await self.push_screen(DetailModal("Workspace diff (read-only)", text))
        except Exception as error:
            self._write(f"Diff failed: {type(error).__name__}: {error}")

    async def action_quit_cleanly(self) -> None:
        await self.action_cancel()
        self.exit(1 if self.agent is None else 0)

    async def on_unmount(self) -> None:
        await self.action_cancel()

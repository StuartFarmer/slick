"""A coding harness expressed as ordinary Python state and a bounded loop."""

import asyncio
import json
from pathlib import Path

from slick import Prompt, ToolError, parse
from slick.backends import BackendError
from slick.tools import prepare_tools
from slick.turns import ToolResult, UserMessage, validate_history

from .context import context_size, render_instructions, summary_prompt
from .state import ContextSummary, HarnessEvent, RunResult, SessionState
from .verification import render_verification, verify


class RunStopped(RuntimeError):
    """A configured budget or lack of progress ended this task."""


class CodingAgent:
    def __init__(self, backend, workspace, config, *, emit=lambda event: None):
        self.backend, self.workspace, self.config = backend, workspace, config
        self.emit = emit
        identity = backend.identity()
        self.state = SessionState(
            provider=identity["provider"],
            model=identity["model"],
            root=str(workspace.root),
            head=workspace.head,
        )
        self.tools = prepare_tools(
            [
                workspace.list_files,
                workspace.search,
                workspace.read_file,
                workspace.edit_file,
                workspace.create_file,
                workspace.run_command,
                workspace.git_diff,
            ]
        )
        self._compacted_last = False
        render_instructions(workspace, config)

    def _event(self, kind, **data):
        self.emit(HarnessEvent(kind, data))

    def _history_valid(self, history):
        validate_history(history, provider=self.state.provider, model=self.state.model)

    def _result(self, status, answer, checks=(), changed_paths=None):
        paths = (
            changed_paths
            if changed_paths is not None
            else sorted({entry["path"] for entry in self.workspace.edit_ledger})
        )
        verification_files = [
            path
            for path in paths
            if any(part in {"tests", "test", ".github"} for part in Path(path).parts)
            or Path(path).name.startswith(("test_", "conftest"))
            or Path(path).name in {"pyproject.toml", "pytest.ini", "tox.ini", "Makefile"}
            or any(
                self.workspace.root / path == self.workspace.root / argument
                for check in self.config.checks
                for argument in check.argv
            )
        ]
        if verification_files:
            answer += "\nVerification-related files changed: " + ", ".join(verification_files)
        return RunResult(
            status=status,
            answer=answer,
            checks=list(checks),
            turns=self.state.turns,
            tool_calls=self.state.tool_calls,
            repairs=self.state.repairs,
            changed_paths=paths,
        )

    def _request_budget(self):
        if self.state.turns >= self.config.limits.max_turns:
            raise RunStopped("Model request budget exhausted")
        self.state.turns += 1

    async def run(self, task: str) -> RunResult:
        if self.state.running:
            raise RuntimeError("A run is already active")
        if not task.strip():
            raise ValueError("Task must not be empty")
        self.state.running = True
        self.state.task = task
        self.state.turns = self.state.tool_calls = self.state.repairs = 0
        self.state.last_result = None
        self._compacted_last = False
        try:
            self.state.last_result = await asyncio.wait_for(
                self._run(task),
                timeout=self.config.limits.task_timeout,
            )
        except asyncio.CancelledError:
            self.state.last_result = self._result(
                "cancelled", "Cancelled; inspect the diff before continuing."
            )
            raise
        except asyncio.TimeoutError:
            self.state.last_result = self._result("blocked", "Task deadline exceeded")
        except RunStopped as error:
            self.state.last_result = self._result("blocked", str(error))
        except Exception as error:
            self.state.last_result = self._result("failed", f"{type(error).__name__}: {error}")
        finally:
            self.state.running = False
            self.state.edit_ledger = list(self.workspace.edit_ledger)
            if self.state.last_result is not None:
                self._event("completed", result=self.state.last_result.model_dump())
        return self.state.last_result

    async def _checks(self, *, baseline=False):
        self._event("status", text="Checking baseline" if baseline else "Verifying changes")
        result = await verify(self.workspace, self.config.checks)
        self._event(
            "verification",
            results=[item.model_dump() for item in result.results],
            passed=result.passed,
            stable=result.stable,
            baseline=baseline,
        )
        return result

    async def _run(self, task):
        self._event("user", text=task)
        self.state.history.append(UserMessage(Prompt("coding_harness/task.j2")(task=task)))
        self.state.baseline = await self._checks(baseline=True)
        previous_failure = None
        while True:
            turn = await self.step()
            if turn.tool_calls:
                continue
            verification = await self._checks()
            feedback = render_verification(verification, self.state.baseline)
            self.state.history.append(UserMessage(feedback))
            changed = await self.workspace.changed_paths()
            if not self.config.checks:
                return self._result("unverified", turn.text, changed_paths=changed)
            fresh = verification.fingerprint == await self.workspace.fingerprint()
            if verification.passed and verification.stable and fresh:
                self.state.fingerprint = verification.fingerprint
                return self._result("verified", turn.text, verification.results, changed)
            failure = (
                verification.fingerprint,
                json.dumps(
                    [item.command.model_dump() for item in verification.results],
                    sort_keys=True,
                ),
            )
            if failure == previous_failure:
                return self._result(
                    "blocked",
                    "Checks failed again without workspace progress.",
                    verification.results,
                    changed,
                )
            if self.state.repairs >= self.config.limits.max_repairs:
                return self._result(
                    "blocked",
                    "Repair budget exhausted.\n" + feedback,
                    verification.results,
                    changed,
                )
            previous_failure = failure
            self.state.repairs += 1
            self._event("status", text=f"Repair {self.state.repairs}: checks need attention")

    async def _context(self):
        instructions = render_instructions(self.workspace, self.config)
        size = context_size(
            self.state.history, instructions=instructions, tools=list(self.tools.values())
        )
        limits = self.config.limits
        if size > limits.context_soft_chars and not self._compacted_last:
            try:
                await self._compact()
            except (ValueError, RuntimeError, BackendError) as error:
                self._event("status", text=f"Context summary failed: {error}")
            size = context_size(
                self.state.history, instructions=instructions, tools=list(self.tools.values())
            )
        if size > limits.context_hard_chars:
            raise RunStopped("Context exceeds the input-size limit; start a new conversation")
        return instructions

    async def step(self):
        instructions = await self._context()
        self._request_budget()
        self._event("status", text="Waiting for model")
        turn = await self.backend.aturn(
            self.state.history, tools=list(self.tools.values()), instructions=instructions
        )
        placeholders = [ToolResult(call.id, "pending") for call in turn.tool_calls]
        self._history_valid([*self.state.history, turn, *placeholders])
        self.state.history.append(turn)
        self._compacted_last = False
        if turn.text:
            usage = {
                key: value
                for key, value in (
                    ("input_tokens", turn.input_tokens),
                    ("output_tokens", turn.output_tokens),
                )
                if value is not None
            }
            self._event("assistant", text=turn.text, usage=usage)
        await self._execute_calls(turn.tool_calls)
        return turn

    async def _execute_calls(self, calls):
        completed = 0
        started = False
        try:
            for call in calls:
                if self.state.tool_calls >= self.config.limits.max_tool_calls:
                    raise RunStopped("Tool call budget exhausted")
                self.state.tool_calls += 1
                started = True
                self._event("tool_started", id=call.id, name=call.name, arguments=call.arguments)
                result = await self._dispatch(call)
                self.state.history.append(result)
                completed += 1
                started = False
                self._event(
                    "tool_finished",
                    id=call.id,
                    name=call.name,
                    content=result.content,
                    is_error=result.is_error,
                )
        except BaseException:
            # Close the group even when the interrupted action's effects are unknown.
            for index, call in enumerate(calls[completed:]):
                detail = (
                    "Interrupted; effects may have occurred. Inspect current state."
                    if index == 0 and started
                    else "Not started: run stopped."
                )
                self.state.history.append(ToolResult(call.id, detail, True))
            raise
        finally:
            self.state.edit_ledger = list(self.workspace.edit_ledger)

    async def _dispatch(self, call):
        if call.argument_error or call.arguments is None:
            return ToolResult(call.id, call.argument_error or "Arguments must be an object", True)
        if call.name not in self.tools:
            return ToolResult(call.id, f"Unknown tool: {call.name}", True)
        try:
            content = await self.tools[call.name].ainvoke(call.arguments)
        except ToolError as error:
            effects = (
                "not executed"
                if error.phase == "arguments"
                else "effects may have occurred; inspect state"
            )
            return ToolResult(call.id, f"{error}; {effects}", True)
        return ToolResult(call.id, content)

    async def compact(self):
        if self.state.running:
            raise RuntimeError("Cannot compact during an active run")
        self.state.running = True
        try:
            await asyncio.wait_for(self._compact(), timeout=self.config.limits.task_timeout)
        finally:
            self.state.running = False

    async def _compact(self):
        if not self.state.history:
            raise ValueError("No conversation to compact")
        self._history_valid(self.state.history)
        self.state.edit_ledger = list(self.workspace.edit_ledger)
        text = summary_prompt(self.state, self.config)
        self._request_budget()
        response = await self.backend.aturn([UserMessage(text)], tools=[])
        if response.tool_calls:
            raise ValueError("Summary must not request tools")
        summary = parse(response.text, ContextSummary)
        checkpoint = Prompt("coding_harness/checkpoint.j2")(
            task=self.state.task,
            summary=summary,
            checks=[check.model_dump() for check in self.config.checks],
        )
        instructions = render_instructions(self.workspace, self.config)
        history = [UserMessage(checkpoint)]
        after = context_size(history, instructions=instructions, tools=list(self.tools.values()))
        if after > self.config.limits.context_hard_chars:
            raise ValueError("Summary exceeds the input-size limit")
        before = context_size(
            self.state.history, instructions=instructions, tools=list(self.tools.values())
        )
        self.state.archived_histories.append(self.state.history)
        self.state.history = history
        self._compacted_last = True
        self._event("compacted", before=before, after=after)

    async def new_conversation(self):
        if self.state.running:
            raise RuntimeError("Cannot reset during an active run")
        self.state.running = True
        try:
            await self.workspace.initialize()
            fingerprint = await self.workspace.fingerprint()
        finally:
            self.state.running = False
        self.state = SessionState(
            provider=self.state.provider,
            model=self.state.model,
            root=str(self.workspace.root),
            head=self.workspace.head,
            fingerprint=fingerprint,
        )
        self.workspace.edit_ledger.clear()
        self._compacted_last = False
        self._event("status", text="New conversation; workspace files preserved")

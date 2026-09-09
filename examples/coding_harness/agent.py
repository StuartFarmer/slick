"""A coding harness expressed as ordinary Python state and a bounded loop."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

from slick import Prompt, Session, parse
from slick.providers import ProviderError
from slick.tools._protocol import validate_response

from .context import (
    context_size,
    history_views,
    render_context,
    render_instructions,
    summary_prompt,
)
from .state import ContextSummary, HarnessEvent, RunResult, SessionState
from .verification import render_verification, verify


class RunStopped(RuntimeError):
    """A configured budget or lack of progress ended this task."""


class CodingAgent:
    def __init__(self, provider, workspace, config, *, emit=lambda event: None):
        self.provider, self.workspace, self.config = provider, workspace, config
        self.emit = emit
        identity = provider.identity()
        self.state = SessionState(
            provider=identity["provider"],
            model=identity["model"],
            root=str(workspace.root),
            head=workspace.head,
        )
        self.session = Session(
            provider=provider,
            tools=[
                workspace.list_files,
                workspace.search,
                workspace.read_file,
                workspace.edit_file,
                workspace.create_file,
                workspace.run_command,
                workspace.git_diff,
            ],
        )
        self._compacted_last = False
        render_instructions(workspace, config)

    def _note(self, text):
        self.state.context_notes.append(
            {"after": len(self.session.history), "role": "user", "text": text}
        )

    def _event(self, kind, **data):
        self.emit(HarnessEvent(kind, data))

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
        self._note(Prompt("coding_harness/task.j2")(task=task))
        self.state.baseline = await self._checks(baseline=True)
        previous_failure = None
        while True:
            text, requests = await self.step()
            if requests:
                continue
            verification = await self._checks()
            feedback = render_verification(verification, self.state.baseline)
            self._note(feedback)
            changed = await self.workspace.changed_paths()
            if not self.config.checks:
                return self._result("unverified", text, changed_paths=changed)
            fresh = verification.fingerprint == await self.workspace.fingerprint()
            if verification.passed and verification.stable and fresh:
                self.state.fingerprint = verification.fingerprint
                return self._result("verified", text, verification.results, changed)
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

    def _size(self, state=None):
        state = self.state if state is None else state
        return context_size(
            render_context(self.workspace, self.config, state, self.session),
            tools=self.session.tools,
            tool_results=self.session.ready_results,
        )

    async def _context(self):
        size = self._size()
        limits = self.config.limits
        if size > limits.context_soft_chars and not self._compacted_last:
            try:
                await self._compact()
            except (ValueError, RuntimeError, ProviderError) as error:
                self._event("status", text=f"Context summary failed: {error}")
            size = self._size()
        if size > limits.context_hard_chars:
            raise RunStopped("Context exceeds the input-size limit; start a new conversation")
        return render_context(self.workspace, self.config, self.state, self.session)

    async def step(self):
        # A restored session may have work that has not started yet.
        await self._execute_calls(self.session.pending_requests)
        context = await self._context()
        self._request_budget()
        self._event("status", text="Waiting for model")
        text, requests = await self.session.acall(context, provider=self.provider)
        self._compacted_last = False
        if text:
            self._event("assistant", text=text)
        await self._execute_calls(requests)
        return text, requests

    async def _execute_calls(self, calls):
        try:
            for call in calls:
                if self.state.tool_calls >= self.config.limits.max_tool_calls:
                    raise RunStopped("Tool call budget exhausted")
                self.state.tool_calls += 1
                self._event(
                    "tool_started", id=call["id"], name=call["name"], arguments=call["arguments"]
                )
                result = await self.session.resolve(call)
                self._event(
                    "tool_finished",
                    id=call["id"],
                    name=call["name"],
                    content=result["content"],
                    is_error=result["is_error"],
                )
        except BaseException:
            self.session.cancel_pending("run stopped")
            raise
        finally:
            self.state.edit_ledger = list(self.workspace.edit_ledger)

    async def compact(self):
        if self.state.running:
            raise RuntimeError("Cannot compact during an active run")
        self.state.running = True
        try:
            await asyncio.wait_for(self._compact(), timeout=self.config.limits.task_timeout)
        finally:
            self.state.running = False

    async def _compact(self):
        if not history_views(self.session, self.state, include_ready=True):
            raise ValueError("No conversation to compact")
        self.state.edit_ledger = list(self.workspace.edit_ledger)
        text = summary_prompt(self.state, self.config, self.session)
        self._request_budget()
        text, requests = validate_response(await self.provider.acall(text, tools=[]))
        if requests:
            raise ValueError("Summary must not request tools")
        summary = parse(text, ContextSummary)
        checkpoint = Prompt("coding_harness/checkpoint.j2")(
            task=self.state.task,
            summary=summary,
            checks=[check.model_dump() for check in self.config.checks],
        )
        cursor = len(self.session.history)
        notes = [{"after": cursor, "role": "user", "text": checkpoint}]
        after = self._size(replace(self.state, context_notes=notes, context_start=cursor))
        if after > self.config.limits.context_hard_chars:
            raise ValueError("Summary exceeds the input-size limit")
        before = self._size()
        self.state.archived_histories.append(
            history_views(self.session, self.state, include_ready=True)
        )
        self.state.context_notes = notes
        self.state.context_start = cursor
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
        self.session = Session(provider=self.provider, tools=self.session.tools)
        self.workspace.edit_ledger.clear()
        self._compacted_last = False
        self._event("status", text="New conversation; workspace files preserved")

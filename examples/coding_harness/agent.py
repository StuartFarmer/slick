"""The coding loop: ask the model, run its tools, check the work, repeat."""

import asyncio
import json
from pathlib import Path

from slick import Prompt, Session
from slick.providers import ProviderError
from slick.tools import prepare_tools

from . import session
from .checks import feedback, run_checks, verification_paths
from .ui import ConsoleUI
from .workspace import Workspace

SKILLS = {"python": "coding_harness/skills/python.j2"}


class CodingAgent:
    def __init__(self, provider, workspace, config, *, ui=None):
        unknown = set(config.skills) - SKILLS.keys()
        if unknown:
            raise ValueError(f"Unknown skills: {', '.join(sorted(unknown))}")
        self.provider = provider
        self.workspace = workspace
        self.config = config
        self.ui = ui or ConsoleUI()
        self.messages = []
        self.running = False
        self.last_result = None
        self.fingerprint = ""
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

    async def run(self, task):
        if self.running:
            raise RuntimeError("A run is already active")
        if not task.strip():
            raise ValueError("Task must not be empty")
        self.running = True
        usage = {"turns": 0, "tool_calls": 0, "repairs": 0}
        self.last_result = None
        try:
            self.last_result = await asyncio.wait_for(
                self._run(task, usage), self.config.limits.task_timeout
            )
        except asyncio.CancelledError:
            self.last_result = self.result("cancelled", "Cancelled; completed edits remain.", usage)
            raise
        except (asyncio.TimeoutError, BudgetExceeded) as error:
            reason = str(error) or "Task deadline exceeded"
            self.last_result = self.result("blocked", reason, usage)
        except Exception as error:
            self.last_result = self.result("failed", f"{type(error).__name__}: {error}", usage)
        finally:
            self.running = False
            if self.last_result is not None:
                self.ui.completed(self.last_result)
        return self.last_result

    async def _run(self, task, usage):
        self.messages.append({"role": "user", "text": task})
        self.ui.user(task)
        baseline = await run_checks(self.workspace, self.config.checks)
        self.ui.checks(baseline["results"], baseline=True)
        self.messages.append({"role": "user", "text": feedback(baseline, baseline=True)})
        # Each task starts fresh; interrupted requests remain observations, never queued work.
        conversation = Session(provider=self.provider, tools=list(self.tools.values()))
        previous_failure = None
        limits = self.config.limits

        while True:
            context = self.context()
            if len(context) > limits.context_soft_chars:
                try:
                    replacement = await session.compact(
                        self.provider,
                        self.messages,
                        limits.context_hard_chars,
                        before_request=lambda: self.count_request(usage),
                    )
                    self.messages = replacement
                    context = self.context()
                    self.ui.status("Compacted earlier conversation")
                except (ValueError, RuntimeError, ProviderError) as error:
                    self.ui.status(f"Could not compact: {error}")
            size = len(
                json.dumps(
                    {
                        "context": context,
                        "results": conversation.ready_results,
                        "tools": [tool.parameters for tool in self.tools.values()],
                    },
                    ensure_ascii=False,
                )
            )
            if size > limits.context_hard_chars:
                raise BudgetExceeded("Context limit reached; start a new conversation")
            self.count_request(usage)
            self.ui.status("Thinking")
            text, calls = await conversation.acall(context)
            self.messages.append({"role": "assistant", "text": text, "calls": calls})
            if text:
                self.ui.assistant(text)
            if calls:
                await self.execute(conversation, usage)
                continue

            self.ui.status("Checking changes")
            verification = await run_checks(self.workspace, self.config.checks)
            self.ui.checks(verification["results"])
            paths = await self.workspace.changed_paths()
            fresh = verification["fingerprint"] == await self.workspace.fingerprint()
            if not fresh:
                verification.update(stable=False, passed=False)
            self.messages.append({"role": "user", "text": feedback(verification)})
            if not self.config.checks:
                return self.result("unverified", text, usage, paths=paths)
            if verification["passed"]:
                self.fingerprint = verification["fingerprint"]
                return self.result("verified", text, usage, verification["results"], paths)
            failure = (
                verification["fingerprint"],
                json.dumps(verification["results"], sort_keys=True),
            )
            if failure == previous_failure or usage["repairs"] >= limits.max_repairs:
                return self.result(
                    "blocked",
                    "Checks still fail.\n" + feedback(verification),
                    usage,
                    verification["results"],
                    paths,
                )
            previous_failure = failure
            usage["repairs"] += 1
            self.ui.status(f"Repair {usage['repairs']}")

    async def execute(self, conversation, usage):
        calls = conversation.pending_requests
        ids = [call["id"] for call in calls]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate tool request IDs")
        completed = 0
        try:
            for call in calls:
                if usage["tool_calls"] >= self.config.limits.max_tool_calls:
                    raise BudgetExceeded("Tool call budget exhausted")
                usage["tool_calls"] += 1
                self.ui.status(f"Running {call['name']}")
                result = await conversation.resolve(call)
                content, error = result["content"], result["is_error"]
                completed += 1
                self.messages.append(
                    {
                        "role": "tool",
                        "text": content,
                        "id": call["id"],
                        "error": error,
                    }
                )
                self.ui.tool(call["name"], content, error)
        finally:
            # A follow-up sees interrupted work, but never schedules it again.
            for call in calls[completed:]:
                self.messages.append(
                    {
                        "role": "tool",
                        "id": call["id"],
                        "error": True,
                        "text": "Stopped before completion; inspect files before retrying.",
                    }
                )

    def context(self):
        return Prompt("coding_harness/context.j2")(
            root=str(self.workspace.root),
            checks=[check.model_dump() for check in self.config.checks],
            limits=self.config.limits.model_dump(),
            skills=[SKILLS[name] for name in self.config.skills],
            history=self.messages,
        )

    def count_request(self, usage):
        if usage["turns"] >= self.config.limits.max_turns:
            raise BudgetExceeded("Model request budget exhausted")
        usage["turns"] += 1

    def result(self, status, answer, usage, checks=(), paths=None):
        paths = paths if paths is not None else sorted(self.workspace.edited_paths)
        changed_checks = verification_paths(paths, self.config.checks, self.workspace.root)
        if changed_checks:
            answer += "\nVerification-related files changed: " + ", ".join(changed_checks)
        return dict(status=status, answer=answer, checks=list(checks), changed_paths=paths, **usage)

    async def compact(self):
        if self.running:
            raise RuntimeError("Cannot compact during an active run")
        self.running = True
        try:
            self.messages = await asyncio.wait_for(
                session.compact(
                    self.provider, self.messages, self.config.limits.context_hard_chars
                ),
                self.config.limits.task_timeout,
            )
            self.ui.status("Compacted earlier conversation")
        finally:
            self.running = False

    async def new_conversation(self):
        if self.running:
            raise RuntimeError("Cannot reset during an active run")
        self.running = True
        try:
            await self.workspace.initialize()
            self.fingerprint = await self.workspace.fingerprint()
            self.messages.clear()
            self.workspace.edited_paths.clear()
            self.last_result = None
        finally:
            self.running = False

    def save(self, path):
        session.save(path, self)


class BudgetExceeded(RuntimeError):
    """An explicit task budget stopped the loop."""


async def create_agent(provider, root, config, *, ui=None, saved=None):
    ui = ui or ConsoleUI()
    if saved is not None and (
        not Path(saved.root).is_absolute() or Path(saved.root).resolve() != Path(saved.root)
    ):
        raise ValueError("Session workspace must be an absolute resolved path")
    workspace = Workspace(
        root, checks=config.checks, decide=ui.approve, command_timeout=config.limits.command_timeout
    )
    await workspace.initialize(create=saved is None)
    agent = CodingAgent(provider, workspace, config, ui=ui)
    agent.fingerprint = await workspace.fingerprint()
    if saved is not None:
        if Path(saved.root) != workspace.root or config != saved.config:
            raise ValueError("Resume configuration does not match saved session")
        agent.messages = [message.model_dump(exclude_none=True) for message in saved.messages]
        if saved.fingerprint != agent.fingerprint:
            agent.messages.append(
                {
                    "role": "user",
                    "text": "Workspace changed since save; read current files and recheck.",
                }
            )
    return agent

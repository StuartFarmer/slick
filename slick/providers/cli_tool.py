"""CLI providers: one prompt in, (final text, []) out.

    provider = CodexCLI(model="YOUR_MODEL_ID")
    provider = ClaudeCLI()                    # use the invoked CLI's model default
    report, requests = provider.call(prompt)  # requests == []

CLI providers (Claude Code headless, codex exec) use the invoked CLI's
authentication and billing configuration. Give a provider everything it
needs in the prompt; each call starts a new invocation in the configured
working directory. `execute()` also accepts per-call workdir and sandbox
settings, whose permission semantics depend on the CLI.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import subprocess
from pathlib import Path

from .base import Provider, ProviderError

DEFAULT_CODEX_BIN = "codex"
DEFAULT_CLAUDE_CMD = "claude -p --output-format text --permission-mode bypassPermissions"


class ExecutionResult:
    """Outcome of one CLI execution.

    final_message: the closing message (report, answer).
    transcript: everything printed while working.
    """

    def __init__(self, final_message: str, transcript: str):
        self.final_message = final_message
        self.transcript = transcript


class Command(Provider):
    """Run an ordinary command with stdin, stdout, timeout, and cancellation handling."""

    provider = "model"

    def __init__(
        self,
        command: str,
        model: str | None = None,
        timeout: int = 3600,
        *,
        workdir: Path | str | None = None,
    ):
        self.command = command
        self.model = model
        self.timeout = timeout
        self.workdir = Path(workdir) if workdir is not None else None

    def call(self, context: str, *, tools=None, tool_results=None):
        """Run the CLI and return (final text, [])."""
        return self.execute(context).final_message, []

    async def acall(self, context: str, *, tools=None, tool_results=None):
        """Run the CLI asynchronously and return (final text, [])."""
        return (await self.aexecute(context)).final_message, []

    def execute(
        self,
        prompt: str,
        workdir: Path | str | None = None,
        sandbox: str = "read-only",
    ) -> ExecutionResult:
        """Low-level engine: run `prompt` through the CLI. Everything flows
        through stdout as strings."""
        workdir = self.workdir if workdir is None else Path(workdir)

        command = self._command(sandbox=sandbox)
        transcript = self._run_process(command, prompt, workdir=workdir)

        return ExecutionResult(final_message=self._parse_final(transcript), transcript=transcript)

    async def aexecute(
        self,
        prompt: str,
        workdir: Path | str | None = None,
        sandbox: str = "read-only",
    ) -> ExecutionResult:
        """Run the CLI asynchronously; cancellation kills and reaps its process."""
        workdir = self.workdir if workdir is None else Path(workdir)
        command = self._command(sandbox=sandbox)
        transcript = await self._arun_process(command, prompt, workdir=workdir)
        return ExecutionResult(final_message=self._parse_final(transcript), transcript=transcript)

    def _command(self, *, sandbox):
        return shlex.split(self.command)

    def _parse_final(self, transcript: str) -> str:
        """Extract the final message; defaults to the whole transcript."""
        return transcript

    def _run_process(self, command: list[str], prompt: str, *, workdir: Path | None) -> str:
        """Feed `prompt` over stdin, return captured stdout."""
        try:
            result = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=workdir,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"{self.provider} timed out after {self.timeout}s.") from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise ProviderError(f"{self.provider} failed (exit {result.returncode}): {detail}")
        return result.stdout

    async def _arun_process(self, command: list[str], prompt: str, *, workdir: Path | None) -> str:
        """Feed stdin and capture stdout without blocking the event loop."""
        encoded_prompt = prompt.encode()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            start_new_session=os.name == "posix",
        )
        communication = asyncio.create_task(process.communicate(encoded_prompt))
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communication), timeout=self.timeout
            )
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            try:
                if os.name == "posix":
                    # Include children that keep the CLI's output pipes open.
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            await communication
            if isinstance(exc, asyncio.TimeoutError):
                raise ProviderError(f"{self.provider} timed out after {self.timeout}s.") from exc
            raise

        if process.returncode != 0:
            detail = (stderr or stdout).decode().strip()[-2000:]
            raise ProviderError(f"{self.provider} failed (exit {process.returncode}): {detail}")
        return stdout.decode()


class ClaudeCLI(Command):
    """Claude Code headless: the final message is stdout."""

    provider = "claude"

    def __init__(self, command: str = DEFAULT_CLAUDE_CMD, **kwargs) -> None:
        super().__init__(command, **kwargs)

    def _command(self, *, sandbox):
        command = shlex.split(self.command)
        if self.model:
            command.extend(["--model", self.model])
        return command


class CodexCLI(Command):
    """codex exec --json: JSONL events stream on stdout; the final message
    is the last agent_message event."""

    provider = "codex"

    def __init__(self, command: str = DEFAULT_CODEX_BIN, **kwargs) -> None:
        super().__init__(command, **kwargs)

    def _command(self, *, sandbox) -> list[str]:
        command = [*shlex.split(self.command), "exec", "--json", "--sandbox", sandbox]
        if self.model:
            command.extend(["--model", self.model])
        command.append("-")
        return command

    def _parse_final(self, transcript: str) -> str:
        final = ""
        for line in transcript.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event["type"] == "item.completed" and event["item"]["type"] == "agent_message":
                final = event["item"]["text"]
        return final

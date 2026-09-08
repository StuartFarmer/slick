"""CLI providers: one prompt in, one text response out.

    provider = get_command()                  # the resolved default provider
    provider = get_command("claude")          # or by name
    report = provider.call(prompt)          # -> str

CLI providers (Claude Code headless, codex exec) use the invoked CLI's
authentication and billing configuration. Give a provider everything it
needs in the prompt; each call starts a new invocation in the configured
working directory. `execute()` also accepts per-call workdir and sandbox
settings, whose permission semantics depend on the CLI.

Which provider you get, in order: the argument to get_command(), whatever
set_default() was last called with, $SLICK_PROVIDER / $SLICK_MODEL, then
DEFAULT_PROVIDER.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ._base import Provider, ProviderError

DEFAULT_CODEX_BIN = "codex"
DEFAULT_CLAUDE_CMD = "claude -p --output-format text --permission-mode bypassPermissions"
DEFAULT_PROVIDER = "codex"


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one CLI execution.

    final_message: the closing message (report, answer).
    transcript: everything printed while working.
    """

    final_message: str
    transcript: str


class Command(Provider):
    """A CLI provider: one prompt in, one text response out."""

    provider: str = "model"

    def __init__(
        self,
        command: str,
        model: str | None = None,
        timeout: int = 3600,
        *,
        workdir: Path | str | None = None,
    ) -> None:
        self.command = command
        self.model = model
        self.timeout = timeout
        self.workdir = Path(workdir) if workdir is not None else None

    def call(self, prompt: str) -> str:
        """Single call with complete context, returning the response text."""
        return self.execute(prompt, sandbox="read-only").final_message

    async def acall(self, prompt: str) -> str:
        """Single native async call, returning the response text."""
        return (await self.aexecute(prompt, sandbox="read-only")).final_message

    def execute(
        self,
        prompt: str,
        workdir: Path | str | None = None,
        sandbox: str = "read-only",
    ) -> ExecutionResult:
        """Low-level engine: run `prompt` through the CLI. Everything flows
        through stdout as strings."""
        workdir = self.workdir if workdir is None else Path(workdir)

        command = self._command(workdir=workdir, sandbox=sandbox)
        transcript = self._run_process(command, prompt, workdir=workdir, label=self.provider)

        return ExecutionResult(final_message=self._parse_final(transcript), transcript=transcript)

    async def aexecute(
        self,
        prompt: str,
        workdir: Path | str | None = None,
        sandbox: str = "read-only",
    ) -> ExecutionResult:
        """Run the CLI asynchronously; cancellation kills and reaps its process."""
        workdir = self.workdir if workdir is None else Path(workdir)
        command = self._command(workdir=workdir, sandbox=sandbox)
        transcript = await self._arun_process(command, prompt, workdir=workdir, label=self.provider)
        return ExecutionResult(final_message=self._parse_final(transcript), transcript=transcript)

    @abstractmethod
    def _command(self, *, workdir: Path | None, sandbox: str) -> list[str]:
        """The CLI command to execute."""

    def _parse_final(self, transcript: str) -> str:
        """Extract the final message; defaults to the whole transcript."""
        return transcript

    def _run_process(
        self,
        command: list[str],
        prompt: str,
        *,
        workdir: Path | None,
        label: str,
    ) -> str:
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
            raise ProviderError(f"{label} timed out after {self.timeout}s.") from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise ProviderError(f"{label} failed (exit {result.returncode}): {detail}")
        return result.stdout

    async def _arun_process(
        self,
        command: list[str],
        prompt: str,
        *,
        workdir: Path | None,
        label: str,
    ) -> str:
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
            await process.wait()
            if isinstance(exc, asyncio.TimeoutError):
                raise ProviderError(f"{label} timed out after {self.timeout}s.") from exc
            raise

        if process.returncode != 0:
            detail = (stderr or stdout).decode().strip()[-2000:]
            raise ProviderError(f"{label} failed (exit {process.returncode}): {detail}")
        return stdout.decode()


class ClaudeCLI(Command):
    """Claude Code headless: the final message is stdout."""

    provider = "claude"

    def __init__(self, command: str = DEFAULT_CLAUDE_CMD, **kwargs) -> None:
        super().__init__(command, **kwargs)

    def _command(self, *, workdir, sandbox) -> list[str]:
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

    def _command(self, *, workdir, sandbox) -> list[str]:
        base = [sys.executable, self.command] if self.command.endswith(".py") else [self.command]
        command = [*base, "exec", "--json", "--sandbox", sandbox]
        if workdir is not None:
            # The child already starts in workdir; a relative --cd would apply it twice.
            command.extend(["--cd", str(workdir.absolute())])
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
            item = event.get("item", {}) if isinstance(event, dict) else {}
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                final = item.get("text", "")
        return final


COMMANDS: dict[str, type[Command]] = {
    "claude": ClaudeCLI,
    "codex": CodexCLI,
}

_default_provider: str | None = None
_default_model: str | None = None


def set_default(provider: str | None = None, model: str | None = None) -> None:
    """Set the process-wide default provider and/or model id."""
    global _default_provider, _default_model
    if provider is not None:
        if provider not in COMMANDS:
            raise KeyError(f"Unknown provider {provider!r}; expected one of {', '.join(COMMANDS)}.")
        _default_provider = provider
    if model is not None:
        _default_model = model


def get_default() -> tuple[str, str | None]:
    """The provider and model id a bare get_command() would use."""
    provider = _default_provider or os.getenv("SLICK_PROVIDER") or DEFAULT_PROVIDER
    return provider, _default_model or os.getenv("SLICK_MODEL")


def get_command(provider: str | None = None, model: str | None = None, **properties) -> Command:
    """Build a CLI provider. Unset arguments fall back to the resolved default."""
    default_provider, default_model = get_default()
    provider = provider or default_provider
    if provider not in COMMANDS:
        raise KeyError(f"Unknown provider {provider!r}; expected one of {', '.join(COMMANDS)}.")
    return COMMANDS[provider](model=model or default_model, **properties)

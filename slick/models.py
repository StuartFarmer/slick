"""Model primitive: one prompt in, one text response out.

    model = get_model()                  # the resolved default backend
    model = get_model("claude")          # or by name
    report = model.call(prompt)          # -> str

Models are CLI-backed (Claude Code headless, codex exec) so calls run on
subscription billing rather than metered API tokens. Give a model
everything it needs in the prompt; there is no workspace and no
conversation. `execute()` is the low-level engine — it takes a workdir
and a sandbox, which is what a long-running workspace task needs.

Which backend you get, in order: the argument to get_model(), whatever
set_default() was last called with, $SLICK_BACKEND / $SLICK_MODEL, then
DEFAULT_BACKEND.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CODEX_BIN = "codex"
DEFAULT_CLAUDE_CMD = "claude -p --output-format text --permission-mode bypassPermissions"
DEFAULT_BACKEND = "codex"


class ModelError(Exception):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one model execution.

    final_message: the closing message (report, answer).
    transcript: everything printed while working.
    """

    final_message: str
    transcript: str


class Model(ABC):
    """A CLI-backed model: one prompt in, one text response out."""

    backend: str = "model"

    def __init__(self, command: str, model: str | None = None, timeout: int = 3600) -> None:
        self.command = command
        self.model = model
        self.timeout = timeout

    def call(self, prompt: str) -> str:
        """Single call with complete context, returning the response text."""
        return self.execute(prompt, sandbox="read-only").final_message

    def execute(
        self,
        prompt: str,
        workdir: Path | str | None = None,
        sandbox: str = "read-only",
    ) -> ExecutionResult:
        """Low-level engine: run `prompt` through the CLI. Everything flows
        through stdout as strings."""
        if workdir:
            workdir = Path(workdir)

        command = self._command(workdir=workdir, sandbox=sandbox)
        transcript = self._run_process(command, prompt, workdir=workdir, label=self.backend)

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
            raise ModelError(f"{label} timed out after {self.timeout}s.") from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise ModelError(f"{label} failed (exit {result.returncode}): {detail}")
        return result.stdout


class ClaudeModel(Model):
    """Claude Code headless: the final message is stdout."""

    backend = "claude"

    def __init__(self, command: str = DEFAULT_CLAUDE_CMD, **kwargs) -> None:
        super().__init__(command, **kwargs)

    def _command(self, *, workdir, sandbox) -> list[str]:
        command = shlex.split(self.command)
        if self.model:
            command.extend(["--model", self.model])
        return command


class CodexModel(Model):
    """codex exec --json: JSONL events stream on stdout; the final message
    is the last agent_message event."""

    backend = "codex"

    def __init__(self, command: str = DEFAULT_CODEX_BIN, **kwargs) -> None:
        super().__init__(command, **kwargs)

    def _command(self, *, workdir, sandbox) -> list[str]:
        base = [sys.executable, self.command] if self.command.endswith(".py") else [self.command]
        command = [*base, "exec", "--json", "--sandbox", sandbox]
        if workdir is not None:
            command.extend(["--cd", str(workdir)])
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


BACKENDS: dict[str, type[Model]] = {
    "claude": ClaudeModel,
    "codex": CodexModel,
}

_default_backend: str | None = None
_default_model: str | None = None


def set_default(backend: str | None = None, model: str | None = None) -> None:
    """Set the process-wide default backend and/or model id."""
    global _default_backend, _default_model
    if backend is not None:
        if backend not in BACKENDS:
            raise KeyError(f"Unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}.")
        _default_backend = backend
    if model is not None:
        _default_model = model


def get_default() -> tuple[str, str | None]:
    """The backend and model id a bare get_model() would use."""
    backend = _default_backend or os.getenv("SLICK_BACKEND") or DEFAULT_BACKEND
    return backend, _default_model or os.getenv("SLICK_MODEL")


def get_model(backend: str | None = None, model: str | None = None, **properties) -> Model:
    """Build a model. Unset arguments fall back to the resolved default."""
    default_backend, default_model = get_default()
    backend = backend or default_backend
    if backend not in BACKENDS:
        raise KeyError(f"Unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}.")
    return BACKENDS[backend](model=model or default_model, **properties)

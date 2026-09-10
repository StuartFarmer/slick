"""Bounded observations from locally approved POSIX commands, not a sandbox."""

import asyncio
import codecs
import os
import signal
import sys
import warnings
from pathlib import Path

from pydantic import BaseModel


class CommandResult(BaseModel, extra="forbid"):
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


class _Output:
    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.size = 0
        self.truncated = False

    def append(self, text):
        remaining = self.limit - self.size
        if remaining and text:
            self.parts.append(text[:remaining])
        self.size += min(len(text), remaining)
        self.truncated |= len(text) > remaining

    def notice(self, text):
        message = "\n" + text
        self.truncated |= self.size + len(message) > self.limit
        self.parts = [self.text()[: max(0, self.limit - len(message))] + message[-self.limit :]]
        self.size = len(self.parts[0])

    async def drain(self, stream):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := await stream.read(16384):
            self.append(decoder.decode(chunk))
        self.append(decoder.decode(b"", final=True))

    def text(self):
        return "".join(self.parts)


def validate_command(argv, timeout):
    if sys.platform not in ("darwin", "linux"):
        raise RuntimeError("The coding harness requires macOS or Linux")
    if (
        not isinstance(argv, list)
        or not argv
        or not isinstance(argv[0], str)
        or not argv[0]
        or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
    ):
        raise ValueError("argv must contain an executable and no NUL bytes")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")


async def _cleanup(process, jobs):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _, pending = await asyncio.wait(jobs, timeout=5)
    if pending:
        # asyncio has no public close-pipes API. Closing the subprocess transport
        # detaches inherited pipes held by descendants outside our process group.
        process._transport.close()
        for job in pending:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        return "Process cleanup incomplete: a descendant retained an output pipe."
    return ""


async def _finish_cleanup(process, jobs):
    cleanup = asyncio.create_task(_cleanup(process, jobs))
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    note = cleanup.result()
    if cancelled:
        if note:
            warnings.warn(note, RuntimeWarning, stacklevel=2)
        raise asyncio.CancelledError
    return note


async def run_process(
    argv: list[str], cwd: Path, timeout: int, *, output_limit: int = 8000
) -> CommandResult:
    validate_command(argv, timeout)
    if type(output_limit) is not int or output_limit <= 0:
        raise ValueError("output_limit must be a positive integer")
    stdout, stderr = _Output(output_limit), _Output(output_limit)
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    )
    cancelled = False
    try:
        process = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        cancelled = True
        # Own the child even when cancellation arrives during process creation.
        while not spawn.done():
            try:
                await asyncio.shield(spawn)
            except asyncio.CancelledError:
                continue
        if spawn.exception() is not None:
            raise asyncio.CancelledError from None
        process = spawn.result()
    except OSError as exc:
        message = str(exc)
        return CommandResult(
            argv=argv,
            exit_code=None,
            stdout="",
            stderr=message[:output_limit],
            timed_out=False,
            truncated=len(message) > output_limit,
        )
    jobs = [
        asyncio.create_task(stdout.drain(process.stdout)),
        asyncio.create_task(stderr.drain(process.stderr)),
        asyncio.create_task(process.wait()),
    ]
    timed_out = False
    try:
        if cancelled:
            raise asyncio.CancelledError
        _, pending = await asyncio.wait(jobs, timeout=timeout)
        if pending:
            timed_out = True
            note = await _finish_cleanup(process, jobs)
            if note:
                stderr.notice(note)
        else:
            for job in jobs:
                job.result()
            # The foreground process may have spawned descendants with their
            # output redirected. They still belong to this command's lifetime.
            await _finish_cleanup(process, jobs)
    except BaseException:
        note = await _finish_cleanup(process, jobs)
        if note:
            warnings.warn(note, RuntimeWarning, stacklevel=2)
        raise
    return CommandResult(
        argv=argv,
        exit_code=process.returncode,
        stdout=stdout.text(),
        stderr=stderr.text(),
        timed_out=timed_out,
        truncated=stdout.truncated or stderr.truncated,
    )

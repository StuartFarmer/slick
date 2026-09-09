"""CLI provider execution, subprocess cleanup, and model selection."""

import asyncio
import os
import shlex
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from slick.providers import (
    ClaudeCLI,
    CodexCLI,
    Command,
    ProviderError,
)


class FakeCommand(Command):
    """A Command with the subprocess replaced, so call/execute wiring is testable."""

    provider = "fake"

    def __init__(self, transcript: str = "report body", **kwargs) -> None:
        super().__init__("fake-cli", **kwargs)
        self.transcript = transcript
        self.runs: list[tuple] = []

    def _command(self, *, sandbox):
        return ["fake-cli", sandbox]

    def _run_process(self, command, prompt, *, workdir):
        self.runs.append((command, prompt, workdir))
        return self.transcript


@pytest.mark.parametrize("asynchronous", [False, True])
def test_base_runs_an_ordinary_command_without_provider_flags(asynchronous):
    argv = [sys.executable, "-c", "import sys; print(sys.argv[1:], end='')", "plain argument"]
    command = Command(shlex.join(argv), model="metadata-only")
    result = asyncio.run(command.acall("input")) if asynchronous else command.call("input")
    assert result == ("['plain argument']", [])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_calls_use_configured_workdir_and_execute_can_override(asynchronous):
    class AsyncFake(FakeCommand):
        async def _arun_process(self, command, prompt, *, workdir):
            return self._run_process(command, prompt, workdir=workdir)

    model = AsyncFake(workdir="/tmp/configured")

    def invoke(method, *args, **kwargs):
        result = getattr(model, ("a" if asynchronous else "") + method)(*args, **kwargs)
        return asyncio.run(result) if asynchronous else result

    invoke("call", "first")
    assert model.runs[-1][2] == Path("/tmp/configured")
    invoke("execute", "second", workdir="/tmp/override")
    assert model.runs[-1][2] == Path("/tmp/override")
    invoke("execute", "third", workdir="")
    assert model.runs[-1][2] == Path(".")
    invoke("call", "fourth")
    assert model.runs[-1][2] == Path("/tmp/configured")
    plain = AsyncFake()
    plain.call("positional context")
    assert plain.runs[-1][2] is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_configured_directory_reaches_the_child_process(tmp_path, asynchronous):
    model = Subprocess.RealCommand(
        [sys.executable, "-c", "import os; print(os.getcwd(), end='')"], workdir=tmp_path
    )
    result = asyncio.run(model.acall("prompt")) if asynchronous else model.call("prompt")
    assert Path(result[0]) == tmp_path.resolve()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_codex_relative_workdir_is_not_applied_twice(tmp_path, monkeypatch, asynchronous):
    workdir = tmp_path / "subdir"
    workdir.mkdir()
    script = tmp_path / "codex wrapper.py"
    script.write_text(
        "import json, os, sys\n"
        "if '--cd' in sys.argv: os.chdir(sys.argv[sys.argv.index('--cd') + 1])\n"
        "print(json.dumps({'type': 'item.completed', 'item': "
        "{'type': 'agent_message', 'text': os.getcwd()}}))\n"
    )
    monkeypatch.chdir(tmp_path)
    model = CodexCLI(command=shlex.join([sys.executable, str(script)]), workdir="subdir")
    result = asyncio.run(model.acall("prompt")) if asynchronous else model.call("prompt")
    assert Path(result[0]) == workdir.resolve()


class Call(unittest.TestCase):
    def test_returns_the_response_text(self):
        self.assertEqual(FakeCommand().call("analyze this"), ("report body", []))

    def test_execute_reports_both_the_final_message_and_the_transcript(self):
        result = FakeCommand().execute("analyze this")
        self.assertEqual(result.final_message, "report body")
        self.assertEqual(result.transcript, "report body")

    def test_an_empty_workdir_is_normalized_to_the_current_directory(self):
        model = FakeCommand()
        model.execute("prompt", workdir="")
        self.assertEqual(model.runs[0][2], Path("."))

    def test_the_prompt_is_fed_to_the_process(self):
        model = FakeCommand()
        model.call("analyze this")
        (_, prompt, _) = model.runs[0]
        self.assertEqual(prompt, "analyze this")

    def test_an_empty_response_is_returned_as_is(self):
        self.assertEqual(FakeCommand(transcript="").call("analyze this"), ("", []))


class Subprocess(unittest.TestCase):
    """The real _run_process, driven through python itself."""

    class RealCommand(Command):
        provider = "real"

        def __init__(self, argv: list[str], **kwargs) -> None:
            super().__init__("python", **kwargs)
            self.argv = argv

        def _command(self, *, sandbox):
            return self.argv

    def test_stdout_is_captured_and_stdin_is_the_prompt(self):
        echo = "import sys; sys.stdout.write(sys.stdin.read())"
        model = self.RealCommand([sys.executable, "-c", echo])
        self.assertEqual(model.call("echoed prompt"), ("echoed prompt", []))

    def test_a_nonzero_exit_raises_with_the_detail(self):
        model = self.RealCommand(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
        )
        with self.assertRaises(ProviderError) as caught:
            model.call("analyze this")
        message = str(caught.exception)
        self.assertIn("exit 3", message)
        self.assertIn("boom", message)

    def test_a_timeout_raises(self):
        model = self.RealCommand([sys.executable, "-c", "import time; time.sleep(5)"], timeout=1)
        with self.assertRaises(ProviderError) as caught:
            model.call("analyze this")
        self.assertIn("timed out", str(caught.exception))


class CodexTranscript(unittest.TestCase):
    def test_the_last_agent_message_wins(self):
        transcript = "\n".join(
            [
                '{"type": "item.completed", "item": {"type": "agent_message", "text": "first"}}',
                '{"type": "item.completed", "item": {"type": "reasoning", "text": "ignored"}}',
                '{"type": "item.completed", "item": {"type": "agent_message", "text": "final"}}',
            ]
        )
        self.assertEqual(CodexCLI()._parse_final(transcript), "final")

    def test_non_json_lines_are_skipped(self):
        transcript = "\n".join(
            [
                "warning: something on stdout",
                '{"type": "item.completed", "item": {"type": "agent_message", "text": "final"}}',
            ]
        )
        self.assertEqual(CodexCLI()._parse_final(transcript), "final")

    def test_a_transcript_with_no_agent_message_yields_empty(self):
        self.assertEqual(CodexCLI()._parse_final("just noise"), "")


class AsyncSubprocess(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_prompt_encoding_does_not_start_a_process(self):
        create_process = asyncio.create_subprocess_exec
        processes = []

        async def record_process(*args, **kwargs):
            process = await create_process(*args, **kwargs)
            processes.append(process)
            return process

        model = Subprocess.RealCommand([sys.executable, "-c", "import sys; sys.stdin.read()"])
        try:
            with patch(
                "slick.providers.cli_tool.asyncio.create_subprocess_exec",
                side_effect=record_process,
            ):
                with self.assertRaises(UnicodeEncodeError):
                    await model.acall("\ud800")
            self.assertEqual(processes, [])
        finally:
            for process in processes:
                if process.returncode is None:
                    process.kill()
                await process.communicate()

    async def test_stdout_is_captured_and_stdin_is_the_prompt(self):
        model = Subprocess.RealCommand(
            [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"]
        )
        self.assertEqual(await model.acall("echoed café"), ("echoed café", []))

    async def test_execute_preserves_workdir_and_sandbox(self):
        class AsyncFake(FakeCommand):
            async def _arun_process(self, command, prompt, *, workdir):
                return self._run_process(command, prompt, workdir=workdir)

        model = AsyncFake()
        result = await model.aexecute("prompt", workdir="/tmp/work", sandbox="workspace-write")
        self.assertEqual(result.final_message, "report body")
        self.assertEqual(result.transcript, "report body")
        self.assertEqual(
            model.runs,
            [(["fake-cli", "workspace-write"], "prompt", Path("/tmp/work"))],
        )
        await model.aexecute("prompt", workdir="")
        self.assertEqual(model.runs[-1][2], Path("."))

    async def test_a_nonzero_exit_raises_with_the_detail(self):
        model = Subprocess.RealCommand(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
        )
        with self.assertRaisesRegex(ProviderError, r"real failed \(exit 3\): boom"):
            await model.acall("prompt")

    async def test_timeout_kills_and_reaps_the_process(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "pid"
            code = (
                "import os, pathlib, sys, time; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
            )
            model = Subprocess.RealCommand([sys.executable, "-c", code, str(pid_path)], timeout=1)
            with self.assertRaisesRegex(ProviderError, "real timed out after 1s"):
                await model.acall("prompt")
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_path.read_text()), 0)

    async def test_cancellation_kills_and_reaps_the_process(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "pid"
            code = (
                "import os, pathlib, sys, time; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
            )
            model = Subprocess.RealCommand([sys.executable, "-c", code, str(pid_path)])
            task = asyncio.create_task(model.acall("prompt"))
            try:

                async def wait_started():
                    while not pid_path.exists():
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_started(), timeout=5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pid_path.read_text()), 0)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_codex_parses_the_final_message_and_keeps_the_transcript(self):
        transcript = (
            "\n".join(
                [
                    '{"type":"item.completed","item":{"type":"agent_message","text":"first"}}',
                    '{"type":"item.completed","item":{"type":"agent_message","text":"final"}}',
                ]
            )
            + "\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake_codex.py"
            script.write_text(f"import sys\nsys.stdin.read()\nsys.stdout.write({transcript!r})\n")
            result = await CodexCLI(command=shlex.join([sys.executable, str(script)])).aexecute(
                "prompt"
            )
        self.assertEqual(result.final_message, "final")
        self.assertEqual(result.transcript, transcript)

    @unittest.skipUnless(os.name == "posix", "Process groups require POSIX")
    async def test_cancellation_closes_pipes_inherited_by_a_child(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "pid"
            code = (
                "import os, pathlib, subprocess, sys; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))"
            )
            model = Subprocess.RealCommand([sys.executable, "-c", code, str(pid_path)])
            task = asyncio.create_task(model.acall("prompt"))
            try:

                async def wait_started():
                    while not pid_path.exists():
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_started(), timeout=5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=2)
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pid_path.read_text()), 0)
            finally:
                if pid_path.exists():
                    try:
                        os.killpg(int(pid_path.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)


class Commands(unittest.TestCase):
    def test_codex_passes_sandbox_and_model(self):
        command = CodexCLI(model="gpt-5")._command(sandbox="read-only")
        self.assertIn("--json", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[command.index("--model") + 1], "gpt-5")
        self.assertEqual(command[-1], "-")

    def test_codex_omits_workdir_and_model_when_unset(self):
        command = CodexCLI()._command(sandbox="read-only")
        self.assertNotIn("--cd", command)
        self.assertNotIn("--model", command)

    def test_claude_appends_the_model(self):
        command = ClaudeCLI(model="claude-opus-4-8")._command(sandbox="read-only")
        self.assertEqual(command[command.index("--model") + 1], "claude-opus-4-8")


if __name__ == "__main__":
    unittest.main()

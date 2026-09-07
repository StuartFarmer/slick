"""Model tests: the prompt-in/text-out contract every step is built on."""

import asyncio
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slick import models as models_module
from slick.models import (
    ClaudeModel,
    CodexModel,
    Model,
    ModelError,
    get_default,
    get_model,
    set_default,
)


class FakeModel(Model):
    """A Model with the subprocess replaced, so call/execute wiring is testable."""

    backend = "fake"

    def __init__(self, transcript: str = "report body", **kwargs) -> None:
        super().__init__("fake-cli", **kwargs)
        self.transcript = transcript
        self.runs: list[tuple] = []

    def _command(self, *, workdir, sandbox):
        return ["fake-cli", sandbox]

    def _run_process(self, command, prompt, *, workdir, label):
        self.runs.append((command, prompt, workdir, label))
        return self.transcript


class Call(unittest.TestCase):
    def test_returns_the_response_text(self):
        self.assertEqual(FakeModel().call("analyze this"), "report body")

    def test_execute_reports_both_the_final_message_and_the_transcript(self):
        result = FakeModel().execute("analyze this")
        self.assertEqual(result.final_message, "report body")
        self.assertEqual(result.transcript, "report body")

    def test_an_empty_workdir_is_normalized_to_the_current_directory(self):
        model = FakeModel()
        model.execute("prompt", workdir="")
        self.assertEqual(model.runs[0][2], Path("."))

    def test_the_prompt_is_fed_to_the_process(self):
        model = FakeModel()
        model.call("analyze this")
        (_, prompt, _, label) = model.runs[0]
        self.assertEqual(prompt, "analyze this")
        self.assertEqual(label, "fake")

    def test_an_empty_response_is_returned_as_is(self):
        # Deciding what an empty response means is the caller's job; @prompt
        # refuses to save one, but Model just reports what it got.
        self.assertEqual(FakeModel(transcript="").call("analyze this"), "")


class Subprocess(unittest.TestCase):
    """The real _run_process, driven through python itself."""

    class RealModel(Model):
        backend = "real"

        def __init__(self, argv: list[str], **kwargs) -> None:
            super().__init__("python", **kwargs)
            self.argv = argv

        def _command(self, *, workdir, sandbox):
            return self.argv

    def test_stdout_is_captured_and_stdin_is_the_prompt(self):
        echo = "import sys; sys.stdout.write(sys.stdin.read())"
        model = self.RealModel([sys.executable, "-c", echo])
        self.assertEqual(model.call("echoed prompt"), "echoed prompt")

    def test_a_nonzero_exit_raises_with_the_detail(self):
        model = self.RealModel(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
        )
        with self.assertRaises(ModelError) as caught:
            model.call("analyze this")
        message = str(caught.exception)
        self.assertIn("exit 3", message)
        self.assertIn("boom", message)

    def test_a_timeout_raises(self):
        model = self.RealModel([sys.executable, "-c", "import time; time.sleep(5)"], timeout=1)
        with self.assertRaises(ModelError) as caught:
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
        self.assertEqual(CodexModel()._parse_final(transcript), "final")

    def test_non_json_lines_are_skipped(self):
        transcript = "\n".join(
            [
                "warning: something on stdout",
                '{"type": "item.completed", "item": {"type": "agent_message", "text": "final"}}',
            ]
        )
        self.assertEqual(CodexModel()._parse_final(transcript), "final")

    def test_a_transcript_with_no_agent_message_yields_empty(self):
        self.assertEqual(CodexModel()._parse_final("just noise"), "")


class AsyncSubprocess(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_prompt_encoding_does_not_start_a_process(self):
        create_process = asyncio.create_subprocess_exec
        processes = []

        async def record_process(*args, **kwargs):
            process = await create_process(*args, **kwargs)
            processes.append(process)
            return process

        model = Subprocess.RealModel([sys.executable, "-c", "import sys; sys.stdin.read()"])
        try:
            with patch("slick.models.asyncio.create_subprocess_exec", side_effect=record_process):
                with self.assertRaises(UnicodeEncodeError):
                    await model.acall("\ud800")
            self.assertEqual(processes, [])
        finally:
            for process in processes:
                if process.returncode is None:
                    process.kill()
                await process.communicate()

    async def test_stdout_is_captured_and_stdin_is_the_prompt(self):
        model = Subprocess.RealModel(
            [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"]
        )
        self.assertEqual(await model.acall("echoed café"), "echoed café")

    async def test_execute_preserves_workdir_and_sandbox(self):
        class AsyncFake(FakeModel):
            async def _arun_process(self, command, prompt, *, workdir, label):
                return self._run_process(command, prompt, workdir=workdir, label=label)

        model = AsyncFake()
        result = await model.aexecute("prompt", workdir="/tmp/work", sandbox="workspace-write")
        self.assertEqual(result.final_message, "report body")
        self.assertEqual(result.transcript, "report body")
        self.assertEqual(
            model.runs,
            [(["fake-cli", "workspace-write"], "prompt", Path("/tmp/work"), "fake")],
        )
        await model.aexecute("prompt", workdir="")
        self.assertEqual(model.runs[-1][2], Path("."))

    async def test_a_nonzero_exit_raises_with_the_detail(self):
        model = Subprocess.RealModel(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
        )
        with self.assertRaisesRegex(ModelError, r"real failed \(exit 3\): boom"):
            await model.acall("prompt")

    async def test_timeout_kills_and_reaps_the_process(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "pid"
            code = (
                "import os, pathlib, sys, time; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
            )
            model = Subprocess.RealModel([sys.executable, "-c", code, str(pid_path)], timeout=1)
            with self.assertRaisesRegex(ModelError, "real timed out after 1s"):
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
            model = Subprocess.RealModel([sys.executable, "-c", code, str(pid_path)])
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
            result = await CodexModel(command=str(script)).aexecute("prompt")
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
            model = Subprocess.RealModel([sys.executable, "-c", code, str(pid_path)])
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
    def test_codex_passes_workdir_sandbox_and_model(self):
        command = CodexModel(model="gpt-5")._command(workdir=Path("/tmp/work"), sandbox="read-only")
        self.assertIn("--json", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[command.index("--cd") + 1], "/tmp/work")
        self.assertEqual(command[command.index("--model") + 1], "gpt-5")
        self.assertEqual(command[-1], "-")

    def test_codex_omits_workdir_and_model_when_unset(self):
        command = CodexModel()._command(workdir=None, sandbox="read-only")
        self.assertNotIn("--cd", command)
        self.assertNotIn("--model", command)

    def test_claude_appends_the_model(self):
        command = ClaudeModel(model="claude-opus-4-8")._command(workdir=None, sandbox="read-only")
        self.assertEqual(command[command.index("--model") + 1], "claude-opus-4-8")

    def test_get_model_selects_the_backend(self):
        self.assertIsInstance(get_model("codex"), CodexModel)
        self.assertIsInstance(get_model("claude"), ClaudeModel)
        with self.assertRaises(KeyError):
            get_model("nope")


class Defaults(unittest.TestCase):
    """Resolution order: argument, then set_default, then env, then built-in."""

    def setUp(self) -> None:
        for name in ("_default_backend", "_default_model"):
            self.addCleanup(setattr, models_module, name, getattr(models_module, name))
            setattr(models_module, name, None)
        for name in ("SLICK_BACKEND", "SLICK_MODEL"):
            self.addCleanup(os.environ.pop, name, None)
            os.environ.pop(name, None)

    def test_the_built_in_default_applies_when_nothing_is_set(self):
        self.assertEqual(get_default(), ("codex", None))
        self.assertIsInstance(get_model(), CodexModel)

    def test_set_default_is_used_by_a_bare_get_model(self):
        set_default(backend="claude", model="claude-opus-4-8")
        self.assertEqual(get_default(), ("claude", "claude-opus-4-8"))
        model = get_model()
        self.assertIsInstance(model, ClaudeModel)
        self.assertEqual(model.model, "claude-opus-4-8")

    def test_the_environment_applies_when_nothing_was_set_in_process(self):
        os.environ["SLICK_BACKEND"] = "claude"
        os.environ["SLICK_MODEL"] = "claude-haiku-4-5"
        self.assertEqual(get_default(), ("claude", "claude-haiku-4-5"))

    def test_set_default_beats_the_environment(self):
        os.environ["SLICK_BACKEND"] = "claude"
        set_default(backend="codex")
        self.assertEqual(get_default()[0], "codex")

    def test_an_argument_beats_everything(self):
        os.environ["SLICK_BACKEND"] = "claude"
        set_default(backend="claude", model="claude-opus-4-8")
        model = get_model("codex", "gpt-5")
        self.assertIsInstance(model, CodexModel)
        self.assertEqual(model.model, "gpt-5")

    def test_an_unknown_default_backend_is_rejected_where_it_is_set(self):
        with self.assertRaises(KeyError):
            set_default(backend="nope")


if __name__ == "__main__":
    unittest.main()

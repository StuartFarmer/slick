import asyncio
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from examples.coding_harness.process import run_process


def run(tmp_path, code, timeout=5, **kwargs):
    return asyncio.run(run_process([sys.executable, "-c", code], tmp_path, timeout, **kwargs))


def test_nonzero_exit_is_an_observation(tmp_path):
    result = run(tmp_path, "import sys; print('failure'); sys.exit(3)")
    assert result.exit_code == 3
    assert result.stdout.strip() == "failure"
    assert not result.timed_out


def test_both_streams_are_drained_and_bounded(tmp_path):
    result = run(
        tmp_path,
        "import os\nfor i in range(300):\n"
        " os.write(1, ('é'*4097).encode()); os.write(2,b'x'*16384)",
    )
    assert result.exit_code == 0
    assert result.stdout == "é" * 8000
    assert result.stderr == "x" * 8000
    assert result.truncated


def test_invalid_output_is_replaced(tmp_path):
    result = run(tmp_path, "import os; os.write(1,b'\\xff')")
    assert result.stdout == "\ufffd"


def test_timeout_reaps_process(tmp_path):
    result = run(tmp_path, "import os,time; print(os.getpid(),flush=True); time.sleep(60)", 1)
    assert result.timed_out
    with pytest.raises(ProcessLookupError):
        os.kill(int(result.stdout.strip()), 0)


def test_cancellation_reaps_process(tmp_path):
    pid_path = tmp_path / "pid"

    async def scenario():
        task = asyncio.create_task(
            run_process(
                [
                    sys.executable,
                    "-c",
                    "import os,time,pathlib; pathlib.Path('pid').write_text(str(os.getpid())); "
                    "time.sleep(60)",
                ],
                tmp_path,
                60,
            )
        )
        try:
            for _ in range(500):
                if pid_path.exists():
                    break
                await asyncio.sleep(0.01)
            assert pid_path.exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid_path.read_text()), 0)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_missing_executable_is_observation(tmp_path):
    result = asyncio.run(run_process(["/no/such/executable"], tmp_path, 5))
    assert result.exit_code is None
    assert result.stderr


@pytest.mark.parametrize(
    "argv,timeout", [([], 5), ([""], 5), (["x\0"], 5), (["x"], 0), (["x"], True)]
)
def test_invalid_arguments_never_launch(tmp_path, argv, timeout):
    with pytest.raises(ValueError):
        asyncio.run(run_process(argv, tmp_path, timeout))


@pytest.mark.parametrize("fill_stderr", [False, True])
def test_escaped_descendant_cannot_hold_runner_open(tmp_path, fill_stderr):
    pid_path = tmp_path / "escaped.pid"
    code = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c', "
        "\"import os,time,pathlib; pathlib.Path('escaped.pid').write_text(str(os.getpid())); "
        'time.sleep(60)"], start_new_session=True)'
    )
    if fill_stderr:
        code = "import os; os.write(2,b'x'*9000); " + code
    started = time.monotonic()
    try:
        result = run(tmp_path, code, 1)
        assert time.monotonic() - started < 8
        assert result.timed_out
        assert "cleanup incomplete" in result.stderr.lower()
    finally:
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cancellation_during_timeout_cleanup_propagates(tmp_path, monkeypatch):
    from examples.coding_harness import process as runner

    original_cleanup = runner._cleanup

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def paused_cleanup(process, jobs):
            entered.set()
            await release.wait()
            return await original_cleanup(process, jobs)

        monkeypatch.setattr(runner, "_cleanup", paused_cleanup)
        task = asyncio.create_task(
            run_process([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, 1)
        )
        try:
            await entered.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_successful_parent_does_not_leave_background_child(tmp_path):
    child = (
        "import os,time,pathlib; pathlib.Path('child.pid').write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    code = (
        "import subprocess,sys,time,pathlib; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);\n"
        "while not pathlib.Path('child.pid').exists(): time.sleep(.01)"
    )
    pid = None
    try:
        result = run(tmp_path, code)
        assert result.exit_code == 0
        pid = int((tmp_path / "child.pid").read_text())
        # Reparented descendants can briefly remain zombies; they must not run.
        deadline = time.monotonic() + 1
        while True:
            try:
                os.kill(pid, 0)
                status_path = Path(f"/proc/{pid}/status")
                zombie = status_path.exists() and "\nState:\tZ" in status_path.read_text()
                alive = not zombie
            except ProcessLookupError:
                alive = False
            if not alive or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert not alive, f"Child {pid} remains running"
    finally:
        if pid is None and (tmp_path / "child.pid").exists():
            pid = int((tmp_path / "child.pid").read_text())
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_launch_failure_observation_is_bounded(tmp_path):
    result = asyncio.run(run_process(["/" + "x" * 20000], tmp_path, 5))
    assert result.exit_code is None
    assert len(result.stderr) <= 8000
    assert result.truncated

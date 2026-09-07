import asyncio
import sys

from examples.coding_harness.state import Check
from examples.coding_harness.verification import render_verification, verify
from examples.coding_harness.workspace import Workspace


def make_workspace(repo, checks):
    async def deny(request):
        return "deny"

    workspace = Workspace(repo, checks=checks, decide=deny)
    asyncio.run(workspace.initialize())
    return workspace


def check(name, code, timeout=5):
    return Check(name=name, argv=[sys.executable, "-c", code], timeout=timeout)


def test_empty_verification_cannot_claim_success(workspace):
    result = asyncio.run(verify(workspace, []))
    assert not result.results and not result.passed


def test_real_checks_and_baseline_feedback(repo):
    checks = [
        check("pass", "print('good')"),
        check("fail", "import sys; print('bad'); sys.exit(1)"),
        check("last", "print('still ran')"),
    ]
    workspace = make_workspace(repo, checks)
    result = asyncio.run(verify(workspace, checks))
    assert not result.passed and result.stable
    assert [item.command.exit_code for item in result.results] == [0, 1, 0]
    assert all(
        item.before_fingerprint == item.after_fingerprint == result.fingerprint
        for item in result.results
    )
    feedback = render_verification(result, result)
    assert "bad" in feedback and "fail" in feedback and sys.executable in feedback
    assert "baseline" in feedback.lower()


def test_all_pass_stable(repo):
    checks = [check("unit", "print('ok')")]
    result = asyncio.run(verify(make_workspace(repo, checks), checks))
    assert result.passed and result.stable


def test_file_change_invalidates_verification(repo):
    checks = [
        check("mutate", "from pathlib import Path; Path('sample.py').write_text('value = 2\\n')")
    ]
    result = asyncio.run(verify(make_workspace(repo, checks), checks))
    assert not result.passed and not result.stable
    assert result.results[0].command.exit_code == 0
    assert result.results[0].before_fingerprint != result.results[0].after_fingerprint


def test_timeout_missing_and_denied_commands(repo):
    checks = [
        check("timeout", "import time; time.sleep(30)", timeout=1),
        Check(name="missing", argv=["/missing/environment/python"]),
    ]
    workspace = make_workspace(repo, checks)
    result = asyncio.run(verify(workspace, [*checks, check("denied", "print('not run')")]))
    assert not result.passed
    assert result.results[0].command.timed_out
    assert result.results[1].command.exit_code is None
    assert "denied" in result.results[2].command.stderr

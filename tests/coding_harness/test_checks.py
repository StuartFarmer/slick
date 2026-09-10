import asyncio
import sys

from examples.coding_harness.checks import Check, feedback, run_checks, verification_paths
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
    result = asyncio.run(run_checks(workspace, []))
    assert not result["results"] and not result["passed"]


def test_real_checks_and_feedback(repo):
    checks = [
        check("pass", "print('good')"),
        check("fail", "raise SystemExit(1)"),
        check("last", "print('still ran')"),
    ]
    result = asyncio.run(run_checks(make_workspace(repo, checks), checks))
    assert not result["passed"] and result["stable"]
    assert [item["command"]["exit_code"] for item in result["results"]] == [0, 1, 0]
    assert "Verification: failed" in feedback(result)
    assert "Baseline" in feedback(result, baseline=True)
    assert "still ran" in feedback(result)


def test_all_pass_stable(repo):
    checks = [check("unit", "print('ok')")]
    result = asyncio.run(run_checks(make_workspace(repo, checks), checks))
    assert result["passed"] and result["stable"]


def test_file_change_invalidates_verification(repo):
    checks = [check("mutate", "from pathlib import Path; Path('sample.py').write_text('changed')")]
    result = asyncio.run(run_checks(make_workspace(repo, checks), checks))
    assert not result["passed"] and not result["stable"]
    assert result["results"][0]["command"]["exit_code"] == 0


def test_timeout_missing_and_denied_commands(repo):
    checks = [
        check("timeout", "import time; time.sleep(30)", timeout=1),
        Check(name="missing", argv=["/missing/environment/python"]),
    ]
    workspace = make_workspace(repo, checks)
    result = asyncio.run(run_checks(workspace, [*checks, check("denied", "print('not run')")]))
    assert not result["passed"]
    assert result["results"][0]["command"]["timed_out"]
    assert result["results"][1]["command"]["exit_code"] is None
    assert "denied" in result["results"][2]["command"]["stderr"]


def test_verification_paths_identify_checks_and_test_files(tmp_path):
    checks = [Check(name="smoke", argv=[sys.executable, "scripts/smoke.py"])]
    assert verification_paths(
        ["src/app.py", "tests/test_app.py", "scripts/smoke.py"], checks, tmp_path
    ) == ["tests/test_app.py", "scripts/smoke.py"]

"""Verification is computed from real process results and complete file hashes."""

from slick import render

from .state import Check, CheckResult, Verification
from .workspace import Workspace


async def verify(workspace: Workspace, checks: list[Check]) -> Verification:
    before = await workspace.fingerprint()
    commands = []
    for check in checks:
        commands.append(await workspace.run_command(check.argv, check.timeout))
    after = await workspace.fingerprint()
    results = [
        CheckResult(
            name=check.name, command=command, before_fingerprint=before, after_fingerprint=after
        )
        for check, command in zip(checks, commands, strict=True)
    ]
    stable = before == after
    passed = (
        bool(results)
        and stable
        and all(
            result.command.exit_code == 0 and not result.command.timed_out for result in results
        )
    )
    return Verification(results=results, fingerprint=after, stable=stable, passed=passed)


def render_verification(current: Verification, baseline: Verification | None) -> str:
    return render("coding_harness/verification.j2", current=current, baseline=baseline)

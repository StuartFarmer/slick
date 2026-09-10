"""Run configured commands and give their actual output back to the agent."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Check(BaseModel, extra="forbid"):
    name: str = Field(min_length=1)
    argv: list[str]
    timeout: int = Field(default=120, strict=True, gt=0)

    @field_validator("argv")
    @classmethod
    def valid_argv(cls, argv):
        if not argv or not argv[0] or any("\0" in arg for arg in argv):
            raise ValueError("argv must contain an executable and no NUL bytes")
        return argv


async def run_checks(workspace, checks):
    before = await workspace.fingerprint()
    results = []
    for check in checks:
        command = await workspace.run_command(check.argv, check.timeout)
        results.append({"name": check.name, "command": command.model_dump()})
    after = await workspace.fingerprint()
    return {
        "results": results,
        "fingerprint": after,
        "stable": before == after,
        "passed": bool(results)
        and before == after
        and all(
            item["command"]["exit_code"] == 0 and not item["command"]["timed_out"]
            for item in results
        ),
    }


def feedback(verification, *, baseline=False):
    phase = "Baseline" if baseline else "Verification"
    verdict = "passed" if verification["passed"] else "failed"
    if not verification["results"]:
        return f"{phase}: no checks configured; work is unverified."
    lines = [f"{phase}: {verdict}"]
    for item in verification["results"]:
        command = item["command"]
        lines.append(
            f"{item['name']} {command['argv']}: exit_code: {command['exit_code']}\n"
            f"{command['stdout']}\n{command['stderr']}"
        )
        if command["timed_out"]:
            lines.append("Command timed out.")
        if command["truncated"]:
            lines.append("Output truncated.")
    if not verification["stable"]:
        lines.append("Workspace changed during checks; results are stale.")
    return "\n".join(lines)


def verification_paths(paths, checks, root):
    return [
        path
        for path in paths
        if any(part in {"tests", "test", ".github"} for part in Path(path).parts)
        or Path(path).name.startswith(("test_", "conftest"))
        or Path(path).name in {"pyproject.toml", "pytest.ini", "tox.ini", "Makefile"}
        or any(root / path == root / arg for check in checks for arg in check.argv)
    ]

from __future__ import annotations

"""CLI tests for `slick models` group (skeleton)."""

import subprocess
import sys
import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "slick.cli", *args], capture_output=True, text=True)


def test_cli_has_models_group_help() -> None:
    """`slick models --help` should show subcommands."""

    cp = _run_cli("models", "--help")
    assert cp.returncode == 0
    stdout = cp.stdout
    assert "list" in stdout
    assert "providers" in stdout
    assert "set-default" in stdout
    assert "show-default" in stdout
    assert "test" in stdout


@pytest.mark.skip(reason="Pending models implementation")
def test_cli_models_list_placeholder() -> None:
    """Placeholder output until wired to models API."""

    cp = _run_cli("models", "list")
    assert cp.returncode == 0
    # assert "TODO: list models" in cp.stdout
    pass



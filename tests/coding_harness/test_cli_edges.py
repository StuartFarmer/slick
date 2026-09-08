"""Offline CLI boundaries; all filesystem effects belong to temporary roots."""

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.coding_harness import __main__ as cli
from examples.coding_harness import workspace as workspace_module
from examples.coding_harness.state import RunResult


def real_provider_args(root):
    return [
        "--provider",
        "openai",
        "--model",
        "offline-test",
        "--workspace",
        str(root),
        "--headless",
        "--task",
        "Inspect the fixture",
    ]


@pytest.mark.parametrize("missing", ["git", "rg"])
def test_missing_workspace_executable_fails_before_model_use(repo, monkeypatch, capsys, missing):
    original = workspace_module.shutil.which
    monkeypatch.setattr(
        workspace_module.shutil,
        "which",
        lambda executable: None if executable == missing else original(executable),
    )
    # Initialization must fail before the provider needs even an identity/model call.
    monkeypatch.setattr(cli, "OpenAIAPI", lambda **kwargs: object())
    before = (repo / "sample.py").read_bytes()
    assert cli.main(real_provider_args(repo)) == 1
    assert f"{missing} is required" in capsys.readouterr().err
    assert (repo / "sample.py").read_bytes() == before


def test_missing_explicit_config_fails_without_constructing_provider(repo, monkeypatch, capsys):
    def forbidden_provider(**kwargs):
        raise AssertionError("A missing config must fail before constructing a provider")

    monkeypatch.setattr(cli, "OpenAIAPI", forbidden_provider)
    missing = repo / "missing-checks.json"
    assert cli.main([*real_provider_args(repo), "--config", str(missing)]) == 1
    error = capsys.readouterr().err
    assert "FileNotFoundError" in error and str(missing) in error


@pytest.mark.parametrize("os_name,platform", [("nt", "win32"), ("posix", "freebsd")])
def test_unsupported_platform_fails_before_demo_creation(monkeypatch, capsys, os_name, platform):
    monkeypatch.setattr(cli, "os", SimpleNamespace(name=os_name))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(platform=platform, stderr=sys.stderr))

    def forbidden_demo(root):
        raise AssertionError("Unsupported platforms must fail before fixture creation")

    monkeypatch.setattr(cli, "create_demo", forbidden_demo)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--dry-run", "--headless", "--task", "Fix total"])
    assert stopped.value.code == 2
    assert "macOS or Linux" in capsys.readouterr().err


@pytest.mark.parametrize(
    "override",
    [
        ["--provider", "openai"],
        ["--dry-run"],
        ["--model", "test"],
        ["--workspace", "/unused"],
        ["--config", "/unused.json"],
    ],
)
def test_resume_rejects_overrides_before_reading_session(tmp_path, capsys, override):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--resume", str(tmp_path / "absent.json"), *override])
    assert stopped.value.code == 2
    assert "--resume cannot be combined" in capsys.readouterr().err


def test_missing_textual_explains_install_command_without_traceback():
    code = """
import builtins
from examples.coding_harness.__main__ import main
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] == 'textual':
        raise ModuleNotFoundError("No module named 'textual'")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
raise SystemExit(main(['--dry-run']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "python -m pip install -r examples/coding_harness/requirements.txt" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "status,exit_code",
    [("verified", 0), ("unverified", 2), ("blocked", 2), ("failed", 1), ("cancelled", 130)],
)
def test_headless_maps_application_status_to_exit_code(tmp_path, monkeypatch, status, exit_code):
    class FinishedAgent:
        async def run(self, task):
            assert task == "offline task"
            return RunResult(
                status=status,
                answer="scripted result",
                checks=[],
                turns=1,
                tool_calls=0,
                repairs=0,
                changed_paths=[],
            )

    async def factory(provider, root, config, *, decide, emit, saved):
        assert await decide(None) == "deny"
        return FinishedAgent()

    monkeypatch.setattr(cli, "create_agent", factory)
    assert asyncio.run(cli._headless(None, tmp_path, None, "offline task", None)) == exit_code


@pytest.mark.parametrize("failure", [RuntimeError("scripted failure"), asyncio.CancelledError()])
def test_demo_root_is_removed_when_execution_fails(monkeypatch, capsys, failure):
    roots = []

    async def fail_after_real_setup(provider, root, config, task, saved):
        roots.append(Path(root))
        assert (root / "pricing.py").is_file()
        assert (root / ".git").is_dir()
        raise failure

    monkeypatch.setattr(cli, "_headless", fail_after_real_setup)
    expected = 130 if isinstance(failure, asyncio.CancelledError) else 1
    assert cli.main(["--dry-run", "--headless", "--task", "Fail after setup"]) == expected
    assert len(roots) == 1 and not roots[0].exists()
    if expected == 1:
        assert "scripted failure" in capsys.readouterr().err


@pytest.mark.parametrize(
    "override",
    [
        ["--provider", "openai"],
        ["--model", "test"],
        ["--workspace", "/unused"],
        ["--config", "/unused.json"],
    ],
)
def test_dry_run_rejects_live_options(capsys, override):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--dry-run", *override])
    assert stopped.value.code == 2
    assert "unrecognized arguments" not in capsys.readouterr().err


def test_default_provider_uses_openai_for_live_workspace(repo, monkeypatch, capsys):
    def unavailable_provider(*, model):
        assert model == "offline-test"
        raise RuntimeError("OpenAI unavailable in offline test")

    monkeypatch.setattr(cli, "OpenAIAPI", unavailable_provider)
    assert cli.main(real_provider_args(repo)[2:]) == 1
    assert "OpenAI unavailable in offline test" in capsys.readouterr().err

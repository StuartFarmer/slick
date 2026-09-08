import asyncio
import subprocess
import sys

from examples.coding_harness.__main__ import create_agent
from examples.coding_harness.demo import DemoProvider, create_demo


async def deny(request):
    return "deny"


def test_demo_repairs_a_real_failure(tmp_path):
    config = create_demo(tmp_path)
    events = []

    async def run():
        agent = await create_agent(
            DemoProvider(tmp_path), tmp_path, config, decide=deny, emit=events.append
        )
        return await agent.run("Fix the total calculation")

    result = asyncio.run(run())
    assert result.status == "verified"
    assert result.repairs == 1
    assert (tmp_path / "pricing.py").read_text() == "def total(values):\n    return sum(values)\n"
    finals = [
        event.data
        for event in events
        if event.kind == "verification" and not event.data["baseline"]
    ]
    assert [verification["passed"] for verification in finals] == [False, True]


def test_headless_cli_runs_without_textual_import():
    code = """
import builtins
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'textual':
        raise AssertionError('headless imported Textual')
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
from examples.coding_harness.__main__ import main
raise SystemExit(main(['--provider','demo','--headless','--task','Fix total']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "Verified" in result.stdout
    assert "1 repair" in result.stdout


def test_real_provider_requires_workspace_and_model():
    result = subprocess.run(
        [sys.executable, "-m", "examples.coding_harness", "--provider", "openai"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "--model" in result.stderr


def test_demo_does_not_inherit_commit_signing(tmp_path, monkeypatch):
    git_config = tmp_path / "gitconfig"
    git_config.write_text("[commit]\n    gpgsign = true\n[gpg]\n    program = /no/such/gpg\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(git_config))
    root = tmp_path / "demo"
    root.mkdir()
    config = create_demo(root)
    assert config.checks

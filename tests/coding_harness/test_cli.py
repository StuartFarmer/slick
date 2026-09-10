import asyncio
import subprocess
import sys

import pytest

from examples.coding_harness import __main__ as cli
from examples.coding_harness.agent import create_agent
from examples.coding_harness.config import HarnessConfig
from examples.coding_harness.demo import DemoProvider


def test_headless_demo_needs_no_textual_import():
    code = """
import sys
sys.modules['textual'] = None
from examples.coding_harness.__main__ import main
raise SystemExit(main(['--dry-run', '--headless', '--task', 'Fix the total calculation']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "Verified" in result.stdout


@pytest.mark.parametrize("extra", [[], ["--model", "model"], ["--workspace", "/tmp"]])
def test_live_run_requires_model_and_workspace(extra):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--headless", "--task", "Fix", *extra])
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--provider", "openai"],
        ["--model", "model"],
        ["--workspace", "/tmp"],
        ["--config", "config.json"],
        ["--resume", "saved.json"],
    ],
)
def test_demo_rejects_live_options(extra):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--dry-run", *extra])
    assert caught.value.code == 2


def test_new_workspace_can_be_created_and_edited(tmp_path):
    root = tmp_path / "new"
    agent = asyncio.run(create_agent(DemoProvider(root), root, HarnessConfig()))
    agent.workspace.create_file("hello.py", "print('hello')\n")
    assert (root / ".git").is_dir()
    assert agent.workspace.read_file("hello.py")["text"] == "print('hello')\n"

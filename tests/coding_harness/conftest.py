import asyncio
import subprocess
from pathlib import Path

import pytest

from slick import prompts


@pytest.fixture(autouse=True)
def harness_templates(monkeypatch):
    monkeypatch.setattr(
        prompts, "TEMPLATE_ROOT", Path(__file__).resolve().parents[2] / "examples" / "prompts"
    )


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("__pycache__/\nignored/\n")
    (tmp_path / "sample.py").write_text("value = 1\n")
    (tmp_path / "unicode.txt").write_text("café\nsecond\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Harness Test",
            "-c",
            "user.email=harness@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return tmp_path


@pytest.fixture
def workspace(repo):
    from examples.coding_harness.workspace import Workspace

    async def deny(request):
        return "deny"

    instance = Workspace(repo, checks=[], decide=deny)
    asyncio.run(instance.initialize())
    return instance

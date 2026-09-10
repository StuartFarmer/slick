"""Tool entry points enforce workspace policy without an agent loop."""

import asyncio
import json
import sys

import pytest

from slick import ToolError
from slick.tools import prepare_tools


def test_direct_tools_cannot_escape_workspace_or_execute_denied_commands(workspace, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("private")
    (workspace.root / "outside").symlink_to(outside)
    tools = prepare_tools([workspace.read_file, workspace.create_file, workspace.run_command])
    with pytest.raises(ToolError):
        tools["read_file"].invoke({"path": "outside"})
    with pytest.raises(ToolError):
        tools["create_file"].invoke({"path": "../escape.txt", "content": "bad"})
    result = json.loads(
        asyncio.run(
            tools["run_command"].ainvoke(
                {"argv": [sys.executable, "-c", "open('effect', 'w').write('bad')"]}
            )
        )
    )
    assert result["exit_code"] is None
    assert not (workspace.root / "effect").exists()
    assert outside.read_text() == "private"

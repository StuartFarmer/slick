import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys

import pytest

from examples.coding_harness.state import Check
from examples.coding_harness.workspace import Workspace
from slick import ToolError, tool
from slick.tools import prepare_tools


def test_all_methods_prepare(workspace):
    names = [
        "list_files",
        "search",
        "read_file",
        "edit_file",
        "create_file",
        "run_command",
        "git_diff",
    ]
    assert set(prepare_tools([getattr(workspace, name) for name in names])) == set(names)


def test_slice_digest_unicode_and_newlines(workspace):
    path = workspace.root / "unicode.txt"
    path.write_bytes("café\r\nsecond\r\n".encode())
    result = json.loads(tool(workspace.read_file).invoke({"path": "unicode.txt", "max_lines": 1}))
    assert result["text"] == "café\r\n"
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["truncated"]


def test_stale_edit_does_not_overwrite_external_changes(workspace):
    target = workspace.root / "sample.py"
    original = workspace.read_file("sample.py")
    target.write_text("value = 2\n")
    with pytest.raises(ToolError):
        tool(workspace.edit_file).invoke(
            {
                "path": "sample.py",
                "old": "value = 1",
                "new": "value = 3",
                "expected_sha256": original.sha256,
            }
        )
    assert target.read_text() == "value = 2\n"
    assert not workspace.edit_ledger


def test_edit_preserves_bytes_and_permissions(workspace):
    path = workspace.root / "sample.py"
    path.write_bytes(b"value = 1\r\n")
    path.chmod(0o751)
    original = workspace.read_file("sample.py")
    result = workspace.edit_file("sample.py", "1", "2", original.sha256)
    assert path.read_bytes() == b"value = 2\r\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o751
    assert "+value = 2" in result.diff
    assert workspace.edit_ledger[0]["before_sha256"] == original.sha256
    assert asyncio.run(workspace.changed_paths()) == ["sample.py"]


@pytest.mark.parametrize("path", ["../escape", "/tmp/escape", ".git/config", "dir/../sample.py"])
def test_disallowed_paths(workspace, path):
    with pytest.raises(ToolError):
        tool(workspace.read_file).invoke({"path": path})
    with pytest.raises(ToolError):
        tool(workspace.create_file).invoke({"path": path, "content": "bad"})


def test_symlinks_and_nonregular_files(workspace):
    (workspace.root / "link").symlink_to(workspace.root / "sample.py")
    (workspace.root / "directory").mkdir()
    for path in ["link", "directory"]:
        with pytest.raises(ValueError):
            workspace.read_file(path)
    assert "link" not in asyncio.run(workspace.list_files()).paths


def test_invalid_text_and_read_bounds(workspace):
    for content in [b"\xff", b"a\0b", b"x" * (1024 * 1024 + 1)]:
        (workspace.root / "data").write_bytes(content)
        with pytest.raises(ValueError):
            workspace.read_file("data")
    for start, maximum in [(0, 2), (1, 0), (True, 2)]:
        with pytest.raises(ValueError):
            workspace.read_file("sample.py", start, maximum)
    (workspace.root / "data").write_text("x" * 30000)
    result = workspace.read_file("data")
    assert len(result.text) == 20000 and result.truncated


def test_ambiguous_and_empty_replacement(workspace):
    original = workspace.read_file("sample.py")
    for old in ["", " ", "missing"]:
        with pytest.raises(ValueError):
            workspace.edit_file("sample.py", old, "replacement", original.sha256)


def test_exclusive_create_and_existing_parent(workspace):
    result = json.loads(
        tool(workspace.create_file).invoke({"path": "new.py", "content": "hello\n"})
    )
    assert result["path"] == "new.py"
    for name in ["new.py", "missing/new.py"]:
        with pytest.raises((ValueError, FileNotFoundError)):
            workspace.create_file(name, "overwrite")
    assert (workspace.root / "new.py").read_text() == "hello\n"


def test_catalog_search_and_diff(workspace):
    (workspace.root / "ignored").mkdir()
    (workspace.root / "ignored" / "secret").write_text("café")
    (workspace.root / "new.txt").write_text("café\n")
    files = asyncio.run(workspace.list_files("*.txt"))
    assert files.paths == ["new.txt", "unicode.txt"]
    hits = asyncio.run(workspace.search("café"))
    assert {hit.path for hit in hits.hits} == {"unicode.txt", "new.txt"}
    assert asyncio.run(workspace.search("no_such_text")).hits == []
    (workspace.root / "sample.py").write_text("value = 2\n")
    subprocess.run(["git", "-C", str(workspace.root), "add", "sample.py"], check=True)
    (workspace.root / "sample.py").write_text("value = 3\n")
    diff = asyncio.run(workspace.git_diff())
    assert "+value = 2" in diff.staged and "+value = 3" in diff.unstaged
    assert diff.untracked == ["new.txt"]


def test_fingerprint_detects_same_mtime_and_missing_tracked(workspace):
    before = asyncio.run(workspace.fingerprint())
    path = workspace.root / "sample.py"
    old = path.stat()
    path.write_text("value = 9\n")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert asyncio.run(workspace.fingerprint()) != before
    path.unlink()
    assert asyncio.run(workspace.changed_paths()) == ["sample.py"]


def test_decisions_exact_argv_and_configured_checks(repo):
    requests = []

    async def decide(request):
        requests.append(request)
        return "session" if len(requests) == 1 else "deny"

    approved = [sys.executable, "-c", "print('allowed')"]
    check = Check(name="unit", argv=[sys.executable, "-c", "print('check')"])
    ws = Workspace(repo, checks=[check], decide=decide, command_timeout=5)

    async def scenario():
        await ws.initialize()
        assert (await ws.run_command(approved)).exit_code == 0
        assert (await ws.run_command(approved)).exit_code == 0
        assert (await ws.run_command(check.argv)).exit_code == 0
        assert (await ws.run_command([*approved, "extra"])).exit_code is None
        assert (await ws.run_command(["git", "status"])).exit_code is None

    asyncio.run(scenario())
    assert len(requests) == 3
    assert requests[0].cwd == str(repo.resolve()) and requests[0].timeout == 5


def test_cancel_decision_has_no_effect(workspace):
    async def scenario():
        entered = asyncio.Event()

        async def decide(request):
            entered.set()
            await asyncio.Event().wait()

        workspace.decide = decide
        task = asyncio.create_task(
            workspace.run_command([sys.executable, "-c", "open('effect','w').write('bad')"])
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert not (workspace.root / "effect").exists()


def test_catalog_and_search_truncate(workspace):
    for index in range(205):
        (workspace.root / f"file{index:03}.txt").write_text("match me\n")
    listed = asyncio.run(workspace.list_files("file*"))
    searched = asyncio.run(workspace.search("match me"))
    assert len(listed.paths) == 200 and listed.truncated
    assert len(searched.hits) == 200 and searched.truncated


def test_permissions_do_not_get_bypassed_by_replace(workspace):
    path = workspace.root / "sample.py"
    before = workspace.read_file("sample.py")
    path.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            workspace.edit_file("sample.py", "1", "2", before.sha256)
        assert path.read_text() == "value = 1\n"
        assert not list(workspace.root.glob(".coding-harness-*"))
    finally:
        path.chmod(0o644)


def test_unborn_repo_and_subdirectory_selection(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    async def deny(request):
        return "deny"

    ws = Workspace(tmp_path, checks=[], decide=deny)
    asyncio.run(ws.initialize())
    assert ws.head is None
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    with pytest.raises(ValueError, match="root"):
        asyncio.run(Workspace(subdir, checks=[], decide=deny).initialize())
    with pytest.raises(ValueError, match="root"):
        asyncio.run(Workspace(subdir, checks=[], decide=deny).initialize(create=True))
    assert not (subdir / ".git").exists()


@pytest.mark.parametrize("kind", ["bare", "broken"])
def test_new_workspace_does_not_reinitialize_existing_git_metadata(tmp_path, kind):
    if kind == "bare":
        subprocess.run(["git", "init", "--bare", "-q", str(tmp_path)], check=True)
        marker = tmp_path / "HEAD"
    else:
        marker = tmp_path / ".git"
        marker.write_text("gitdir: /missing/worktree/metadata\n")
    before = marker.read_bytes()

    async def deny(request):
        return "deny"

    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(Workspace(tmp_path, checks=[], decide=deny).initialize(create=True))
    assert marker.read_bytes() == before
    assert not (tmp_path / ".git").is_dir()


def test_workspace_reinitialization_requires_existing_repository(tmp_path):
    async def deny(request):
        return "deny"

    with pytest.raises(RuntimeError, match="Git inspection failed"):
        asyncio.run(Workspace(tmp_path, checks=[], decide=deny).initialize())
    assert not (tmp_path / ".git").exists()


def test_session_approval_is_not_shared_between_workspaces(workspace, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    count = []

    async def decide(request):
        count.append(request.cwd)
        return "session" if len(count) == 1 else "deny"

    workspace.decide = decide
    argv = [sys.executable, "-c", "print('cwd check')"]

    async def scenario():
        assert (await workspace.run_command(argv)).exit_code == 0
        workspace.root = other.resolve()
        assert (await workspace.run_command(argv)).exit_code is None

    asyncio.run(scenario())
    assert len(count) == 2 and count[0] != count[1]


def test_catalog_rejects_ambiguous_replacement_characters(workspace):
    # Command decoding replaces invalid UTF-8 bytes with U+FFFD. Refusing that
    # character also rejects a literal U+FFFD name, instead of silently hashing
    # an undecodable filename as a missing tracked file on Linux.
    (workspace.root / "ambiguous-\ufffd.txt").write_text("contents")
    with pytest.raises(RuntimeError, match="UTF-8"):
        asyncio.run(workspace.fingerprint())


def test_overlapping_matches_are_ambiguous(workspace):
    path = workspace.root / "sample.py"
    path.write_text("aaa")
    original = workspace.read_file("sample.py")
    with pytest.raises(ValueError, match="exactly once"):
        workspace.edit_file("sample.py", "aa", "b", original.sha256)
    assert path.read_text() == "aaa"

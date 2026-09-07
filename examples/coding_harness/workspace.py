"""Precise local file operations and explicit command decisions."""

import asyncio
import difflib
import fnmatch
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from .process import run_process, validate_command
from .state import (
    Check,
    CommandDecision,
    CommandRequest,
    CommandResult,
    DiffResult,
    EditResult,
    FileList,
    FileSlice,
    SearchHit,
    SearchResult,
)

MAX_FILE_BYTES = 1024 * 1024
MAX_TEXT = 20000
CATALOG_LIMIT = 10 * 1024 * 1024


class Workspace:
    def __init__(
        self,
        root: Path,
        *,
        checks: list[Check],
        decide: Callable[[CommandRequest], Awaitable[CommandDecision]],
        command_timeout: int = 120,
    ):
        validate_command(["git"], command_timeout)
        self.root = Path(root).resolve()
        self.checks = list(checks)
        self.decide = decide
        self.command_timeout = command_timeout
        self.configured_commands = {(str(self.root), tuple(check.argv)) for check in checks}
        self.allowed_commands = set(self.configured_commands)
        self.head = None
        self.branch = ""
        self.initial_status = ""
        self.initial_digests = {}
        self.edit_ledger = []

    async def initialize(self) -> None:
        if not self.root.is_dir():
            raise ValueError("Workspace root must be an existing directory")
        for executable in ("git", "rg"):
            if shutil.which(executable) is None:
                raise RuntimeError(f"{executable} is required")
        root = await self._git(["rev-parse", "--show-toplevel"])
        if Path(root.stdout.strip()).resolve() != self.root:
            raise ValueError("Select the Git worktree root, not a subdirectory")
        head = await self._git(["rev-parse", "--verify", "HEAD"], allow_failure=True)
        self.head = head.stdout.strip() if head.exit_code == 0 else None
        branch = await self._git(["symbolic-ref", "--short", "-q", "HEAD"], allow_failure=True)
        self.branch = branch.stdout.strip()
        status = await self._git(["status", "--short"], output_limit=MAX_TEXT)
        self.initial_status = status.stdout + ("\n[truncated]" if status.truncated else "")
        self.initial_digests = await self._digests()

    async def _git(self, args, *, output_limit=CATALOG_LIMIT, allow_failure=False):
        result = await run_process(
            ["git", "-c", "core.quotePath=false", *args],
            self.root,
            self.command_timeout,
            output_limit=output_limit,
        )
        if not allow_failure and (result.exit_code != 0 or result.timed_out):
            raise RuntimeError(f"Git inspection failed: {result.stderr or result.stdout}")
        return result

    def _path(self, path: str, *, missing=False) -> Path:
        relative = Path(path)
        if (
            not path
            or relative.is_absolute()
            or ".." in relative.parts
            or any(part.lower() == ".git" for part in relative.parts)
        ):
            raise ValueError("Path must be relative and cannot traverse or access .git")
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Symlink paths are not permitted")
        if current == self.root or not current.resolve().is_relative_to(self.root):
            raise ValueError("Path must identify a file inside the workspace")
        if current.exists():
            if not current.is_file():
                raise ValueError("Path must identify a regular file")
        elif not missing:
            raise FileNotFoundError(path)
        return current

    async def _catalog(self) -> list[str]:
        result = await self._git(["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        if result.truncated:
            raise RuntimeError("Git file catalog exceeded its size limit; scan incomplete")
        if "\ufffd" in result.stdout:
            raise RuntimeError(
                "Git catalog contains undecodable UTF-8 or U+FFFD filenames; "
                "rename them before scanning"
            )
        paths = []
        for path in sorted(set(result.stdout.split("\0")) - {""}):
            try:
                self._path(path, missing=True)
            except ValueError:
                continue
            paths.append(path)
        return paths

    def _hash_files(self, paths):
        digests = {}
        for path in paths:
            target = self._path(path, missing=True)
            try:
                with target.open("rb") as handle:
                    digest = hashlib.sha256()
                    while chunk := handle.read(65536):
                        digest.update(chunk)
                    digests[path] = digest.hexdigest()
            except FileNotFoundError:
                digests[path] = "missing"
        return digests

    async def _digests(self):
        paths = await self._catalog()
        task = asyncio.create_task(asyncio.to_thread(self._hash_files, paths))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Finish the read-only scan before allowing another task to begin.
            try:
                await asyncio.shield(task)
            finally:
                raise

    async def fingerprint(self) -> str:
        digests = await self._digests()
        encoded = json.dumps(digests, sort_keys=True, ensure_ascii=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    async def changed_paths(self) -> list[str]:
        current = await self._digests()
        return sorted(
            path
            for path in current.keys() | self.initial_digests.keys()
            if current.get(path) != self.initial_digests.get(path)
        )

    async def list_files(self, pattern: str = "*") -> FileList:
        """List tracked and nonignored untracked files matching a glob."""
        paths = [path for path in await self._catalog() if fnmatch.fnmatchcase(path, pattern)]
        return FileList(paths=paths[:200], truncated=len(paths) > 200)

    async def search(self, query: str, glob: str = "*") -> SearchResult:
        """Search permitted workspace files with ripgrep and return bounded hits."""
        paths = [path for path in await self._catalog() if (self.root / path).is_file()]
        hits, used, truncated = [], 0, False
        for index in range(0, len(paths), 100):
            result = await run_process(
                [
                    "rg",
                    "--json",
                    "--no-follow",
                    "--glob",
                    glob,
                    "--",
                    query,
                    *paths[index : index + 100],
                ],
                self.root,
                self.command_timeout,
                output_limit=1024 * 1024,
            )
            if result.exit_code not in (0, 1) or result.timed_out:
                raise RuntimeError(f"Search failed: {result.stderr}")
            truncated |= result.truncated
            for line in result.stdout.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if result.truncated:
                        break
                    raise
                if record["type"] != "match":
                    continue
                data = record["data"]
                if "text" not in data["path"] or "text" not in data["lines"]:
                    continue
                path, text = data["path"]["text"], data["lines"]["text"]
                self._path(path)
                if len(hits) == 200 or used >= MAX_TEXT:
                    return SearchResult(hits=hits, truncated=True)
                bounded = text[: MAX_TEXT - used]
                truncated |= len(bounded) < len(text)
                hits.append(SearchHit(path=path, line=data["line_number"], text=bounded))
                used += len(bounded)
        return SearchResult(hits=hits, truncated=truncated)

    def _read(self, path):
        target = self._path(path)
        with target.open("rb") as handle:
            content = handle.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("Text files must be at most 1 MiB")
        if b"\0" in content:
            raise ValueError("Binary files are not supported")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must contain UTF-8") from exc
        return target, content, text

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> FileSlice:
        """Read a UTF-8 line slice with the full file digest needed for editing."""
        if any(type(value) is not int or value <= 0 for value in (start_line, max_lines)):
            raise ValueError("Line bounds must be positive integers")
        _, content, text = self._read(path)
        lines = text.splitlines(keepends=True)
        selected = "".join(lines[start_line - 1 : start_line - 1 + max_lines])
        bounded = selected[:MAX_TEXT]
        return FileSlice(
            path=path,
            text=bounded,
            start_line=start_line,
            end_line=start_line + len(bounded.splitlines()) - 1,
            sha256=hashlib.sha256(content).hexdigest(),
            truncated=len(selected) > MAX_TEXT or start_line - 1 + max_lines < len(lines),
        )

    def _edit_result(self, path, before, after):
        before_digest = hashlib.sha256(before).hexdigest() if before is not None else None
        after_digest = hashlib.sha256(after).hexdigest()
        diff = "".join(
            difflib.unified_diff(
                (before or b"").decode("utf-8").splitlines(keepends=True),
                after.decode("utf-8").splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        if len(diff) > MAX_TEXT:
            diff = diff[: MAX_TEXT - 13] + "\n[truncated]\n"
        self.edit_ledger.append(
            {
                "path": path,
                "before_sha256": before_digest,
                "after_sha256": after_digest,
                "diff": diff,
            }
        )
        return EditResult(path=path, sha256=after_digest, diff=diff)

    def edit_file(self, path: str, old: str, new: str, expected_sha256: str) -> EditResult:
        """Replace exactly one nonempty text occurrence if the full digest matches."""
        target, content, text = self._read(path)
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("File changed since read; read again before editing")
        if not old or text.find(old) < 0 or text.find(old) != text.rfind(old):
            raise ValueError("old must be nonempty and match exactly once")
        after = text.replace(old, new, 1).encode("utf-8")
        if len(after) > MAX_FILE_BYTES or b"\0" in after:
            raise ValueError("Replacement must be UTF-8 text of at most 1 MiB")
        mode = stat.S_IMODE(target.stat().st_mode)
        if not os.access(target, os.W_OK):
            raise PermissionError(f"File is not writable: {path}")
        fd, staged = tempfile.mkstemp(prefix=".coding-harness-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(after)
                os.fchmod(handle.fileno(), mode)
            # Recheck immediately before replace to catch ordinary concurrent edits.
            current, latest, _ = self._read(path)
            if current != target or latest != content:
                raise ValueError("File changed while staging replacement")
            os.replace(staged, target)
        finally:
            if os.path.exists(staged):
                os.unlink(staged)
        return self._edit_result(path, content, after)

    def create_file(self, path: str, content: str) -> EditResult:
        """Exclusively create a UTF-8 file within an existing permitted parent."""
        target = self._path(path, missing=True)
        if target.exists():
            raise ValueError("File already exists")
        after = content.encode("utf-8")
        if len(after) > MAX_FILE_BYTES or b"\0" in after:
            raise ValueError("Content must be UTF-8 text of at most 1 MiB")
        with target.open("xb") as handle:
            try:
                handle.write(after)
            except BaseException:
                target.unlink()
                raise
        return self._edit_result(path, None, after)

    async def run_command(self, argv: list[str], timeout: int = 120) -> CommandResult:
        """Run an exact argv locally after approval; nonzero exits are observations."""
        validate_command(argv, timeout)
        timeout = min(timeout, self.command_timeout)
        key = (str(self.root), tuple(argv))
        if key not in self.allowed_commands:
            decision = await self.decide(
                CommandRequest(argv=list(argv), cwd=str(self.root), timeout=timeout)
            )
            if decision not in ("once", "session"):
                return CommandResult(
                    argv=argv,
                    exit_code=None,
                    stdout="",
                    stderr="Command denied; nothing was executed.",
                    timed_out=False,
                    truncated=False,
                )
            if decision == "session":
                self.allowed_commands.add(key)
        return await run_process(argv, self.root, timeout)

    async def git_diff(self) -> DiffResult:
        """Inspect current staged/unstaged changes, including pre-existing changes."""
        args = ["diff", "--no-ext-diff", "--no-textconv"]
        unstaged = await self._git(args, output_limit=MAX_TEXT)
        staged = await self._git([*args, "--cached"], output_limit=MAX_TEXT)
        untracked = await self._git(["ls-files", "-z", "--others", "--exclude-standard"])
        if untracked.truncated:
            raise RuntimeError("Untracked catalog exceeded its size limit")
        if "\ufffd" in untracked.stdout:
            raise RuntimeError(
                "Git catalog contains undecodable UTF-8 or U+FFFD filenames; "
                "rename them before scanning"
            )
        paths = []
        for path in untracked.stdout.split("\0"):
            if not path:
                continue
            try:
                self._path(path)
            except (ValueError, FileNotFoundError):
                continue
            paths.append(path)
        return DiffResult(
            unstaged=unstaged.stdout,
            staged=staged.stdout,
            untracked=paths[:200],
            truncated=unstaged.truncated or staged.truncated or len(paths) > 200,
        )

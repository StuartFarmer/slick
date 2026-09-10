"""A conversation on disk, or shortened into a summary. Neither replays tools."""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from slick import Prompt

from .config import HarnessConfig

MAX_SESSION_BYTES = 20 * 1024 * 1024


class Message(BaseModel, extra="forbid"):
    role: Literal["user", "assistant", "tool"]
    text: str
    calls: list[dict] | None = None
    id: str | None = None
    error: bool | None = None


class SavedSession(BaseModel, extra="forbid"):
    version: Literal[2]
    provider: Literal["openai", "anthropic", "litellm", "demo"]
    model: str
    root: str
    fingerprint: str
    config: HarnessConfig
    messages: list[Message] = Field(default_factory=list)


def save(path, agent):
    if agent.running:
        raise ValueError("Save requires an idle agent")
    path = Path(path).expanduser()
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"Session destination already exists: {path}")
    path = path.resolve()
    if path.is_relative_to(agent.workspace.root):
        raise ValueError("Save the session outside the worktree")
    saved = SavedSession(
        version=2,
        **agent.provider.identity(),
        root=str(agent.workspace.root),
        fingerprint=agent.fingerprint,
        config=agent.config,
        messages=agent.messages,
    )
    payload = saved.model_dump_json(exclude_none=True, indent=2).encode("utf-8")
    if len(payload) > MAX_SESSION_BYTES:
        raise ValueError("Session exceeds the 20 MiB save limit")
    # Publish without overwriting another file; NamedTemporaryFile is mode 0600.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load(path):
    with Path(path).expanduser().open("rb") as handle:
        payload = handle.read(MAX_SESSION_BYTES + 1)
    if len(payload) > MAX_SESSION_BYTES:
        raise ValueError("Session exceeds the 20 MiB load limit")
    data = json.loads(payload)
    if not isinstance(data, dict) or data.get("version") != 2:
        raise ValueError("Unsupported session format; start a new conversation")
    return SavedSession.model_validate(data)


async def compact(provider, messages, limit, *, before_request=None):
    # Keep the last complete assistant/tool exchange, so recent observations stay exact.
    starts = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    cut = starts[-1] if starts else 0
    if cut == 0:
        raise ValueError("Not enough conversation to compact")
    prompt = Prompt("coding_harness/compact.j2")(history=messages[:cut])
    if len(prompt) > limit:
        raise ValueError("Summary input exceeds the context limit; start a new conversation")
    if before_request is not None:
        before_request()
    summary, calls = await provider.acall(prompt, tools=[])
    if calls or not summary.strip():
        raise ValueError("Compaction requires a nonempty summary without tool calls")
    replacement = [{"role": "user", "text": "Earlier conversation summary:\n" + summary}]
    replacement.extend(messages[cut:])
    if len(json.dumps(replacement, ensure_ascii=False)) > limit:
        raise ValueError("Summary exceeds the context limit")
    return replacement

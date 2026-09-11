"""Persistent JSON messages over two-ended, local SQLite channels."""

import asyncio
import json
import math
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter, ValidationError


class Inbox:
    """Send synchronously; await incoming messages without blocking on an empty queue.

    Share the database path and peer address with another process or adapter.
    recv() consumes messages; request() retains replies for recovery.
    """

    def __init__(self, path: str | Path, *, poll_interval: float = 0.1):
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        if str(path) in {"", ":memory:"}:
            raise ValueError("Inbox requires a persistent database path")
        self.path = Path(path).resolve()
        self.poll_interval = poll_interval
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS channels (id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS requests (
                    channel TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS exchanges (
                    channel TEXT PRIMARY KEY,
                    key TEXT UNIQUE,
                    payload TEXT NOT NULL,
                    response TEXT,
                    rejected_response TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    channel TEXT NOT NULL,
                    recipient INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_recipient
                    ON messages (channel, recipient, id);
            """)

    @contextmanager
    def _connect(self, *, timeout: float = 5):
        with closing(sqlite3.connect(self.path, timeout=timeout)) as connection, connection:
            yield connection

    @staticmethod
    def _address(channel: str) -> tuple[str, int]:
        if not isinstance(channel, str):
            raise ValueError("Invalid channel address")
        identifier, separator, side = channel.partition(":")
        if not identifier or not separator or side not in {"0", "1"}:
            raise ValueError("Invalid channel address")
        return identifier, int(side)

    def new_channel(self) -> str:
        """Create a channel and return its first endpoint's address."""
        identifier = uuid4().hex
        with self._connect() as connection:
            connection.execute("INSERT INTO channels (id) VALUES (?)", (identifier,))
        return f"{identifier}:0"

    def peer(self, channel: str) -> str:
        """Return the opposite endpoint's address to give to the other participant."""
        identifier, side = self._address(channel)
        return f"{identifier}:{1 - side}"

    @staticmethod
    def _encode(message: Any) -> str:
        if isinstance(message, BaseModel):
            message = message.model_dump(mode="json")
        return json.dumps(message, allow_nan=False)

    @staticmethod
    def _same(first: str, second: str) -> bool:
        return json.dumps(json.loads(first), sort_keys=True) == json.dumps(
            json.loads(second), sort_keys=True
        )

    async def request(
        self, message: Any, *, output_type: Any = None, key: str | None = None
    ) -> Any:
        """Publish or reconnect to a request and return its retained reply.

        A stable key reuses the same request across restarts. Without one, create a
        fresh request. Invalid replies reopen the request and raise ValidationError.
        """
        if key is not None and (not isinstance(key, str) or not key.strip()):
            raise ValueError("Request key must be a nonblank string")
        adapter = TypeAdapter(output_type) if output_type is not None else None
        payload = self._encode(message)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT channel, payload FROM exchanges WHERE key = ?", (key,)
            ).fetchone()
            if existing is not None:
                identifier, previous = existing
                if not self._same(previous, payload):
                    raise ValueError("Request key already belongs to different message content")
            else:
                identifier = uuid4().hex
                connection.execute("INSERT INTO channels (id) VALUES (?)", (identifier,))
                connection.execute(
                    "INSERT INTO exchanges (channel, key, payload) VALUES (?, ?, ?)",
                    (identifier, key, payload),
                )
                connection.execute(
                    "INSERT INTO messages (channel, recipient, payload) VALUES (?, 1, ?)",
                    (identifier, payload),
                )
                connection.execute(
                    "INSERT INTO requests (channel, payload) VALUES (?, ?)", (identifier, payload)
                )
        response = await self._response(identifier)
        try:
            return adapter.validate_json(response) if adapter is not None else json.loads(response)
        except ValidationError as error:
            with self._connect() as connection:
                changed = connection.execute(
                    "UPDATE exchanges SET response = NULL, rejected_response = ?, error = ? "
                    "WHERE channel = ? AND response = ?",
                    (response, str(error), identifier, response),
                )
                if changed.rowcount:
                    connection.execute(
                        "INSERT OR REPLACE INTO requests (channel, payload) VALUES (?, ?)",
                        (identifier, payload),
                    )
                    connection.execute(
                        "DELETE FROM messages WHERE channel = ? AND recipient = 0", (identifier,)
                    )
            raise

    async def _response(self, identifier: str) -> str:
        while True:
            try:
                with self._connect(timeout=0) as connection:
                    row = connection.execute(
                        "SELECT response FROM exchanges WHERE channel = ?", (identifier,)
                    ).fetchone()
                    if row[0] is not None:
                        return row[0]
            except sqlite3.OperationalError as error:
                if str(error) not in {"database is locked", "database table is locked"}:
                    raise
            await asyncio.sleep(self.poll_interval)

    def pending(self) -> list[tuple[str, Any]]:
        """List unanswered requests as (reply endpoint, decoded message), oldest first.

        This is a snapshot, not a claim: multiple readers can see the same request.
        Ordinary channel messages and replies are excluded.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT channel, payload FROM requests ORDER BY rowid"
            ).fetchall()
        return [(f"{identifier}:1", json.loads(payload)) for identifier, payload in rows]

    def send(self, channel: str, message: Any) -> None:
        """Queue a JSON value or Pydantic model for the opposite endpoint."""
        identifier, side = self._address(channel)
        payload = self._encode(message)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if side == 1:
                exchange = connection.execute(
                    "SELECT response FROM exchanges WHERE channel = ?", (identifier,)
                ).fetchone()
                if exchange is not None and exchange[0] is not None:
                    if self._same(exchange[0], payload):
                        return
                    raise ValueError("Request already has a different response")
                connection.execute(
                    "UPDATE exchanges SET response = ? WHERE channel = ?", (payload, identifier)
                )
            inserted = connection.execute(
                "INSERT INTO messages (channel, recipient, payload) "
                "SELECT id, ?, ? FROM channels WHERE id = ?",
                (1 - side, payload, identifier),
            )
            if not inserted.rowcount:
                raise ValueError(f"Unknown channel: {channel}")
            if side == 1:
                connection.execute("DELETE FROM requests WHERE channel = ?", (identifier,))

    async def recv(self, channel: str) -> Any:
        """Wait for and consume this endpoint's oldest message, decoded from JSON.

        Cancellation while waiting leaves queued messages intact. Apply a deadline
        with asyncio.wait_for when needed.
        """
        identifier, side = self._address(channel)
        while True:
            try:
                # No await between claiming and returning: cancellation cannot lose a claim.
                with self._connect(timeout=0) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if not connection.execute(
                        "SELECT 1 FROM channels WHERE id = ?", (identifier,)
                    ).fetchone():
                        raise ValueError(f"Unknown channel: {channel}")
                    row = connection.execute(
                        "SELECT id, payload FROM messages WHERE channel = ? AND recipient = ? "
                        "ORDER BY id LIMIT 1",
                        (identifier, side),
                    ).fetchone()
                    if row is not None:
                        message = json.loads(row[1])
                        connection.execute("DELETE FROM messages WHERE id = ?", (row[0],))
                        return message
            except sqlite3.OperationalError as error:
                if str(error) not in {"database is locked", "database table is locked"}:
                    raise
            # ponytail: polling adds up to one interval of latency; use notifications if needed.
            await asyncio.sleep(self.poll_interval)

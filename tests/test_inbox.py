"""Persistent, two-ended channels without review-specific messages."""

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import BaseModel, ValidationError

from slick import Inbox


def test_duplex_fifo_persistence_and_cancellation(tmp_path):
    class Message(BaseModel):
        text: str

    async def scenario():
        path = tmp_path / "inbox.db"
        inbox = Inbox(path, poll_interval=0.01)
        channel = inbox.new_channel()
        peer = inbox.peer(channel)
        assert inbox.peer(peer) == channel
        inbox.send(channel, Message(text="draft"))
        inbox.send(channel, None)
        inbox.send(channel, [1, "two", False])

        # An endpoint never receives its own outgoing messages.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(inbox.recv(channel), 0.03)

        reviewer = Inbox(path, poll_interval=0.01)
        assert await reviewer.recv(peer) == {"text": "draft"}
        assert await reviewer.recv(peer) is None
        assert await reviewer.recv(peer) == [1, "two", False]
        reviewer.send(peer, {"response": "looks good"})
        assert await inbox.recv(channel) == {"response": "looks good"}

        waiting = asyncio.create_task(inbox.recv(channel))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        reviewer.send(peer, "after cancellation")
        assert await Inbox(path).recv(channel) == "after cancellation"

    asyncio.run(asyncio.wait_for(scenario(), 2))


def test_competing_receivers_claim_each_message_once(tmp_path):
    path = tmp_path / "inbox.db"
    inbox = Inbox(path, poll_interval=0.001)
    channel = inbox.new_channel()
    other = inbox.new_channel()
    inbox.send(other, "separate channel")
    for number in range(20):
        inbox.send(channel, number)

    def consume():
        async def scenario():
            return [await inbox.recv(inbox.peer(channel)) for _ in range(10)]

        return asyncio.run(asyncio.wait_for(scenario(), 3))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(consume)
        second = pool.submit(consume)
        assert sorted(first.result() + second.result()) == list(range(20))
    assert asyncio.run(inbox.recv(inbox.peer(other))) == "separate channel"


def test_database_lock_does_not_block_waiting_coroutine(tmp_path):
    async def scenario():
        inbox = Inbox(tmp_path / "inbox.db", poll_interval=0.001)
        channel = inbox.new_channel()
        inbox.send(inbox.peer(channel), "queued")
        connection = sqlite3.connect(inbox.path)
        try:
            connection.execute("BEGIN EXCLUSIVE")
            waiting = asyncio.create_task(inbox.recv(channel))
            await asyncio.sleep(0.02)
            assert not waiting.done()
            connection.rollback()
            assert await waiting == "queued"
        finally:
            connection.close()

    asyncio.run(asyncio.wait_for(scenario(), 1))


def test_invalid_addresses_and_messages_fail_explicitly(tmp_path):
    inbox = Inbox(tmp_path / "inbox.db")
    channel = inbox.new_channel()
    for invalid in ("bad", "missing:0", channel[:-1] + "2"):
        with pytest.raises(ValueError, match="channel"):
            inbox.send(invalid, "message")
        with pytest.raises(ValueError, match="channel"):
            asyncio.run(inbox.recv(invalid))
    for invalid in (object(), float("nan")):
        with pytest.raises((ValueError, TypeError)):
            inbox.send(channel, invalid)
    inbox.send(channel, "still usable")
    assert asyncio.run(inbox.recv(inbox.peer(channel))) == "still usable"
    for interval in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="poll_interval"):
            Inbox(tmp_path / "inbox.db", poll_interval=interval)


def test_requests_are_discoverable_until_replied_to(tmp_path):
    class Reply(BaseModel):
        accepted: bool

    async def scenario():
        inbox = Inbox(tmp_path / "inbox.db", poll_interval=0.001)
        reviewer = Inbox(inbox.path)
        ordinary = inbox.new_channel()
        inbox.send(ordinary, "not a request")
        assert reviewer.pending() == []

        waiting = asyncio.create_task(inbox.request({"draft": "one"}, output_type=Reply))
        await asyncio.sleep(0)
        pending = reviewer.pending()
        assert len(pending) == 1
        channel, draft = pending[0]
        assert draft == {"draft": "one"}
        assert Inbox(inbox.path).pending() == pending  # Listing does not consume or claim.
        assert await reviewer.recv(channel) == draft
        assert reviewer.pending() == pending  # Reading the draft does not answer it.
        with pytest.raises(TypeError):
            reviewer.send(channel, object())
        assert reviewer.pending() == pending
        reviewer.send(channel, {"accepted": True})
        assert reviewer.pending() == []  # Hidden as soon as a reply is committed.
        assert await waiting == Reply(accepted=True)

        for response in (None, [1, "two"], {"anything": False}):
            waiting = asyncio.create_task(inbox.request("raw"))
            await asyncio.sleep(0)
            channel, _ = reviewer.pending()[0]
            reviewer.send(channel, response)
            assert await waiting == response

        waiting = asyncio.create_task(inbox.request("typed", output_type=Reply))
        await asyncio.sleep(0)
        reviewer.send(reviewer.pending()[0][0], {"wrong": "shape"})
        with pytest.raises(ValidationError):
            await waiting
        channel, _ = reviewer.pending()[0]
        reviewer.send(channel, {"accepted": False})
        assert reviewer.pending() == []

        waiting = asyncio.create_task(inbox.request("survives cancellation"))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        channel, message = Inbox(inbox.path).pending()[0]
        assert message == "survives cancellation"
        reviewer.send(channel, "late reply")
        assert await inbox.recv(inbox.peer(channel)) == "late reply"

    asyncio.run(asyncio.wait_for(scenario(), 2))


def test_keyed_requests_reconnect_and_retain_responses(tmp_path):
    async def scenario():
        inbox = Inbox(tmp_path / "inbox.db", poll_interval=0.001)
        waiting = asyncio.create_task(inbox.request({"draft": 1}, key="review:1"))
        await asyncio.sleep(0)
        channel, _ = inbox.pending()[0]
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        reopened = Inbox(inbox.path)
        with pytest.raises(ValueError, match="different"):
            await reopened.request({"draft": 2}, key="review:1")
        reopened.send(channel, {"approved": True})
        reopened.send(channel, {"approved": True})  # Delivery retry is harmless.
        with pytest.raises(ValueError, match="already"):
            reopened.send(channel, {"approved": False})
        with pytest.raises(ValueError, match="already"):
            reopened.send(channel, {"approved": 1})  # JSON true and 1 are different replies.
        for _ in range(2):
            assert await reopened.request({"draft": 1}, key="review:1") == {"approved": True}
        assert reopened.pending() == []

        waiting = asyncio.create_task(inbox.request("typed", key="typed", output_type=int))
        await asyncio.sleep(0)
        channel, _ = inbox.pending()[0]
        inbox.send(channel, "bad integer")
        with pytest.raises(ValidationError):
            await waiting
        assert inbox.pending() == [(channel, "typed")]
        inbox.send(channel, 42)
        assert await inbox.request("typed", key="typed", output_type=int) == 42

    asyncio.run(asyncio.wait_for(scenario(), 2))

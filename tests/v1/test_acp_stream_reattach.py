"""ACP sessions survive process-stream transport failures via reattach."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import verifiers.v1.acp as acp_module
from verifiers.v1.acp import (
    RESYNC_MAGIC,
    ACPConfig,
    ACPHarnessSession,
    _packet,
    _PacketReader,
)
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


def _read_frames(data: bytes) -> list[dict]:
    frames = []
    view = memoryview(data)
    while view:
        size = int.from_bytes(view[:8], "big")
        frames.append(json.loads(bytes(view[8 : 8 + size]).decode()))
        view = view[8 + size :]
    return frames


class _Stream:
    """An async byte stream a test can feed, fail, or close."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return item


class _FlakyProcess:
    """A runner whose first stream dies mid-turn; reattach yields a fresh one."""

    def __init__(self, *, answered_before_drop: bool) -> None:
        self.answered_before_drop = answered_before_drop
        self.stdout = _Stream()
        self.stderr = _Stream()
        self.reattached = 0
        self.writes: list[dict] = []
        self.last_seq = 0
        self.last_response: dict | None = None

    async def write(self, data: bytes) -> None:
        request = _read_frames(data)[0]
        self.writes.append(request)
        if request.get("operation") == "prompt":
            seq = request["seq"]
            if self.reattached == 0:
                # First attempt: maybe answer (response lost), then the
                # transport dies exactly like a dropped gateway stream.
                if self.answered_before_drop:
                    self.last_seq = seq
                    self.last_response = {"ok": True, "reply": "early", "seq": seq}
                self.stdout.queue.put_nowait(
                    ConnectionError("process stream RPC failed (internal)")
                )
                return
            if seq <= self.last_seq:
                return  # runner dedupes replayed requests
            self.last_seq = seq
            self.last_response = {"ok": True, "reply": "recovered", "seq": seq}
            self.stdout.queue.put_nowait(_packet(self.last_response))
        elif request.get("operation") == "sync":
            nonce = request["nonce"].encode()
            sync = {
                "ok": True,
                "sync": True,
                "last_seq": self.last_seq,
                "last_response": self.last_response,
            }
            data = json.dumps(sync).encode()
            self.stdout.queue.put_nowait(
                b"\x00garbage-partial-frame"  # host may reattach mid-frame
                + RESYNC_MAGIC
                + nonce
                + RESYNC_MAGIC
                + len(data).to_bytes(8, "big")
                + data
            )

    async def reattach(self) -> bool:
        self.reattached += 1
        self.stdout = _Stream()
        self.stderr = _Stream()
        return True

    async def wait(self) -> int:
        return 0

    async def terminate(self) -> None:
        pass

    async def kill(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


def _session(process: _FlakyProcess) -> ACPHarnessSession:
    session = ACPHarnessSession(
        SimpleNamespace(config=SimpleNamespace(id="rlm")),
        SimpleNamespace(),
        Trace.model_construct(id="trace", stop_condition="test"),
        SimpleNamespace(),
        "http://intercept",
        "secret",
        {},
        TaskData(prompt="hello"),
        ACPConfig(env={}, command=["rlm"], prompt="hello"),
    )
    session._process = process
    session._reader = _PacketReader(process.stdout)
    return session


@pytest.mark.asyncio
async def test_turn_resends_after_stream_drop(monkeypatch) -> None:
    monkeypatch.setattr(acp_module, "RECONNECT_BACKOFF_SECONDS", 0.0)
    process = _FlakyProcess(answered_before_drop=False)
    session = _session(process)

    result = await session._run(None)

    assert result.stdout == "recovered"
    assert process.reattached == 1
    operations = [w.get("operation") for w in process.writes]
    assert operations == ["prompt", "sync", "prompt"]


@pytest.mark.asyncio
async def test_turn_recovers_lost_response_without_resend(monkeypatch) -> None:
    monkeypatch.setattr(acp_module, "RECONNECT_BACKOFF_SECONDS", 0.0)
    process = _FlakyProcess(answered_before_drop=True)
    session = _session(process)

    result = await session._run(None)

    assert result.stdout == "early"
    assert process.reattached == 1
    operations = [w.get("operation") for w in process.writes]
    assert operations == ["prompt", "sync"]


@pytest.mark.asyncio
async def test_turn_fails_when_runtime_cannot_reattach(monkeypatch) -> None:
    monkeypatch.setattr(acp_module, "RECONNECT_BACKOFF_SECONDS", 0.0)

    class NoReattach(_FlakyProcess):
        async def reattach(self) -> bool:
            return False

    process = NoReattach(answered_before_drop=False)
    session = _session(process)

    with pytest.raises(ConnectionError, match="process stream RPC failed"):
        await session._run(None)
    assert session._process is None  # torn down, next turn would restart

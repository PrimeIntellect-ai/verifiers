"""An episode hook's `keep` hands an undelivered episode to `discard` exactly once, and
never while `keep` is still running."""

import asyncio
import threading

import msgpack
import pytest

import verifiers.v1 as vf
from verifiers.v1.serve import server as server_module
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.server import EnvServer, _Keeping
from verifiers.v1.serve.types import CancelRequest, RunResponse


class Hook:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.release = threading.Event()
        self.release.set()
        self.keeping = False
        self.calls: list[str] = []

    def keep(self, episode, sampling) -> None:
        self.keeping = True
        self.release.wait()
        self.calls.append("keep")
        self.keeping = False
        if self.fail:
            raise ValueError("misaligned arrays")

    def discard(self, episode) -> None:
        assert not self.keeping, "discard ran while keep was storing"
        self.calls.append("discard")


def test_a_failed_keep_discards_once():
    hook = Hook(fail=True)
    keeping = _Keeping(hook, episode=None)
    with pytest.raises(ValueError):
        keeping.run(sampling=None)
    keeping.abandon()
    assert hook.calls == ["keep", "discard"]


def test_abandoning_a_running_keep_discards_after_it():
    hook = Hook()
    hook.release.clear()
    keeping = _Keeping(hook, episode=None)
    thread = threading.Thread(target=keeping.run, args=(None,))
    thread.start()
    keeping.abandon()
    hook.release.set()
    thread.join()
    assert hook.calls == ["keep", "discard"]


def test_abandoning_a_finished_keep_discards_once():
    hook = Hook()
    keeping = _Keeping(hook, episode=None)
    keeping.run(sampling=None)
    keeping.abandon()
    assert hook.calls == ["keep", "discard"]


class Frontend:
    async def send_multipart(self, frames, copy=True) -> None:
        pass


def _server() -> EnvServer:
    server = EnvServer.__new__(EnvServer)
    server.frontend = Frontend()
    server._running, server._kept, server._delivered = {}, {}, {}
    return server


def _cancel(request_id: str) -> bytes:
    return msgpack.packb(CancelRequest(request_id=request_id).model_dump(mode="json"))


def test_a_cancel_after_delivery_discards_once():
    """The client dropped a reply whose run its caller cancelled; the cancel that follows
    frees the delivered episode's arrays."""

    async def run() -> list[str]:
        server, hook = _server(), Hook()
        server._remember_delivered("run-1", hook, episode=None)
        await server._handle(b"client", b"req-2", b"cancel", _cancel("run-1"))
        await server._handle(b"client", b"req-3", b"cancel", _cancel("run-1"))
        return hook.calls

    assert asyncio.run(run()) == ["discard"]


def test_a_cancel_before_the_reply_leaves_discards_once():
    """A cancel landing after the run finished but before its reply was sent."""

    async def run() -> list[str]:
        server, hook = _server(), Hook()
        server._kept[b"run-1"] = (hook, None)
        await server._handle(b"client", b"req-2", b"cancel", _cancel("run-1"))
        assert not server._kept and not server._delivered
        return hook.calls

    assert asyncio.run(run()) == ["discard"]


def test_a_stale_delivery_expires_without_discard(monkeypatch):
    monkeypatch.setattr(server_module, "LATE_CANCEL_SECONDS", 0.01)

    async def run() -> list[str]:
        server, hook = _server(), Hook()
        server._remember_delivered("run-1", hook, episode=None)
        await asyncio.sleep(0.05)
        assert not server._delivered, "expired without waiting for another delivery"
        await server._handle(b"client", b"req-2", b"cancel", _cancel("run-1"))
        return hook.calls

    assert asyncio.run(run()) == []


def test_a_run_cancelled_while_decoding_its_reply_cancels_on_the_server():
    """The reply arrived, so the server counts the episode delivered; a cancel during
    the client's final decode must still reach the server."""

    async def run() -> bool:
        client = EnvClient("tcp://127.0.0.1:1")
        cancelled: list[str] = []
        sent: dict[str, str] = {}

        async def request(*args, request_id: str, **kwargs) -> RunResponse:
            sent["id"] = request_id
            return RunResponse(head={}, traces=[])

        async def decode(*args):
            raise asyncio.CancelledError

        client._request = request
        client._decode = decode
        client._cancel_run = cancelled.append
        try:
            with pytest.raises(asyncio.CancelledError):
                await client.run(
                    client=vf.TrainClientConfig(),
                    model="m",
                    sampling=vf.SamplingConfig(),
                    task_data={},
                )
        finally:
            await client.close()
        return cancelled == [sent["id"]]

    assert asyncio.run(run())

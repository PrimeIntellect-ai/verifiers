"""An episode hook's `keep` hands an undelivered episode to `discard` exactly once, and
never while `keep` is still running."""

import asyncio
import threading

import msgpack
import pytest

from verifiers.v1.serve.server import EnvServer, _Keeping
from verifiers.v1.serve.types import CancelRequest


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


def test_a_cancel_after_delivery_discards():
    """The client dropped a reply whose run its caller cancelled; the cancel that follows
    frees the delivered episode's arrays."""

    class Frontend:
        async def send_multipart(self, frames, copy=True) -> None:
            pass

    server = EnvServer.__new__(EnvServer)
    server.frontend = Frontend()
    server._running, server._kept, server._delivered = {}, {}, {}
    hook = Hook()
    server._remember_delivered("run-1", hook, episode=None)
    cancel = msgpack.packb(CancelRequest(request_id="run-1").model_dump(mode="json"))
    asyncio.run(server._handle(b"client", b"req-2", b"cancel", cancel))
    asyncio.run(server._handle(b"client", b"req-3", b"cancel", cancel))
    assert hook.calls == ["discard"]

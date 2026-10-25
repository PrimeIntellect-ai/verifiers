"""An episode hook's `keep` hands an undelivered episode to `discard` exactly once, and
never while `keep` is still running."""

import threading

import pytest

from verifiers.v1.serve.server import _Keeping


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

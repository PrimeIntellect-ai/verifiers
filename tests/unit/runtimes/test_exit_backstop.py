"""The atexit backstop frees only runtimes whose own teardown did not run to the end."""

import asyncio

import pytest

from verifiers.v1.runtimes import base


class Counted(base.RuntimeBackend):
    """A runtime that counts how often its resource is freed."""

    def __init__(self) -> None:
        super().__init__()
        self.freed = 0

    async def start(self) -> None:
        return None

    async def _run(self, runtime, argv, env):  # pragma: no cover - never run
        raise NotImplementedError

    async def _read(self, *args, **kwargs):  # pragma: no cover - never run
        raise NotImplementedError

    async def _write(self, *args, **kwargs):  # pragma: no cover - never run
        raise NotImplementedError

    def cleanup(self) -> None:
        self.freed += 1


async def test_a_stopped_runtime_is_not_freed_again_at_exit() -> None:
    backend = Counted()
    base.register(backend)
    await backend.stop()
    base.cleanup_at_exit()
    assert backend.freed == 1


async def test_a_runtime_whose_teardown_was_cut_is_freed_at_exit() -> None:
    backend = Counted()
    base.register(backend)
    base.cleanup_at_exit()
    assert backend.freed == 1


async def test_a_stop_cancelled_mid_teardown_still_leaves_the_backstop_nothing_to_do() -> (
    None
):
    backend = Counted()
    base.register(backend)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_teardown() -> None:
        started.set()
        await release.wait()
        backend.cleanup()

    backend.teardown = slow_teardown  # type: ignore[method-assign]
    stopping = asyncio.ensure_future(backend.stop())
    await started.wait()
    stopping.cancel()  # the shielded teardown runs on to its end
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await stopping
    base.cleanup_at_exit()
    assert backend.freed == 1

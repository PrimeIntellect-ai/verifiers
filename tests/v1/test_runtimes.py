"""Runtime.stop under cancellation: teardown must run to completion when the owning
task is cancelled (Ctrl-C / SIGTERM), then re-raise the cancellation — an interrupted
teardown leaks the container / paid sandbox."""

import asyncio

import pytest

from verifiers.v1.runtimes.base import ProgramResult, Runtime


class FakeRuntime(Runtime):
    """Teardown gated on an event so tests can cancel mid-teardown deterministically."""

    def __init__(self, fail: bool = False) -> None:
        super().__init__(name="fake")
        self.fail = fail
        self.teardown_started = asyncio.Event()
        self.release = asyncio.Event()
        self.teardown_finished = False

    async def start(self) -> None: ...

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return ProgramResult(0, "", "")

    async def read(self, path: str) -> bytes:
        return b""

    async def write(self, path: str, data: bytes) -> None: ...

    async def teardown(self) -> None:
        self.teardown_started.set()
        await self.release.wait()
        if self.fail:
            raise RuntimeError("boom")
        self.teardown_finished = True


async def test_stop_completes_teardown_despite_cancellation():
    # Shaped like the real Ctrl-C path: the owner's body is cancelled, its `finally`
    # runs `stop`, and a second cancellation lands mid-teardown.
    rt = FakeRuntime()

    async def owner() -> None:
        try:
            await asyncio.Event().wait()  # the rollout body
        finally:
            await rt.stop()

    task = asyncio.create_task(owner())
    await asyncio.sleep(0)
    task.cancel()  # Ctrl-C: cancels the body; the finally runs stop
    await rt.teardown_started.wait()
    task.cancel()  # second cancellation lands mid-teardown
    await asyncio.sleep(0)
    assert not task.done()  # still blocking on teardown, not orphaned
    rt.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rt.teardown_finished  # teardown ran to completion
    assert task.cancelled()  # and the cancellation still propagated


async def test_stop_direct_cancellation_mid_teardown():
    rt = FakeRuntime()
    task = asyncio.create_task(rt.stop())
    await rt.teardown_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    rt.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rt.teardown_finished


async def test_teardown_error_propagates_when_not_cancelled():
    # The call sites' `except Exception: logger.warning(...)` depends on this.
    rt = FakeRuntime(fail=True)
    rt.release.set()
    with pytest.raises(RuntimeError, match="boom"):
        await rt.stop()


async def test_modal_backstop_survives_truncated_teardown():
    # Loop death (second Ctrl-C) cancels the shield's inner task directly — simulated by
    # cancelling `teardown()` itself mid-terminate. The `_sandbox` guard must survive the
    # truncation so the atexit backstop (`cleanup`) can still terminate the paid sandbox.
    from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime

    class FakeSandbox:
        def __init__(self) -> None:
            self.terminated = False
            self.aio_started = asyncio.Event()
            sandbox = self

            class Terminate:
                def __call__(self) -> None:
                    sandbox.terminated = True

                async def aio(self) -> None:
                    sandbox.aio_started.set()
                    await asyncio.Event().wait()  # never completes; cancelled mid-await

            self.terminate = Terminate()

    rt = ModalRuntime(ModalConfig(), name="fake")
    sandbox = FakeSandbox()
    rt._sandbox = sandbox
    task = asyncio.create_task(rt.teardown())
    await sandbox.aio_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rt._sandbox is sandbox  # guard retained: the backstop is still armed
    rt.cleanup()  # the atexit backstop
    assert sandbox.terminated


async def test_cancelled_stop_reraises_cancellation_when_teardown_fails():
    # The cancellation must win (a bare RuntimeError would be eaten by the call sites'
    # `except Exception` and the cancellation lost); the teardown error is chained.
    rt = FakeRuntime(fail=True)
    task = asyncio.create_task(rt.stop())
    await rt.teardown_started.wait()
    task.cancel()
    rt.release.set()
    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task
    assert isinstance(excinfo.value.__cause__, RuntimeError)

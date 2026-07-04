"""Runtime.stop teardown reporting and abort semantics: an interrupt landing
mid-teardown is reported once (with the in-flight drain snapshot) so the user can
decide whether to wait or Ctrl-C again, and a user abort (KeyboardInterrupt) is never
subordinated to a pending cancellation."""

import asyncio
import logging

import pytest

from verifiers.v1.runtimes.base import ProgramResult, Runtime


class FakeRuntime(Runtime):
    """Teardown gated on an event so tests can interrupt mid-teardown deterministically."""

    def __init__(self, exc: BaseException | None = None) -> None:
        super().__init__(name="fake")
        self.exc = exc
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
        if self.exc is not None:
            raise self.exc
        self.teardown_finished = True


async def test_interrupted_drain_reported_as_one_aggregate_line(caplog, monkeypatch):
    # The legacy logging setup disables propagation on the package logger at import;
    # re-enable it so caplog's root handler sees the report.
    monkeypatch.setattr(logging.getLogger("verifiers"), "propagate", True)
    runtimes = [FakeRuntime() for _ in range(3)]
    tasks = [asyncio.create_task(rt.stop()) for rt in runtimes]
    for rt in runtimes:
        await rt.teardown_started.wait()
    with caplog.at_level(logging.WARNING, logger="verifiers.v1.runtimes.base"):
        for task in tasks:  # Ctrl-C lands while all three stops are in flight
            task.cancel()
        await asyncio.sleep(0)
        for rt in runtimes:
            rt.release.set()
        for task in tasks:
            with pytest.raises(asyncio.CancelledError):
                await task
    reports = [r for r in caplog.records if "Ctrl-C again" in r.getMessage()]
    assert len(reports) == 1  # one aggregate line, not one per teardown
    assert "3 in-flight teardown(s)" in reports[0].getMessage()
    done = [r for r in caplog.records if "teardowns finished" in r.getMessage()]
    assert len(done) == 1  # one completion line closes the drain
    assert all(rt.teardown_finished for rt in runtimes)


async def test_later_drain_reports_again(caplog, monkeypatch):
    # The dedup flag resets when the drain empties: a second interrupted drain in the
    # same process (e.g. a long-lived server) reports again.
    monkeypatch.setattr(logging.getLogger("verifiers"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="verifiers.v1.runtimes.base"):
        for _ in range(2):
            rt = FakeRuntime()
            task = asyncio.create_task(rt.stop())
            await rt.teardown_started.wait()
            task.cancel()
            rt.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    reports = [r for r in caplog.records if "Ctrl-C again" in r.getMessage()]
    assert len(reports) == 2


async def test_no_report_without_interrupt(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("verifiers"), "propagate", True)
    rt = FakeRuntime()
    rt.release.set()
    with caplog.at_level(logging.WARNING, logger="verifiers.v1.runtimes.base"):
        await rt.stop()
    assert not [r for r in caplog.records if "finishing teardown" in r.getMessage()]


class UserAbort(BaseException):
    """Stand-in for KeyboardInterrupt: a signal delivered while the interpreter executes
    teardown code manifests as the teardown raising it. Task.__step re-raises the real
    KeyboardInterrupt/SystemExit into the event loop (killing it — correct in production,
    fatal to the test session), so the shield-layer semantics are asserted with a plain
    BaseException, which run_shielded treats identically."""


async def test_teardown_base_exception_wins_over_cancellation():
    # The user's abort must win over the pending cancellation, not get chained under it.
    rt = FakeRuntime(exc=UserAbort())
    task = asyncio.create_task(rt.stop())
    await rt.teardown_started.wait()
    task.cancel()
    rt.release.set()
    with pytest.raises(UserAbort):
        await task


async def test_teardown_error_propagates_when_not_cancelled():
    # The call sites' `except Exception: logger.warning(...)` depends on this.
    rt = FakeRuntime(exc=RuntimeError("boom"))
    rt.release.set()
    with pytest.raises(RuntimeError, match="boom"):
        await rt.stop()

"""A Prime command cannot wait forever when its sandbox restarts underneath it.

The provider can keep a restarted sandbox `RUNNING` under the same id while the
command it was running is gone. Scoring has no deadline when a taskset ignores
timeouts, so `run()` must turn that into an explicit failure instead of waiting
on an exit event that will never arrive.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeBackend, PrimeConfig
from verifiers.v1.runtimes.backends import prime

BOOT = datetime(2026, 10, 8, 17, 42, 50, tzinfo=UTC)


class FakeProcess:
    def __init__(self, exit_code: int | None = None) -> None:
        self.closed = False
        self._exit_code = exit_code
        self._exited = asyncio.Event()
        self.stdout = self._empty()
        self.stderr = self._empty()

    async def _empty(self):
        if False:
            yield b""

    async def wait(self) -> int:
        if self._exit_code is None:
            await self._exited.wait()
        assert self._exit_code is not None
        return self._exit_code

    async def aclose(self) -> None:
        self.closed = True
        self._exited.set()


class FakeClient:
    def __init__(self, observations, process: FakeProcess) -> None:
        self.observations = list(observations)
        self.process = process

    async def open_process(self, *_args, **_kwargs) -> FakeProcess:
        return self.process

    async def get(self, _sandbox: str):
        if not self.observations:
            raise AssertionError("unexpected sandbox observation")
        observation = (
            self.observations.pop(0)
            if len(self.observations) > 1
            else self.observations[0]
        )
        if isinstance(observation, Exception):
            raise observation
        status, started_at = observation
        return SimpleNamespace(status=status, started_at=started_at)


def backend(monkeypatch, observations, *, exit_code=None, boot=BOOT) -> PrimeBackend:
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime, "_WATCH_INITIAL_SECONDS", 0)
    monkeypatch.setattr(prime, "_WATCH_MAX_SECONDS", 0)
    process = FakeProcess(exit_code)
    runtime = PrimeBackend(PrimeConfig(allow=[]), name="test")
    runtime.info.id = "box"
    runtime._client = FakeClient(observations, process)
    runtime._boot_started_at = boot
    return runtime


async def test_a_restarted_sandbox_fails_a_command_that_would_wait_forever(
    monkeypatch,
):
    runtime = backend(monkeypatch, [("RUNNING", BOOT + timedelta(minutes=1))])

    with pytest.raises(SandboxError, match="restarted during command"):
        await runtime.run(runtime.bind(), ["bash", "/tests/test.sh"], {})

    assert runtime._client.process.closed


@pytest.mark.parametrize("status", ["TERMINATED", "ERROR", "TIMEOUT"])
async def test_a_terminal_sandbox_fails_the_waiting_command(monkeypatch, status):
    runtime = backend(monkeypatch, [(status, BOOT)])

    with pytest.raises(SandboxError, match=f"became {status} during command"):
        await runtime.run(runtime.bind(), ["bash", "/tests/test.sh"], {})

    assert runtime._client.process.closed


async def test_a_failed_status_observation_is_not_a_confirmed_loss(monkeypatch):
    runtime = backend(
        monkeypatch,
        [
            RuntimeError("provider unavailable"),
            ("RUNNING", BOOT + timedelta(minutes=1)),
        ],
    )

    with pytest.raises(SandboxError, match="restarted during command"):
        await runtime.run(runtime.bind(), ["bash", "/tests/test.sh"], {})


async def test_a_finished_command_keeps_its_result_while_the_box_stays_running(
    monkeypatch,
):
    runtime = backend(monkeypatch, [("RUNNING", BOOT)], exit_code=0)

    result = await runtime.run(runtime.bind(), ["bash", "/tests/test.sh"], {})

    assert result.exit_code == 0
    assert runtime._client.process.closed

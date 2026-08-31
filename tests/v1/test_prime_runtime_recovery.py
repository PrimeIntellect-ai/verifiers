import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime

ROUTE_MISS = RuntimeError('Download failed: HTTP 502: {"error":"sandbox_not_found"}')


def make_runtime(monkeypatch, *, grace: float = 600.0) -> PrimeRuntime:
    monkeypatch.setattr("verifiers.v1.runtimes.prime.ensure_prime_auth", lambda: None)
    runtime = PrimeRuntime(PrimeConfig(connectivity_grace=grace))
    runtime.info.id = "sandbox-1"
    runtime._client = AsyncMock()
    return runtime


@pytest.mark.asyncio
async def test_background_poll_survives_eight_minute_route_outage_without_relaunch(
    monkeypatch,
):
    runtime = make_runtime(monkeypatch)
    now = 0.0

    runtime._clock = lambda: now

    async def advance(delay: float) -> None:
        nonlocal now
        now += delay

    runtime._sleep = advance
    job = SimpleNamespace(job_id="job-1")
    runtime._client.start_background_job.return_value = job
    runtime._client.get.return_value = SimpleNamespace(status="RUNNING")

    async def poll(_sandbox_id, _job):
        if now < 480:
            raise ROUTE_MISS
        return SimpleNamespace(
            completed=True,
            exit_code=0,
            stdout="done",
            stderr="",
        )

    runtime._client.get_background_job.side_effect = poll

    result = await runtime.run(["solver"], {})

    assert result.stdout == "done"
    runtime._client.start_background_job.assert_awaited_once()
    assert runtime._client.get_background_job.await_count > 1
    runtime._client.create.assert_not_awaited()
    runtime._client.delete.assert_not_awaited()
    assert runtime.info.id == "sandbox-1"


@pytest.mark.asyncio
async def test_confirmed_sandbox_deletion_fails_without_grace_delay(monkeypatch):
    runtime = make_runtime(monkeypatch)
    runtime._client.start_background_job.return_value = SimpleNamespace(job_id="job-1")
    runtime._client.get_background_job.side_effect = ROUTE_MISS
    runtime._client.get.return_value = SimpleNamespace(status="TERMINATED")
    runtime._sleep = AsyncMock()

    with pytest.raises(SandboxError, match="prime exec failed"):
        await runtime.run(["solver"], {})

    runtime._sleep.assert_not_awaited()
    runtime._client.get_background_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_idempotent_read_retries_on_same_sandbox(monkeypatch):
    runtime = make_runtime(monkeypatch)
    runtime._client.get.return_value = SimpleNamespace(status="RUNNING")
    runtime._sleep = AsyncMock()
    calls = 0

    async def download(sandbox_id, _target, local_path):
        nonlocal calls
        calls += 1
        assert sandbox_id == "sandbox-1"
        if calls == 1:
            raise ROUTE_MISS
        await asyncio.to_thread(Path(local_path).write_bytes, b"payload")

    runtime._client.download_file.side_effect = download

    assert await runtime._read("result.txt") == b"payload"
    assert calls == 2


@pytest.mark.asyncio
async def test_non_route_read_error_is_not_hidden(monkeypatch):
    runtime = make_runtime(monkeypatch)
    runtime._client.download_file.side_effect = RuntimeError("permission denied")
    runtime._sleep = AsyncMock()

    with pytest.raises(SandboxError, match="permission denied"):
        await runtime._read("secret.txt")

    runtime._sleep.assert_not_awaited()

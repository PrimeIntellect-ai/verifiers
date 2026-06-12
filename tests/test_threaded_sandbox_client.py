from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verifiers.utils import threaded_sandbox_client
from verifiers.utils.threaded_sandbox_client import ThreadedAsyncSandboxClient


@pytest.mark.asyncio
async def test_wait_for_creation_polls_with_short_worker_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(threaded_sandbox_client.asyncio, "sleep", sleep)
    monkeypatch.setattr(threaded_sandbox_client.random, "uniform", lambda *_: 1.2)
    client = object.__new__(ThreadedAsyncSandboxClient)
    client.metrics = {}
    client.get = AsyncMock(
        side_effect=[
            SimpleNamespace(status="PENDING"),
            SimpleNamespace(status="RUNNING"),
            SimpleNamespace(status="RUNNING"),
        ]
    )
    client._is_sandbox_reachable = AsyncMock(return_value=True)

    await client.wait_for_creation("sbx-1", stability_checks=2)

    assert client.get.await_count == 3
    assert client._is_sandbox_reachable.await_count == 2
    assert sleeps == [1.2, 0.6]
    assert client.metrics["readiness_polls"] == 3
    assert client.metrics["readiness_seconds"] >= 0


@pytest.mark.asyncio
async def test_background_job_polling_adds_jitter_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(threaded_sandbox_client.asyncio, "sleep", sleep)
    monkeypatch.setattr(threaded_sandbox_client.random, "uniform", lambda *_: 0.8)
    client = object.__new__(ThreadedAsyncSandboxClient)
    client.metrics = {}
    client.start_background_job = AsyncMock(return_value="job-1")
    client.get_background_job = AsyncMock(
        side_effect=[
            SimpleNamespace(completed=False),
            SimpleNamespace(completed=True),
        ]
    )

    result = await client.run_background_job("sbx-1", "echo hi", poll_interval=2)

    assert result.completed is True
    assert sleeps == [1.6]
    assert client.metrics["background_job_polls"] == 2
    assert client.metrics["background_job_seconds"] >= 0

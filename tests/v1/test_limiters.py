import asyncio

import pytest

from verifiers.v1.runtimes import limiters


def test_creation_limiter_discards_cursor_from_another_monotonic_epoch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    monkeypatch.setattr(limiters.time, "time", lambda: 1_800_000_000.0)
    monkeypatch.setattr(limiters.time, "monotonic", lambda: 77_817.111942894)

    limiter = limiters.CreationLimiter("prime-tunnel", per_sec=1)
    # This cursor came from another node's monotonic epoch.
    limiter._path.write_text("4346537.619615014")

    assert limiter._reserve() == 0
    assert float(limiter._path.read_text()) == 1_800_000_001.0


async def test_creation_limiter_paces_concurrent_shared_reservations(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    monkeypatch.setattr(limiters.time, "time", lambda: 1_800_000_000.0)
    waits: list[float] = []

    async def record_sleep(wait: float) -> None:
        waits.append(wait)

    monkeypatch.setattr(limiters.asyncio, "sleep", record_sleep)
    workers = [limiters.CreationLimiter("prime-tunnel", per_sec=4) for _ in range(4)]

    async def reserve(worker: limiters.CreationLimiter) -> None:
        async with worker:
            pass

    await asyncio.gather(*(reserve(worker) for worker in workers))

    assert sorted(waits) == pytest.approx([0.25, 0.5, 0.75])
    assert float(workers[0]._path.read_text()) == pytest.approx(1_800_000_001.0)

"""Unit tests for the cross-process creation limiter (leaky bucket via `filelock`)."""

import itertools
import multiprocessing as mp
import time
from pathlib import Path

import pytest
from filelock import SoftFileLock

from verifiers.v1.runtimes import limiters


def _make(lim_dir, name: str, per_sec: float) -> limiters.CreationLimiter:
    limiters.LIMITER_DIR = Path(lim_dir)
    return limiters.CreationLimiter(name, per_sec)


def _worker(lim_dir: str, name: str, n: int, per_sec: float, q) -> None:
    lim = _make(lim_dir, name, per_sec)
    for _ in range(n):
        time.sleep(lim._reserve())
        q.put(time.time())


def test_cursor_advances_one_interval_per_reservation(monkeypatch, tmp_path):
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    lim = limiters.CreationLimiter("unit", per_sec=10)  # 100ms slots
    lim._reserve()
    waits = [lim._reserve() for _ in range(3)]
    interval = 1 / 10
    for prev, cur in itertools.pairwise(waits):
        assert cur - prev == pytest.approx(interval, abs=0.05)


def test_corrupt_bucket_resets_cursor(monkeypatch, tmp_path):
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    lim = limiters.CreationLimiter("corrupt", per_sec=10)
    (tmp_path / "corrupt.bucket").write_text("garbage-not-a-float")
    assert lim._reserve() == 0.0  # resets instead of failing every create
    assert float((tmp_path / "corrupt.bucket").read_text()) > time.time()


def test_backlog_over_five_minutes_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    lim = limiters.CreationLimiter("backlog", per_sec=10)
    (tmp_path / "backlog.bucket").write_text(repr(time.time() + 400))
    with pytest.raises(TimeoutError):
        lim._reserve()


def test_soft_lock_env_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    monkeypatch.setenv("VERIFIERS_LIMITER_SOFT_LOCK", "1")
    lim = limiters.CreationLimiter("soft", per_sec=10)
    assert isinstance(lim._lock, SoftFileLock)


def test_aggregate_rate_holds_across_processes(tmp_path):
    # 250ms slots keep the assertion robust on loaded CI runners: wake times carry
    # unbounded scheduler overshoot, so a tight slot would flake even when the
    # bucket is perfectly enforced. Half a slot still discriminates sharply
    # against a broken lock, where overlapping reservations collapse to ~0ms.
    per_sec, n_proc, n_res = 4.0, 4, 3
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(str(tmp_path), "cross", n_res, per_sec, q))
        for _ in range(n_proc)
    ]
    for p in procs:
        p.start()
    wakes = sorted(q.get(timeout=60) for _ in range(n_proc * n_res))
    for p in procs:
        p.join(60)
    assert all(p.exitcode == 0 for p in procs)
    interval = 1 / per_sec
    min_gap = min(b - a for a, b in itertools.pairwise(wakes))
    # The bucket is shared: no two creations anywhere may land closer than one slot
    # (modulo sleep-overshoot), regardless of how many processes contend.
    assert min_gap >= interval * 0.5

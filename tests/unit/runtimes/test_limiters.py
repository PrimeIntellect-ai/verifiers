import os

import pytest

from verifiers.v1._runtimes import limiters


@pytest.fixture
def limiter(monkeypatch, tmp_path):
    monkeypatch.setattr(limiters, "LIMITER_DIR", tmp_path)
    monkeypatch.setattr(limiters.time, "time", lambda: 100.0)
    return limiters.CreationLimiter("sandbox", "run", per_sec=2)


def test_reservation_recovers_a_nul_padded_cursor(limiter, tmp_path):
    bucket = tmp_path / "sandbox-run.bucket"
    bucket.write_text("\x00" * 18 + "123.0")

    assert limiter._reserve() == 23.0
    assert bucket.read_text() == "123.5"


def test_reservation_opens_the_bucket_without_append(limiter, monkeypatch):
    open_file = os.open
    open_flags = []

    def record_open(path, flags, mode):
        open_flags.append(flags)
        return open_file(path, flags, mode)

    monkeypatch.setattr(limiters.os, "open", record_open)

    limiter._reserve()

    # A local filesystem uses the new EOF after truncate, even with O_APPEND.
    assert len(open_flags) == 1
    assert not open_flags[0] & os.O_APPEND


def test_reservations_are_spaced_by_the_interval(limiter):
    assert [limiter._reserve() for _ in range(3)] == [0.0, 0.5, 1.0]

"""The host-side Prime call timings: one summary a minute, not one line per call."""

import logging

from verifiers.v1.runtimes import prime


def test_call_stats_summarize_once_an_interval_and_start_over(
    caplog, monkeypatch
) -> None:
    clock = iter([0.0, 10.0, 70.0])
    monkeypatch.setattr(prime.time, "monotonic", lambda: next(clock))
    stats = prime._CallStats()  # since = 0.0
    with caplog.at_level(logging.INFO, logger=prime.stats_logger.name):
        stats.record("exec.output", 2.0)  # at 10 s: inside the interval
        assert caplog.records == []
        stats.record("exec.output", 4.0)  # at 70 s: the summary
    [summary] = [r for r in caplog.records if r.name == prime.stats_logger.name]
    assert summary.levelno == logging.INFO
    assert stats.samples == {} and stats.since == 70.0

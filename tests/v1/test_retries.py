"""Whole-rollout retry behavior (the episode is the retry atom)."""

import verifiers.v1 as vf
from verifiers.v1.retries import RolloutRetryConfig, run_episode_with_retry
from verifiers.v1.trace import Episode, Trace, TraceTask


class Boom(Exception):
    pass


def _record(*, trace_error: Exception | None = None) -> Episode:
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")))
    if trace_error is not None:
        trace.capture_error(trace_error)
    return Episode.of(trace)


async def test_record_retry_is_record_atomic():
    """A retryable error on any trace retries the whole episode; a final good attempt
    returns clean, a final bad one carries the earlier attempts' errors."""
    calls = 0

    def flaky_then_good():
        async def attempt() -> Episode:
            nonlocal calls
            calls += 1
            return _record(trace_error=Boom("transient") if calls == 1 else None)

        return attempt()

    retry = RolloutRetryConfig(max_retries=2, include=["Boom"])
    episode = await run_episode_with_retry(flaky_then_good, retry)
    assert calls == 2
    assert episode.ok and not episode.errors  # a good final attempt returns clean

    calls = 0

    def always_bad():
        async def attempt() -> Episode:
            nonlocal calls
            calls += 1
            return _record(trace_error=Boom(f"attempt {calls}"))

        return attempt()

    episode = await run_episode_with_retry(always_bad, retry)
    assert calls == 3  # 1 attempt + 2 retries, all bad
    # the final episode carries the retried attempts' trace errors as its history
    assert [e.message for e in episode.errors] == ["attempt 1", "attempt 2"]
    assert episode.traces[0].error is not None and not episode.ok

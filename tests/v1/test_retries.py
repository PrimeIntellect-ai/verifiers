"""Whole-rollout retry behavior (trace- and record-level)."""

import verifiers.v1 as vf
from verifiers.v1.retries import (
    RolloutRetryConfig,
    run_record_with_retry,
    run_with_retry,
)
from verifiers.v1.trace import RolloutRecord, Trace, TraceTask


class Boom(Exception):
    pass


async def test_run_with_retry_awaits_thunks():
    """The retry path must await a plain callable returning an awaitable — tenacity only
    auto-awaits coroutine functions, and callers hand it `lambda: agent.run(task)`."""
    calls = 0

    def thunk():
        async def attempt() -> Trace:
            nonlocal calls
            calls += 1
            trace = Trace(
                task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi"))
            )
            if calls == 1:
                trace.capture_error(Boom("transient"))
            return trace

        return attempt()

    retry = RolloutRetryConfig(max_retries=2, include=["Boom"])
    trace = await run_with_retry(thunk, retry)
    assert calls == 2  # first attempt errored retryably, second succeeded
    assert trace.error is None


def _record(*, trace_error: Exception | None = None) -> RolloutRecord:
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")))
    if trace_error is not None:
        trace.capture_error(trace_error)
    return RolloutRecord.of(trace)


async def test_record_retry_is_record_atomic():
    """A retryable error on any trace retries the whole record; a final good attempt
    returns clean, a final bad one carries the earlier attempts' errors."""
    calls = 0

    def flaky_then_good():
        async def attempt() -> RolloutRecord:
            nonlocal calls
            calls += 1
            return _record(trace_error=Boom("transient") if calls == 1 else None)

        return attempt()

    retry = RolloutRetryConfig(max_retries=2, include=["Boom"])
    record = await run_record_with_retry(flaky_then_good, retry)
    assert calls == 2
    assert record.ok and not record.errors  # a good final attempt returns clean

    calls = 0

    def always_bad():
        async def attempt() -> RolloutRecord:
            nonlocal calls
            calls += 1
            return _record(trace_error=Boom(f"attempt {calls}"))

        return attempt()

    record = await run_record_with_retry(always_bad, retry)
    assert calls == 3  # 1 attempt + 2 retries, all bad
    # the final record carries the retried attempts' trace errors as its history
    assert [e.message for e in record.errors] == ["attempt 1", "attempt 2"]
    assert record.traces[0].error is not None and not record.ok

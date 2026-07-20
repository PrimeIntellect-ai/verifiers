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


async def test_any_captured_error_counts_for_retry():
    """A retryable failure masked by a later capture (e.g. a teardown error recorded
    after the real one) still retries — the policy matches all captured errors, not
    just the most recent."""
    calls = 0

    def masked_then_good():
        async def attempt() -> Episode:
            nonlocal calls
            calls += 1
            episode = _record(trace_error=Boom("real") if calls == 1 else None)
            if calls == 1:
                episode.traces[0].capture_error(RuntimeError("teardown"))
            return episode

        return attempt()

    retry = RolloutRetryConfig(max_retries=1, include=["Boom"])
    episode = await run_episode_with_retry(masked_then_good, retry)
    assert calls == 2
    assert episode.ok


async def test_user_respond_attempt_is_time_bounded(monkeypatch):
    """A wedged user-server connection (accepted, then silent) fails its attempt at
    the configured bound instead of sitting in the transport's read timeout, so the
    retry — on a fresh session — recovers the turn within the harness window."""
    import asyncio
    import contextlib
    import json
    from types import SimpleNamespace

    from verifiers.v1.mcp import launch

    attempts = 0

    @contextlib.asynccontextmanager
    async def wedged_then_good(url):
        nonlocal attempts
        attempts += 1

        async def call_tool(name, args):
            if attempts == 1:
                await asyncio.Event().wait()  # the wedge: connected, never answers
            payload = json.dumps({"messages": [{"role": "user", "content": "next"}]})
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=payload)])

        yield SimpleNamespace(call_tool=call_tool)

    monkeypatch.setattr(launch, "user_session", wedged_then_good)
    messages = await launch.user_respond("http://sim", "hi", 1, timeout=0.05)
    assert attempts == 2
    assert [m.content for m in messages] == ["next"]

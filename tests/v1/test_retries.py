"""Whole-rollout retry behavior (the episode is the retry atom)."""

import verifiers.v1 as vf
from verifiers.v1.retries import RolloutRetryConfig, run_episode_with_retry
from verifiers.v1.trace import EpisodeRecord, Trace, TraceTask


class Boom(Exception):
    pass


def _record(*, trace_error: Exception | None = None) -> EpisodeRecord:
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")))
    if trace_error is not None:
        trace.capture_error(trace_error)
    return EpisodeRecord.of(trace)


async def test_record_retry_is_record_atomic():
    """A retryable error on any trace retries the whole episode; a final good attempt
    returns clean, a final bad one carries the earlier attempts' errors."""
    calls = 0

    def flaky_then_good():
        async def attempt() -> EpisodeRecord:
            nonlocal calls
            calls += 1
            return _record(trace_error=Boom("transient") if calls == 1 else None)

        return attempt()

    retry = RolloutRetryConfig(max_retries=2, include=["Boom"])
    episode = await run_episode_with_retry(flaky_then_good, retry)
    assert calls == 2
    assert (
        episode.ok and not episode.episode.errors
    )  # a good final attempt returns clean

    calls = 0

    def always_bad():
        async def attempt() -> EpisodeRecord:
            nonlocal calls
            calls += 1
            return _record(trace_error=Boom(f"attempt {calls}"))

        return attempt()

    episode = await run_episode_with_retry(always_bad, retry)
    assert calls == 3  # 1 attempt + 2 retries, all bad
    # the final episode carries the retried attempts' trace errors as its history
    assert [e.message for e in episode.episode.errors] == ["attempt 1", "attempt 2"]
    assert episode.traces[0].error is not None and not episode.ok


async def test_any_captured_error_counts_for_retry():
    """A retryable failure masked by a later capture (e.g. a teardown error recorded
    after the real one) still retries — the policy matches all captured errors, not
    just the most recent."""
    calls = 0

    def masked_then_good():
        async def attempt() -> EpisodeRecord:
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


def _trace(error: Exception | None = None) -> Trace:
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")))
    if error is not None:
        trace.capture_error(error)
    return trace


async def test_agent_retries_its_own_rollout(monkeypatch):
    """`--env.<agent>.retries`: a retryable trace error reruns just this agent's
    rollout; a good final attempt returns clean."""
    monkeypatch.setattr("verifiers.v1.agent.backoff", lambda attempt: 0.0)
    calls = {"n": 0}

    async def flaky_then_good(self, task, runtime, shared_tools, on_trace):
        calls["n"] += 1
        return _trace(RuntimeError("blip") if calls["n"] == 1 else None)

    monkeypatch.setattr(vf.Agent, "_run_once", flaky_then_good)
    agent = vf.Agent(
        vf.AgentConfig(model="m", retries=RolloutRetryConfig(max_retries=1))
    )
    trace = await agent.run(vf.Task(vf.TaskData(idx=0, prompt="hi")))
    assert calls["n"] == 2
    assert not trace.errors  # a good final attempt returns clean


async def test_agent_never_retries_into_a_borrowed_box(monkeypatch):
    """A borrowed box's state is no longer the task's start state: the error
    stands after one attempt instead of rerunning setup into a dirty world."""
    from verifiers.v1.runtimes import SubprocessConfig, make_runtime

    calls = {"n": 0}

    async def always_bad(self, task, runtime, shared_tools, on_trace):
        calls["n"] += 1
        return _trace(RuntimeError("boom"))

    monkeypatch.setattr(vf.Agent, "_run_once", always_bad)
    agent = vf.Agent(
        vf.AgentConfig(model="m", retries=RolloutRetryConfig(max_retries=3))
    )
    box = make_runtime(SubprocessConfig())
    trace = await agent.run(vf.Task(vf.TaskData(idx=0, prompt="hi")), runtime=box)
    assert calls["n"] == 1 and trace.error is not None

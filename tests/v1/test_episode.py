"""Episode guards and slot observability (no live env — a stub stands in)."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.episode import Episode, RunSlot
from verifiers.v1.retries import RolloutRetryConfig
from verifiers.v1.trace import RolloutRecord, Trace, TraceTask


class GroupTask(vf.Task):
    @vf.group_reward
    async def spread(self, traces: list[Trace]) -> list[float]:
        return [0.0 for _ in traces]


def _task(cls=vf.Task) -> vf.Task:
    return cls(vf.TaskData(idx=0, prompt="hi"))


class StubEnv:
    """Duck-types the one method Episode consumes: mint a single-trace record, publishing
    the trace via `on_trace` (as the engine does the moment a run starts)."""

    def __init__(self) -> None:
        self.runs = 0

    async def run_record(self, task: vf.Task, ctx, *, on_trace=None) -> RolloutRecord:
        self.runs += 1
        trace = Trace(task=TraceTask(type=type(task).__name__, data=task.data))
        if on_trace is not None:
            on_trace(trace)
        trace.is_completed = True
        return RolloutRecord.of(trace, env="stub")


def test_group_reward_task_needs_two_rollouts():
    with pytest.raises(ValueError, match="need >=2"):
        Episode(StubEnv(), _task(GroupTask), None, n=1, retry=RolloutRetryConfig())


def test_episode_needs_a_rollout():
    with pytest.raises(ValueError, match="n >= 1"):
        Episode(StubEnv(), _task(), None, n=0, retry=RolloutRetryConfig())


async def test_slots_observe_traces_and_flip_done():
    env = StubEnv()
    episode = Episode(env, _task(), None, n=3, retry=RolloutRetryConfig())
    assert [s.traces for s in episode.slots] == [[]] * 3
    assert not any(s.done for s in episode.slots)
    completed: list[RolloutRecord] = []

    async def on_complete(record: RolloutRecord) -> None:
        completed.append(record)

    records = await episode.run(on_complete=on_complete)
    assert env.runs == 3
    assert [s.record for s in episode.slots] == records
    assert [s.traces for s in episode.slots] == [list(r.traces) for r in records]
    assert all(s.done for s in episode.slots)
    assert completed == records


async def test_group_scored_episode_delays_done_until_group_scoring():
    env = StubEnv()
    episode = Episode(env, _task(GroupTask), None, n=2, retry=RolloutRetryConfig())
    done_at_complete: list[bool] = []

    async def on_complete(record: RolloutRecord) -> None:
        # By the time completion callbacks fire, group scoring ran and slots are done.
        done_at_complete.append(all(s.done for s in episode.slots))

    records = await episode.run(on_complete=on_complete)
    assert done_at_complete == [True, True]
    assert all("spread" in t.rewards for r in records for t in r.traces)


def test_finished_slot_from_saved_record():
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=7, prompt="hi")))
    record = RolloutRecord.of(trace, env="stub")
    slot = RunSlot.finished(record)
    assert slot.done and slot.record is record and slot.task.data.idx == 7
    assert slot.traces == [trace]


class Boom(Exception):
    pass


async def test_run_with_retry_awaits_thunks():
    """The retry path must await a plain callable returning an awaitable — tenacity only
    auto-awaits coroutine functions, and callers hand it `lambda: agent.run(task)`."""
    from verifiers.v1.retries import run_with_retry

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
    from verifiers.v1.retries import run_record_with_retry

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

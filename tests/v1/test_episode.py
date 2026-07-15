"""Episode guards and slot observability (no live agent — a stub stands in)."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.episode import Episode, RunSlot
from verifiers.v1.retries import RolloutRetryConfig
from verifiers.v1.trace import Trace, TraceTask


class GroupTask(vf.Task):
    @vf.group_reward
    async def spread(self, traces: list[Trace]) -> list[float]:
        return [0.0 for _ in traces]


def _task(cls=vf.Task) -> vf.Task:
    return cls(vf.TaskData(idx=0, prompt="hi"))


class StubAgent:
    """Duck-types the one method Episode consumes: mint a trace, publish it via
    `on_trace` (as the engine does the moment a run starts), return it finished."""

    def __init__(self) -> None:
        self.runs = 0

    async def run(self, task: vf.Task, *, on_trace=None) -> Trace:
        self.runs += 1
        trace = Trace(task=TraceTask(type=type(task).__name__, data=task.data))
        if on_trace is not None:
            on_trace(trace)
        trace.is_completed = True
        return trace


def test_group_reward_task_needs_two_rollouts():
    with pytest.raises(ValueError, match="need >=2"):
        Episode(StubAgent(), _task(GroupTask), n=1, retry=RolloutRetryConfig())


def test_episode_needs_a_rollout():
    with pytest.raises(ValueError, match="n >= 1"):
        Episode(StubAgent(), _task(), n=0, retry=RolloutRetryConfig())


async def test_slots_observe_traces_and_flip_done():
    agent = StubAgent()
    episode = Episode(agent, _task(), n=3, retry=RolloutRetryConfig())
    assert [s.trace for s in episode.slots] == [None] * 3
    assert not any(s.done for s in episode.slots)
    completed: list[Trace] = []

    async def on_complete(trace: Trace) -> None:
        completed.append(trace)

    traces = await episode.run(on_complete=on_complete)
    assert agent.runs == 3
    assert [s.trace for s in episode.slots] == traces
    assert all(s.done for s in episode.slots)
    assert completed == traces


async def test_group_scored_episode_delays_done_until_group_scoring():
    agent = StubAgent()
    episode = Episode(agent, _task(GroupTask), n=2, retry=RolloutRetryConfig())
    done_at_complete: list[bool] = []

    async def on_complete(trace: Trace) -> None:
        # By the time completion callbacks fire, group scoring ran and slots are done.
        done_at_complete.append(all(s.done for s in episode.slots))

    traces = await episode.run(on_complete=on_complete)
    assert done_at_complete == [True, True]
    assert all("spread" in t.rewards for t in traces)


def test_finished_slot_from_saved_trace():
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=7, prompt="hi")))
    slot = RunSlot.finished(trace)
    assert slot.done and slot.trace is trace and slot.task.data.idx == 7


class Boom(Exception):
    pass


async def test_run_with_retry_awaits_thunks():
    """The retry path must await a plain callable returning an awaitable — tenacity only
    auto-awaits coroutine functions, and Episode hands it `lambda: agent.run(task)`."""
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

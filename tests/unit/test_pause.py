"""Pausing an episode: its model calls wait, and the paused time is not the agent's."""

import asyncio

from verifiers.v1.clients import EvalClientConfig, ModelContext, ModelRoute
from verifiers.v1.env import RunSlot
from verifiers.v1.pause import Pause, pausing
from verifiers.v1.rollout import OWNER_HOLD, Rollout
from verifiers.v1.session import RolloutSession


class HeldRollout:
    set_paused = Rollout.set_paused
    _begin_hold = Rollout._begin_hold
    _end_hold = Rollout._end_hold
    _segment_agent_time = Rollout._segment_agent_time
    _suspend_box = Rollout._suspend_box

    def __init__(self) -> None:
        # No memory snapshots: a pause stops only the agent deadline.
        self._memory = False
        self._closed = False
        self._harness_session = None
        self._suspension = None
        self._unpaused = asyncio.Event()
        self._unpaused.set()
        self._holds: set[str] = set()
        self._hold_started = None
        self._hold_deadline = None
        self._held_time = 0.0
        self._segment_start = asyncio.get_running_loop().time()
        self._deadline = None
        self.deadline_at = None


def test_a_slot_has_its_own_pause():
    first, second = RunSlot(task=None), RunSlot(task=None)  # type: ignore[arg-type]
    first.pause.pause()
    assert first.pause.paused and not second.pause.paused


def test_a_session_follows_its_episodes_pause():
    pause = Pause()
    session = RolloutSession(
        ModelRoute(ModelContext(model="m", client=EvalClientConfig())),
        trace=None,  # type: ignore[arg-type]
        running=pause.running,
    )
    pause.pause()
    assert not session.running.is_set()
    pause.resume()
    assert session.running.is_set()
    # A session outside any episode always runs.
    assert RolloutSession(
        ModelRoute(ModelContext(model="m", client=EvalClientConfig())),
        trace=None,  # type: ignore[arg-type]
    ).running.is_set()


async def test_rollouts_built_in_an_episode_see_its_pause():
    pause = Pause()
    token = pausing.set(pause)
    try:
        seen = await asyncio.create_task(asyncio.sleep(0, pausing.get()))
    finally:
        pausing.reset(token)
    assert seen is pause and pausing.get() is None


async def test_a_pause_stops_the_agent_deadline():
    run = HeldRollout()
    pause = Pause()
    pause.attach(run)  # type: ignore[arg-type]
    loop = asyncio.get_running_loop()
    async with asyncio.timeout_at(loop.time() + 0.2) as deadline:
        run._deadline = deadline
        run.deadline_at = deadline.when()
        pause.pause()
        assert deadline.when() is None
        await asyncio.sleep(0.3)
        pause.resume()
        when = deadline.when()
        assert when is not None and when >= loop.time() + 0.15
    assert run._segment_agent_time() < 0.1


async def test_a_rollout_attached_while_paused_holds_at_once():
    run = HeldRollout()
    pause = Pause()
    pause.pause()
    pause.attach(run)  # type: ignore[arg-type]
    assert run._hold_started is not None
    pause.resume()
    assert run._hold_started is None


async def test_an_owner_release_does_not_end_a_pause():
    run = HeldRollout()
    pause = Pause()
    pause.attach(run)  # type: ignore[arg-type]
    loop = asyncio.get_running_loop()
    async with asyncio.timeout_at(loop.time() + 10) as deadline:
        run._deadline = deadline
        run.deadline_at = deadline.when()
        run._begin_hold(OWNER_HOLD)
        pause.pause()
        run._end_hold(OWNER_HOLD)
        await asyncio.sleep(0)
        assert deadline.when() is None
        pause.resume()
        assert deadline.when() is not None

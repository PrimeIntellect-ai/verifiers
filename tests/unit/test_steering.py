"""`Steering`: a caller's messages reach the rollout's running turn, once per message ID
and attempt; with no turn running nothing is delivered, and the receipt says so."""

import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import _Steers

INJECTED = {"outcome": "injected"}
NO_RUNNING_TURN = {"outcome": "promptRequired", "reason": "noRunningTurn"}


class FakeRun:
    """A rollout whose harness takes every message, after `gate` opens."""

    def __init__(self) -> None:
        self.ok, self.closed = True, False
        self.delivered: list[str] = []
        self.gate = asyncio.Event()
        self.gate.set()

    async def steer(self, message: str, *, message_id: str | None = None) -> dict:
        await self.gate.wait()
        self.delivered.append(message)
        return dict(INJECTED)


@pytest.mark.asyncio
async def test_nothing_is_delivered_with_no_rollout_running():
    assert await vf.Steering().steer("change course") == NO_RUNNING_TURN


@pytest.mark.asyncio
async def test_a_message_id_is_delivered_once_per_attempt():
    steering, first = vf.Steering(), FakeRun()
    steers = _Steers(first)
    steering._bind(steers)
    assert await steering.steer("stop", message_id="m1") == INJECTED
    assert await steering.steer("stop", message_id="m1") == INJECTED
    assert first.delivered == ["stop"]
    with pytest.raises(ValueError):
        await steering.steer("other", message_id="m1")
    await steering._unbind(steers)

    # a retry starts over: the message is not replayed, only sent again
    retry = FakeRun()
    steering._bind(_Steers(retry))
    assert await steering.steer("stop", message_id="m1") == INJECTED
    assert retry.delivered == ["stop"]


@pytest.mark.asyncio
async def test_a_delivery_cut_off_by_the_turn_ending_had_no_running_turn():
    steering, run = vf.Steering(), FakeRun()
    run.gate.clear()
    steers = _Steers(run)
    steering._bind(steers)
    pending = asyncio.create_task(steering.steer("too late"))
    await asyncio.sleep(0)
    await steering._unbind(steers)
    assert await pending == NO_RUNNING_TURN
    assert run.delivered == []


@pytest.mark.asyncio
async def test_a_stopped_rollout_takes_nothing():
    steering, run = vf.Steering(), FakeRun()
    run.ok = False
    steering._bind(_Steers(run))
    assert (await steering.steer("hello"))["outcome"] == "promptRequired"
    assert run.delivered == []


def test_a_steering_serves_one_rollout_at_a_time():
    steering = vf.Steering()
    steering._bind(_Steers(FakeRun()))
    with pytest.raises(RuntimeError):
        steering._bind(_Steers(FakeRun()))

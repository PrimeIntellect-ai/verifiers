"""Steer any running agent: a per-process registry of live rollouts and `steer()`.

Every rollout registers while it is open, by its trace id and, for an env episode's
agent, by `(episode id, seat name)`. `steer()` finds the rollout and delivers a user
message to it — through the harness's native mid-turn steering where it has one
(ACP), else through the interception, which adds the message to the rollout's next
model request (`RolloutSession.queue_steer`). Each steer is recorded on the trace
(`Trace.steers`).

The registry is per process: an env server's `steer` request (`serve.types.SteerRequest`)
reaches the rollouts of the worker process that runs them.
"""

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from verifiers.v1.rollout import Rollout

logger = logging.getLogger(__name__)

SeatKey = tuple[str, str]
"""`(episode id, seat name)` of an env episode's agent."""

_by_trace: dict[str, "Rollout"] = {}
_by_seat: dict[SeatKey, list["Rollout"]] = {}


class SteerReceipt(BaseModel):
    """What became of one steer."""

    queued: bool
    """Whether a live rollout accepted the message."""
    reason: str | None = None
    """Why it was not accepted."""
    trace_id: str | None = None
    """The rollout it went to."""
    delivery: Literal["native", "interception"] | None = None


def register(rollout: "Rollout", seat: SeatKey | None = None) -> None:
    """Make an open rollout steerable."""
    _by_trace[rollout.trace.id] = rollout
    if seat is not None:
        _by_seat.setdefault(seat, []).append(rollout)


def unregister(rollout: "Rollout", seat: SeatKey | None = None) -> None:
    """Forget a closed rollout. Idempotent."""
    if _by_trace.get(rollout.trace.id) is rollout:
        del _by_trace[rollout.trace.id]
    if seat is not None and seat in _by_seat:
        live = [other for other in _by_seat[seat] if other is not rollout]
        if live:
            _by_seat[seat] = live
        else:
            del _by_seat[seat]


def live_rollout(
    *,
    trace_id: str | None = None,
    episode_id: str | None = None,
    seat: str | None = None,
) -> "Rollout | None":
    """The live rollout for a trace id, or the latest one opened for a seat."""
    if trace_id is not None:
        return _by_trace.get(trace_id)
    assert episode_id is not None and seat is not None
    rollouts = _by_seat.get((episode_id, seat))
    return rollouts[-1] if rollouts else None


async def steer(
    text: str,
    *,
    trace_id: str | None = None,
    episode_id: str | None = None,
    seat: str | None = None,
    sender: str | None = None,
    message_id: str | None = None,
) -> SteerReceipt:
    """Steer a running agent of this process: the rollout of `trace_id`, or the seat
    `seat` of episode `episode_id` (its latest open rollout). The message reaches the
    model as a user message at the agent's next safe point: mid-turn for a harness with
    native steering, else before its next model call. A rollout between interaction
    turns takes it on its next turn's first model call. Returns whether it was
    accepted; it never raises for a missing or finished rollout."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("steering requires a nonempty message")
    by_seat = episode_id is not None or seat is not None
    if (trace_id is not None) == by_seat:
        raise ValueError("steer by trace_id, or by episode_id and seat (not both)")
    if by_seat and (episode_id is None or seat is None):
        raise ValueError("steering a seat takes both episode_id and seat")
    rollout = live_rollout(trace_id=trace_id, episode_id=episode_id, seat=seat)
    if rollout is None:
        target = trace_id if trace_id is not None else f"{seat!r} of {episode_id}"
        return SteerReceipt(queued=False, reason=f"no running agent {target}")
    trace_id = rollout.trace.id
    try:
        receipt = await rollout.steer(
            text, message_id=message_id, sender=sender, queue_when_idle=True
        )
    except (RuntimeError, ValueError) as error:
        return SteerReceipt(queued=False, reason=str(error), trace_id=trace_id)
    if receipt.get("outcome") != "injected":
        return SteerReceipt(
            queued=False,
            reason=str(receipt.get("reason") or receipt.get("outcome")),
            trace_id=trace_id,
        )
    return SteerReceipt(
        queued=True,
        trace_id=trace_id,
        delivery=receipt.get("delivery", "native"),
    )

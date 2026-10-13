"""Steer any running agent: a per-process registry of live rollouts and `steer()`.

Every rollout registers by its trace id while it is open. `steer()` finds the rollout
and delivers a user message to it: through the harness's native mid-turn steering where
it has one (ACP), else through the interception, which adds the message to the
rollout's next model request (`RolloutSession.queue_steer`). Each steer is recorded on
the trace (`Trace.steers`). An episode agent's `message_agent` tool resolves a sibling
seat to its live trace id and steers it here too.

The registry is per process: an env server's `steer` request (`serve.types.SteerRequest`)
reaches the rollouts of the worker process that runs them.
"""

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from verifiers.v1.rollout import Rollout

logger = logging.getLogger(__name__)

_by_trace: dict[str, "Rollout"] = {}


class SteerReceipt(BaseModel):
    """What became of one steer."""

    queued: bool = False
    """Whether a live rollout accepted the message."""
    reason: str | None = None
    """Why it was not accepted."""
    trace_id: str | None = None
    """The rollout it went to."""
    delivery: Literal["native", "interception"] | None = None


def register(rollout: "Rollout") -> None:
    """Make an open rollout steerable."""
    _by_trace[rollout.trace.id] = rollout


def unregister(rollout: "Rollout") -> None:
    """Forget a closed rollout. Idempotent."""
    if _by_trace.get(rollout.trace.id) is rollout:
        del _by_trace[rollout.trace.id]


async def steer(
    text: str,
    *,
    trace_id: str,
    sender: str | None = None,
    message_id: str | None = None,
) -> SteerReceipt:
    """Steer the running rollout of `trace_id` in this process. The message reaches the
    model as a user message at the agent's next safe point: mid-turn for a harness with
    native steering, else before its next model call. A rollout between interaction
    turns takes it on its next turn's first model call. Returns whether it was
    accepted; it never raises for a missing or finished rollout."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("steering requires a nonempty message")
    rollout = _by_trace.get(trace_id)
    if rollout is None:
        return SteerReceipt(queued=False, reason=f"no running agent {trace_id}")
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

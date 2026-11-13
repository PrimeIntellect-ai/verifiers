"""Saving running rollouts, so another process can continue them.

A retry restores a checkpoint in the process that took it, from the live `Checkpoint`.
An owner that outlives its process (a training run that stops and resumes) saves every
running rollout instead (`save_all`): each one takes a memory checkpoint now and
returns its trace as plain data (`restorable`), which holds everything a restore
needs. A later process hands that trace back (`RunSlot.restore`, `Agent.run(restore=)`),
and the rollout continues in a new box from the checkpoint: its trace up to there, and
the harness program the box still runs.

The owner holds the snapshot of a saved rollout (`Checkpoint.pinned`): no rollout
deletes it, and the owner deletes it once nothing will restore it
(`RuntimeBackend.discard`).
"""

from __future__ import annotations

import asyncio
import weakref
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from verifiers.v1.session import ReplayResponse
from verifiers.v1.state import state_cls
from verifiers.v1.trace import Checkpoint, Trace
from verifiers.v1.types import ToolMessage
from verifiers.v1.utils.logging import get_logger

if TYPE_CHECKING:
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.harness import Harness
    from verifiers.v1.rollout import Rollout
    from verifiers.v1.task import Task

logger = get_logger(__name__)

LIVE: weakref.WeakSet[Rollout] = weakref.WeakSet()
"""The rollouts of this process that run a box (between `open` and `close`)."""

restoring: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "vf_restoring", default=None
)
"""The saved trace the episode this task runs continues from (`Env.run_episode`), in a
list its first agent run takes it out of."""


def restorable(trace: Trace, checkpoint: Checkpoint) -> dict[str, Any]:
    """`trace` as plain data that any process can continue from `checkpoint`: its dump
    with token arrays, and on the checkpoint's entry what the checkpoint holds only
    while its rollout lives (`Checkpoint.restore`). Every value survives a pickle or a
    msgpack round trip with raw bytes and numpy arrays."""
    data = trace.model_dump(context={"token_arrays": True})
    entry = next(c for c in data["checkpoints"] if c["id"] == checkpoint.id)
    entry["restore"] = {
        "state": TypeAdapter(type(checkpoint.state)).dump_python(
            checkpoint.state, mode="json"
        ),
        "info": dict(checkpoint.info or {}),
        "pins": [
            (prefix, call_id, message.model_dump(mode="json"))
            for (prefix, call_id), message in checkpoint.pins.items()
        ],
        "replays": {
            key: (path, digest, reply.status, reply.body, reply.content_type)
            for key, ((path, digest), reply) in checkpoint.replays.items()
        },
        "decisions": [
            (
                prefix,
                call_id,
                None if verdict is None else verdict.model_dump(mode="json"),
            )
            for (prefix, call_id), verdict in checkpoint.decisions.items()
        ],
        "gated": sorted(checkpoint.gated),
        "prepared": dict(checkpoint.prepared),
        "live": checkpoint.live,
    }
    return data


def restore_entry(trace: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint entry a saved trace continues from: the newest one that holds
    what a restore needs."""
    for entry in reversed(trace.get("checkpoints") or []):
        if entry.get("restore") is not None:
            return entry
    raise ValueError(f"trace {trace.get('id')} holds no checkpoint to continue from")


def open_restore(data: dict[str, Any], task: Task) -> tuple[Trace, Checkpoint]:
    """The trace and its live checkpoint, for `task`'s rollout to restore."""
    saved = restore_entry(data)
    trace = Trace.model_validate(data)
    checkpoint = next(c for c in trace.checkpoints if c.id == saved["id"])
    held, checkpoint.restore = saved["restore"], None
    trace.state = state_cls(type(task)).model_validate(held["state"])
    checkpoint.state = trace.state.model_copy(deep=True)
    checkpoint.info = dict(held["info"])
    checkpoint.pins = {
        (prefix, call_id): ToolMessage.model_validate(message)
        for prefix, call_id, message in held["pins"]
    }
    checkpoint.replays = {
        key: ((path, digest), ReplayResponse(status, body, content_type))
        for key, (path, digest, status, body, content_type) in held["replays"].items()
    }
    checkpoint.decisions = {
        (prefix, call_id): (
            None if verdict is None else ToolMessage.model_validate(verdict)
        )
        for prefix, call_id, verdict in held.get("decisions", ())
    }
    checkpoint.gated = {(prefix, call_id) for prefix, call_id in held.get("gated", ())}
    checkpoint.prepared = dict(held.get("prepared", {}))
    checkpoint.live = held["live"]
    # The owner saved it and offers it: it is restorable, also for a later retry.
    checkpoint.time.end = checkpoint.time.end or checkpoint.captured
    return trace, checkpoint


def unsaveable(config: AgentConfig, harness: Harness | None = None) -> str | None:
    """Why the rollouts of an agent with `config` take no memory snapshots, so `save_all`
    returns nothing for them; None when they take them. A task with tool
    servers or services of its own rules its rollout out as well, which only the
    rollout knows."""
    from verifiers.v1.runtimes import backend_cls
    from verifiers.v1.utils.loaders import harness_class

    if config.checkpoint is None:
        return "rollout checkpointing is off (agent.checkpoint)"
    if not backend_cls(config.runtime).SUPPORTS_MEMORY_SNAPSHOTS:
        return f"the {config.runtime.type!r} runtime takes no memory snapshots"
    if harness is not None:
        supported, name = harness.SUPPORTS_MEMORY_RESTORE, harness.config.id
    elif config.harness is not None:
        supported = harness_class(config.harness.id).SUPPORTS_MEMORY_RESTORE
        name = config.harness.id
    else:
        return None
    if not supported:
        return f"harness {name!r} cannot continue from a memory snapshot"
    return None


def release_all() -> None:
    """Let every rollout that `save_all(hold=True)` holds go on."""
    for rollout in list(LIVE):
        rollout.release()


async def save_all(
    skipped: dict[str, int] | None = None, *, hold: bool = False
) -> list[dict[str, Any]]:
    """Take a memory checkpoint of every running rollout of this process and return
    their traces as saved (`restorable`). A rollout that takes no memory snapshots,
    runs no harness segment right now, or whose snapshot fails has none: it starts
    over when its owner runs it again. `skipped` collects why, as a count per reason.
    The snapshots are pinned: the caller owns them. With `hold`, each saved rollout
    waits at its checkpoint until `release_all`, so the caller can record which
    rollouts are still running without one of them finishing meanwhile."""

    async def save(rollout: Rollout) -> dict[str, Any] | None:
        try:
            point = await rollout.save(hold=hold)
            reason = rollout.save_skipped
        except Exception:
            logger.warning("saving rollout %s failed", rollout.trace.id, exc_info=True)
            point, reason = None, "its save raised an error"
        if point is None and skipped is not None:
            reason = reason or "unknown"
            skipped[reason] = skipped.get(reason, 0) + 1
        return point

    rollouts = list(LIVE)
    points = [
        point
        for point in await asyncio.gather(*(save(rollout) for rollout in rollouts))
        if point is not None
    ]
    logger.info("Saved %d of %d running rollouts", len(points), len(rollouts))
    return points

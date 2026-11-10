"""Restore points: a running rollout's memory checkpoint in a form another process can
continue.

A retry restores a checkpoint in the process that took it, from the live `Checkpoint`.
An owner that outlives its process (a training run that stops and resumes) saves every
running rollout instead (`save_all`): each one takes a memory checkpoint now and
returns a `RestorePoint`, made of plain data the owner can store. A later process hands
the point back (`RunSlot.restore`, `Agent.run(restore=)`), and the rollout continues in
a new box from the checkpoint: its trace up to there, and the harness program the box
still runs.

The owner holds the snapshot of a restore point (`Checkpoint.pinned`): no rollout
deletes it, and the owner deletes it once nothing will restore it
(`RuntimeBackend.discard`).
"""

from __future__ import annotations

import asyncio
import weakref
from contextvars import ContextVar
from dataclasses import dataclass
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

restoring: ContextVar[list[RestorePoint] | None] = ContextVar(
    "vf_restoring", default=None
)
"""The restore point the episode this task runs continues from (`Env.run_episode`), in
a list its first agent run takes it out of."""


@dataclass
class RestorePoint:
    """A rollout as a memory checkpoint left it, as plain data: every field survives a
    pickle or a msgpack round trip with raw bytes and numpy arrays."""

    trace: dict[str, Any]
    """The trace when the checkpoint was taken, with its token arrays
    (`Trace.model_dump(context={"token_arrays": True})`)."""
    checkpoint: str
    """The id of the checkpoint on `trace` to continue from."""
    state: Any
    """The trace's typed state, as JSON data."""
    info: dict[str, Any]
    pins: list[tuple[str, str, dict[str, Any]]]
    """The pinned tool results: conversation prefix, call id and the message."""
    replays: dict[str, tuple[str, bytes, int, bytes, str]]
    """The replies the box may ask for again: request path and digest, then the reply's
    status, body and content type."""
    live: dict[str, Any]
    """`Checkpoint.live`."""

    @property
    def id(self) -> str:
        """The trace's id."""
        return self.trace["id"]

    @property
    def snapshot(self) -> dict[str, Any]:
        """The box snapshot the point continues from (`Snapshot` as data)."""
        for checkpoint in self.trace["checkpoints"]:
            if checkpoint["id"] == self.checkpoint:
                return checkpoint["snapshot"]
        raise ValueError(f"restore point {self.id} lost its checkpoint")

    @classmethod
    def of(cls, trace: Trace, checkpoint: Checkpoint) -> RestorePoint:
        return cls(
            trace=trace.model_dump(context={"token_arrays": True}),
            checkpoint=checkpoint.id,
            state=TypeAdapter(type(checkpoint.state)).dump_python(
                checkpoint.state, mode="json"
            ),
            info=dict(checkpoint.info or {}),
            pins=[
                (prefix, call_id, message.model_dump(mode="json"))
                for (prefix, call_id), message in checkpoint.pins.items()
            ],
            replays={
                key: (path, digest, reply.status, reply.body, reply.content_type)
                for key, ((path, digest), reply) in checkpoint.replays.items()
            },
            live=checkpoint.live,
        )

    def open(self, task: Task) -> tuple[Trace, Checkpoint]:
        """The trace and its live checkpoint, for `task`'s rollout to restore."""
        trace = Trace.model_validate(self.trace)
        checkpoint = next(c for c in trace.checkpoints if c.id == self.checkpoint)
        trace.state = state_cls(type(task)).model_validate(self.state)
        checkpoint.state = trace.state.model_copy(deep=True)
        checkpoint.info = dict(self.info)
        checkpoint.pins = {
            (prefix, call_id): ToolMessage.model_validate(message)
            for prefix, call_id, message in self.pins
        }
        checkpoint.replays = {
            key: ((path, digest), ReplayResponse(status, body, content_type))
            for key, (path, digest, status, body, content_type) in self.replays.items()
        }
        checkpoint.live = self.live
        return trace, checkpoint


def unsaveable(config: AgentConfig, harness: Harness | None = None) -> str | None:
    """Why the rollouts of an agent with `config` take no memory snapshots, so `save_all`
    returns no restore point for them; None when they take them. A task with tool
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
) -> list[RestorePoint]:
    """Take a memory checkpoint of every running rollout of this process and return
    their restore points. A rollout that takes no memory snapshots, runs no harness
    segment right now, or whose snapshot fails has none: it starts over when its owner
    runs it again. `skipped` collects why, as a count per reason. The snapshots are
    pinned: the caller owns them. With `hold`, each saved rollout waits at its restore
    point until `release_all`, so the caller can record which rollouts are still
    running without one of them finishing meanwhile."""

    async def save(rollout: Rollout) -> RestorePoint | None:
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

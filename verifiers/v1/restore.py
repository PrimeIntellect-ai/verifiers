"""Saving running rollouts, so another process can continue them.

A retry restores a checkpoint in the process that took it, from the live `Checkpoint`.
An owner that outlives its process (a training run that stops and resumes) saves every
running rollout instead (`save_all`): each one supplies a memory checkpoint and
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
import functools
import time
import weakref
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from verifiers.v1.state import state_cls
from verifiers.v1.trace import Checkpoint, Trace
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
    with token arrays, and on the checkpoint's entry what the host held beside the trace
    (`Checkpoint.session`). Every value survives a pickle or a msgpack round trip with
    raw bytes and numpy arrays."""
    data = trace.model_dump(context={"token_arrays": True})
    # Only this snapshot is transferred. A fork must never retry from, or
    # garbage-collect, another live rollout's private snapshots.
    entry = next(c for c in data["checkpoints"] if c["id"] == checkpoint.id)
    data["checkpoints"] = [entry]
    entry["session"] = checkpoint.session
    return data


def restore_entry(trace: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint entry a saved trace continues from: the newest one that holds
    what a restore needs."""
    for entry in reversed(trace.get("checkpoints") or []):
        if entry.get("session") is not None:
            return entry
    raise ValueError(f"trace {trace.get('id')} holds no checkpoint to continue from")


def open_restore(data: dict[str, Any], task: Task) -> tuple[Trace, Checkpoint]:
    """The trace and its checkpoint, for `task`'s rollout to restore."""
    saved = restore_entry(data)
    trace = Trace.model_validate(data)
    checkpoint = next(c for c in trace.checkpoints if c.id == saved["id"])
    assert checkpoint.session is not None
    trace.state = state_cls(type(task)).model_validate(checkpoint.session["state"])
    # The owner saved it and offers it: it is restorable, also for a later retry.
    checkpoint.time.end = checkpoint.time.end or checkpoint.captured
    return trace, checkpoint


def unsaveable(config: AgentConfig, harness: Harness | None = None) -> str | None:
    """Why the rollouts of an agent with `config` take no memory snapshots, so `save_all`
    returns nothing for them; None when they take them. A task with services, shared
    tools or tool servers off Prime VMs of their own rules its rollout out as well,
    which only the rollout knows."""
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
    skipped: dict[str, int] | None = None,
    *,
    hold: bool = False,
    watchdog: bool = True,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Take a memory checkpoint of every running rollout of this process and return
    their traces as saved (`restorable`). A rollout that takes no memory snapshots,
    runs no harness segment right now, or whose snapshot fails has none: it starts
    over when its owner runs it again. `skipped` collects why, as a count per reason.
    The snapshots are pinned: the caller owns them. With `hold`, each saved rollout
    waits for `release_all` and may export a recent periodic checkpoint, so the caller can record which
    rollouts are still running without one of them finishing meanwhile. Each hold
    ends by itself `HOLD_SECONDS` after the save returns, unless `watchdog` is False:
    the caller then arms the watchdogs (`arm_hold_watchdogs`) once it handed the saves
    over. A save of many rollouts can take minutes, so a watchdog armed at each
    rollout's own save would let the first ones go on before the caller has them.
    A rollout whose save takes longer than `timeout` seconds has no restore point:
    its save goes on, and gives its snapshot and hold back to the rollout when it
    ends."""

    async def save(rollout: Rollout) -> dict[str, Any] | None:
        try:
            point = await rollout.save(hold=hold, watchdog=False)
            reason = rollout.save_skipped
        except Exception:
            logger.warning("saving rollout %s failed", rollout.trace.id, exc_info=True)
            point, reason = None, "its save raised an error"
        if point is None and skipped is not None:
            reason = reason or "unknown"
            skipped[reason] = skipped.get(reason, 0) + 1
        return point

    rollouts = list(LIVE)
    started = time.monotonic()
    saves = [asyncio.ensure_future(save(rollout)) for rollout in rollouts]
    try:
        late = (await asyncio.wait(saves, timeout=timeout))[1] if saves else set()
    except BaseException:
        # The caller never gets the saves done so far: nobody would release their
        # rollouts or delete their snapshots, so both go back to the rollouts.
        for rollout, task in zip(rollouts, saves):
            if not task.done():
                task.cancel()
            elif (
                not task.cancelled()
                and task.exception() is None
                and (point := task.result()) is not None
            ):
                rollout.disown(point)
        raise
    points = []
    for rollout, task in zip(rollouts, saves):
        if task in late:
            task.add_done_callback(functools.partial(_disown_late, rollout))
            if skipped is not None:
                reason = f"its save took longer than {timeout:g}s"
                skipped[reason] = skipped.get(reason, 0) + 1
        elif (point := task.result()) is not None:
            points.append(point)
    logger.info(
        "Saved %d of %d running rollouts in %.1fs",
        len(points),
        len(rollouts),
        time.monotonic() - started,
    )
    if hold and watchdog:
        arm_hold_watchdogs()
    return points


def _disown_late(rollout: Rollout, task: asyncio.Future) -> None:
    if not task.cancelled() and (point := task.result()) is not None:
        rollout.disown(point)


def arm_hold_watchdogs() -> None:
    """Start the self-release of every rollout a `save_all(hold=True)` holds, for an
    owner that does more work before it can `release_all`."""
    for rollout in list(LIVE):
        rollout.arm_hold_watchdog()

"""Saving running rollouts, so another process can continue them.

A retry restores a checkpoint in the process that took it, from the live `Checkpoint`.
An owner that outlives its process (a training run that stops and resumes) saves every
running rollout instead (`save_all`): each one supplies a memory checkpoint and
returns its trace as plain data (`restorable`), which holds everything a restore
needs. A later process hands that trace back (`RunSlot.restore`, `Agent.run(restore=)`),
and the rollout continues in a new box from the checkpoint: its trace up to there, and
the harness program the box still runs.

An episode of several agent runs continues the same way (`episode_restore_point`). The
env's program runs again from the start, and each agent call, keyed by its seat and its
number within the episode, replays: a call that had finished returns its saved trace, a
call that was running continues from its checkpoint, and a call that had not started
runs fresh (`EpisodeCalls`).

The owner holds the snapshot of a saved rollout (`Checkpoint.pinned`): no rollout
deletes it, and the owner deletes it once nothing will restore it
(`RuntimeBackend.discard`).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import time
import weakref
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from verifiers.v1.state import state_cls
from verifiers.v1.trace import Checkpoint, Trace
from verifiers.v1.utils.logging import get_logger

if TYPE_CHECKING:
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.env import RunSlot
    from verifiers.v1.harness import Harness
    from verifiers.v1.rollout import Rollout
    from verifiers.v1.task import Task

logger = get_logger(__name__)

LIVE: weakref.WeakSet[Rollout] = weakref.WeakSet()
"""The rollouts of this process that run a box (between `open` and `close`)."""

EPISODE_POINT_VERSION = 1

CallKey = tuple[str, int]
"""An agent call within an episode: its seat, and how many calls that seat made before."""


class ReplayDiverged(Exception):
    """A continued episode asked for a call the restore point does not hold: the env's
    program took another path than in the saved episode."""


def fingerprint(task: Task) -> str:
    """What an agent call was asked to do, to tell that a replayed call is the saved one."""
    data = task.data.model_dump_json(exclude_none=True)
    return hashlib.sha256(f"{type(task).__qualname__}:{data}".encode()).hexdigest()


@dataclass
class _Call:
    seat: str
    n: int
    fingerprint: str
    parent: CallKey | None
    """The call whose scoring made this one (an agentic judge's), if any."""
    trace: Trace | None = None
    finished: bool = False
    interaction: bool = False


@dataclass
class EpisodeCalls:
    """The agent calls of one episode attempt, in the order they started, and the saved
    calls a continued episode replays (`memo`, from `episode_restore_point`)."""

    memo: dict[CallKey, dict[str, Any]] = field(default_factory=dict)
    calls: dict[CallKey, _Call] = field(default_factory=dict)
    counts: Counter[str] = field(default_factory=Counter)
    diverged: str | None = None
    """Why the replay stopped, once a call did not match the saved one."""

    @classmethod
    def continuing(cls, point: dict[str, Any] | None) -> EpisodeCalls:
        """The calls of an attempt that continues `point`: an episode restore point, or
        the saved trace of a single rollout (`restorable`), which continues the first
        call of its seat."""
        if point is None:
            return cls()
        if "episode_point" not in point:
            seat = (point.get("agent") or {}).get("name") or "agent"
            point = {
                "calls": [
                    {
                        "seat": seat,
                        "n": 0,
                        "fingerprint": None,
                        "parent": None,
                        "state": "running",
                        "trace": point,
                    }
                ]
            }
        return cls(memo={(c["seat"], c["n"]): c for c in point["calls"]})

    def start(
        self,
        seat: str,
        task: Task,
        parent: CallKey | None,
        *,
        interaction: bool = False,
    ) -> tuple[CallKey, dict[str, Any] | None]:
        """Key a new call of `seat` and return what the restore point holds for it.
        Raises `ReplayDiverged` when the saved call was asked to do something else. An
        `interaction` is keyed and recorded, but never saved, so it replays nothing."""
        key = (seat, self.counts[seat])
        self.counts[seat] += 1
        call = _Call(seat, key[1], fingerprint(task), parent, interaction=interaction)
        self.calls[key] = call
        saved = None if interaction else self.memo.pop(key, None)
        if saved is None:
            return key, None
        if (
            saved["fingerprint"] is not None
            and saved["fingerprint"] != call.fingerprint
        ):
            self.diverged = (
                f"{seat} call {key[1]} was asked for another task than at the save"
            )
            raise ReplayDiverged(self.diverged)
        return key, saved

    def minted(self, key: CallKey, trace: Trace) -> None:
        self.calls[key].trace = trace

    def finished(self, key: CallKey, trace: Trace) -> None:
        call = self.calls[key]
        call.trace, call.finished = trace, True

    def replay(
        self, key: CallKey, saved: dict[str, Any], task: Task
    ) -> tuple[Trace, list[Trace]]:
        """The finished call `saved`, and the calls its scoring made, in the order they
        finished, as traces for the episode. They keep their keys, so a later save of
        this episode holds them again."""
        children = sorted(
            (c for c in self.memo.values() if c["parent"] == [*key]),
            key=lambda c: c["order"],
        )
        nested: list[Trace] = []
        for child in children:
            child_key = (child["seat"], child["n"])
            self.memo.pop(child_key)
            self.counts[child["seat"]] = max(self.counts[child["seat"]], child["n"] + 1)
            trace = _finished_trace(child, None)
            self.calls[child_key] = _Call(
                child["seat"], child["n"], child["fingerprint"], key, trace, True
            )
            nested.append(trace)
        trace = _finished_trace(saved, task)
        self.finished(key, trace)
        return trace, nested


def _finished_trace(saved: dict[str, Any], task: Task | None) -> Trace:
    trace = Trace.model_validate(saved["trace"])
    if task is not None and saved.get("task_state") is not None:
        trace.state = state_cls(type(task)).model_validate(saved["task_state"])
    return trace


def scoring_point(trace: Trace, snapshot: dict[str, Any]) -> dict[str, Any]:
    """`trace`, of a rollout whose agent is done and which is in scoring, as a saved
    trace (`restorable`) that continues its scoring in a box restored from `snapshot`:
    the box another run of its scoring took while it ran in this rollout's box. The
    restored rollout scores again, without running its agent (`Rollout.resumes_scoring`)."""
    data = trace.model_dump(context={"token_arrays": True})
    session = {
        "stage": "scoring",
        "finalized": bool(trace.timing.finalize.end),
        "stop": [trace.stop_condition, trace.stop_info],
        "state": TypeAdapter(type(trace.state)).dump_python(trace.state, mode="json"),
        "info": copy.deepcopy(trace.info),
        "rewrites": (len(trace.request_rewrites), len(trace.response_rewrites)),
        "pins": [],
        "decisions": [],
        "gated": [],
        "prepared": {},
        "replays": {},
    }
    data["checkpoints"] = [
        {
            "id": snapshot["id"],
            "snapshot": snapshot,
            "node": len(trace.nodes) - 1 if trace.nodes else None,
            "session": session,
            "captured": time.time(),
            "pinned": True,
        }
    ]
    return data


def _skip_reason(trace_id: str) -> str:
    for rollout in list(LIVE):
        if rollout.trace.id == trace_id and rollout.save_skipped:
            return rollout.save_skipped
    return "it took no memory snapshot"


def episode_restore_point(
    slot: RunSlot, saved: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    """The restore point of `slot`'s running episode, built from the rollouts `save_all`
    just saved, or None and why the episode has none. The point holds every call that
    finished with its trace, and every running call with its saved trace; a call that
    made no turn yet runs fresh on replay. It is all or nothing: a running call that
    made turns but has no saved trace leaves the episode without a restore point.
    `point["snapshots"]` are the snapshots it continues from, which the caller owns;
    the caller also owns every saved rollout the point does not use."""
    calls = slot.calls
    if calls is None:
        return None, "it has not started"
    by_trace = {trace["id"]: trace for trace in saved}
    nested = {call.parent for call in calls.calls.values() if call.parent is not None}

    def scoring_box(key: CallKey, trace: Trace) -> dict[str, Any] | None:
        """The snapshot of `key`'s box that a running call of its scoring took, which
        holds that call and `key`'s scored work."""
        box = trace.agent.runtime.id if trace.agent.runtime is not None else None
        for child in calls.calls.values():
            saved_child = by_trace.get(child.trace.id) if child.trace else None
            runtime = child.trace.agent.runtime if child.trace else None
            if (
                child.parent == key
                and saved_child is not None
                and runtime is not None
                and runtime.id == box
            ):
                return restore_entry(saved_child)["snapshot"]
        return None

    entries: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for order, (key, call) in enumerate(calls.calls.items()):
        trace = call.trace
        if call.interaction:
            reason = "it is an interaction, which no checkpoint can continue"
            return None, f"{call.seat} call {call.n}: {reason}"
        if trace is None:
            continue
        entry: dict[str, Any] = {
            "seat": call.seat,
            "n": call.n,
            "fingerprint": call.fingerprint,
            "parent": None if call.parent is None else [*call.parent],
            "order": order,
        }
        if call.finished:
            try:
                state = TypeAdapter(type(trace.state)).dump_python(
                    trace.state, mode="json"
                )
            except Exception:  # noqa: BLE001 - the state's values decide it
                return (
                    None,
                    f"{call.seat} call {call.n}: its task state cannot be serialized",
                )
            entries.append(
                {
                    **entry,
                    "state": "finished",
                    "trace": trace.model_dump(context={"token_arrays": True}),
                    "task_state": state,
                }
            )
            continue
        point = by_trace.get(trace.id)
        if point is None and key in nested and (box := scoring_box(key, trace)):
            try:
                point = scoring_point(trace, box)
            except Exception:  # noqa: BLE001 - the state's values decide it
                reason = "its task state cannot be serialized"
                return None, f"{call.seat} call {call.n}: {reason}"
        if point is None:
            if call.parent is not None or key in nested:
                return (
                    None,
                    f"{call.seat} call {call.n}: it runs inside another run's scoring",
                )
            if trace.num_turns == 0:
                continue
            return None, f"{call.seat} call {call.n}: {_skip_reason(trace.id)}"
        entries.append({**entry, "state": "running", "trace": point})
        snapshot = restore_entry(point)["snapshot"]
        # A call in scoring continues from the snapshot its scoring's call took.
        if all(s["id"] != snapshot["id"] for s in snapshots):
            snapshots.append(snapshot)
    point = {
        "episode_point": EPISODE_POINT_VERSION,
        "episode_id": slot.id,
        "calls": entries,
        "snapshots": snapshots,
    }
    return point, None


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
    skipped: dict[str, int] | None = None,
    *,
    hold: bool = False,
    watchdog: bool = True,
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
    rollout's own save would let the first ones go on before the caller has them."""

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
    saves = [asyncio.ensure_future(save(rollout)) for rollout in rollouts]
    try:
        results = await asyncio.gather(*saves)
    except BaseException:
        # The caller never gets the saves done so far: nobody would release their
        # rollouts or delete their snapshots, so both go back to the rollouts.
        for rollout, task in zip(rollouts, saves):
            if (
                task.done()
                and not task.cancelled()
                and task.exception() is None
                and (point := task.result()) is not None
            ):
                rollout.disown(point)
        raise
    points = [point for point in results if point is not None]
    logger.info("Saved %d of %d running rollouts", len(points), len(rollouts))
    if hold and watchdog:
        arm_hold_watchdogs()
    return points


def arm_hold_watchdogs() -> None:
    """Start the self-release of every rollout a `save_all(hold=True)` holds, for an
    owner that does more work before it can `release_all`."""
    for rollout in list(LIVE):
        rollout.arm_hold_watchdog()

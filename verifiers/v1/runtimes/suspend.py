"""Suspend instead of delete when a process shuts down; opt in with `VF_SUSPEND_DIR`.

An env server that stops (SIGTERM, or its parent gone) tears its rollouts down, and the
teardown deletes their boxes. With `VF_SUSPEND_DIR` set, a rollout another process can
take over in its live box (`Trace.live`) keeps the box instead: its runtime pauses it
(processes and memory kept, only the disk billed) and leaves it, recorded one JSON line
per box in `<dir>/<hostname>-<pid>.jsonl`, for an orchestrator to re-adopt from its
restore point or delete (`pause_boxes`, `delete_boxes`). The tunnel the box's program
reaches stays registered with it. A run cancelled on its own still deletes its box: only
a process shutdown suspends, or a run whose client is gone (`run_scope`), which is how a
dead orchestrator looks from a server that outlives it for a while.
"""

import asyncio
import json
import logging
import os
import socket
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.runtimes.base import RuntimeBackend

logger = logging.getLogger(__name__)

SUSPEND_DIR = "VF_SUSPEND_DIR"
"""The opt-in: the directory the suspended boxes are recorded in."""
PAUSE_BUDGET = 60.0
"""Seconds from the shutdown's start every pause shares."""

_shutdown_at: float | None = None
_live: dict[int, tuple[str, str | None]] = {}
"""Per runtime backend (by id) holding an adoptable rollout: its trace and tunnel."""
_kept_tunnels: set[str] = set()
_secrets: dict[str, tuple[int, "RunScope | None"]] = {}
"""Per adoptable rollout's secret: its runtime backend (by id) and server run scope."""


class RunScope:
    """One server run: `lost` once its client is gone."""

    lost = False


_scope: ContextVar[RunScope | None] = ContextVar("vf_suspend_scope", default=None)


def run_scope() -> RunScope:
    """Open the scope of the server run in the current task: the tasks it starts share it,
    so its teardown suspends the box once the scope is `lost`."""
    scope = RunScope()
    _scope.set(scope)
    return scope


def shutting_down() -> None:
    """Mark the process as shutting down (its SIGTERM handler): teardowns from now on
    suspend the boxes they would delete, when opted in."""
    global _shutdown_at
    if _shutdown_at is None:
        _shutdown_at = time.monotonic()


def suspending() -> bool:
    if not os.environ.get(SUSPEND_DIR):
        return False
    scope = _scope.get()
    return _shutdown_at is not None or (scope is not None and scope.lost)


def track(
    backend: "RuntimeBackend",
    trace_id: str,
    tunnel_id: str | None,
    secrets: tuple[str, ...] = (),
) -> None:
    """`backend` holds a rollout another process can adopt (`Trace.live`), whose program
    authenticates with `secrets`."""
    _live[id(backend)] = (trace_id, tunnel_id)
    scope = _scope.get()
    for secret in secrets:
        _secrets[secret] = (id(backend), scope)


def untrack(backend: "RuntimeBackend") -> None:
    _live.pop(id(backend), None)
    for secret, (owner, _) in list(_secrets.items()):
        if owner == id(backend):
            del _secrets[secret]


def held(secret: str) -> bool:
    """Whether a request under `secret` must wait rather than be refused: its rollout's
    box is being suspended. Between its session closing and its box pausing, its program
    may still ask; a refusal would end it, a request that never returns is frozen with
    it and retried on resume against the process that adopts it."""
    entry = _secrets.get(secret)
    if entry is None or not os.environ.get(SUSPEND_DIR):
        return False
    scope = entry[1]
    return _shutdown_at is not None or (scope is not None and scope.lost)


def keeps_tunnel(tunnel_id: str | None) -> bool:
    """Whether a suspended (or suspending) box's program reaches this tunnel, so its
    registration must outlive the process."""
    if not suspending() or tunnel_id is None:
        return False
    return tunnel_id in _kept_tunnels or any(t == tunnel_id for _, t in _live.values())


def claims(backend: "RuntimeBackend") -> bool:
    """Whether a shutdown keeps `backend`'s box rather than deleting it."""
    return suspending() and backend.SUPPORTS_PAUSE and id(backend) in _live


async def suspend(backend: "RuntimeBackend") -> bool:
    """At a shutdown (or once the run's client is gone), pause `backend`'s box instead of
    deleting it, within what is left of `PAUSE_BUDGET`, and record it. Returns whether the box is kept: a box whose pause
    failed is kept too, unpaused. Marks the backend `suspended` before the first await,
    so the atexit backstop leaves it alone as well."""
    if not claims(backend):
        return False
    backend.suspended = True
    start = _shutdown_at if _shutdown_at is not None else time.monotonic()
    left = max(0.0, start + PAUSE_BUDGET - time.monotonic())
    try:
        async with asyncio.timeout(left):
            await backend.pause()
        paused = True
    except Exception as e:  # noqa: BLE001 - kept either way
        logger.warning("suspend: pausing sandbox %s failed: %s", backend.info.id, e)
        paused = False
    _record(backend, paused)
    return True


def suspend_at_exit(backend: "RuntimeBackend") -> bool:
    """The atexit backstop's `suspend`: no loop to pause on, so a claimed box is kept
    unpaused (or as an earlier `suspend` left it)."""
    if backend.suspended:
        return True
    if not claims(backend):
        return False
    backend.suspended = True
    _record(backend, False)
    return True


def _record(backend: "RuntimeBackend", paused: bool) -> None:
    trace_id, tunnel_id = _live[id(backend)]
    if tunnel_id is not None:
        _kept_tunnels.add(tunnel_id)
    directory = os.environ[SUSPEND_DIR]
    line = {
        "sandbox_id": backend.info.id,
        "trace_id": trace_id,
        "tunnel_id": tunnel_id,
        "paused": paused,
        "time": time.time(),
    }
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{socket.gethostname()}-{os.getpid()}.jsonl")
        with open(path, "a") as manifest:
            manifest.write(json.dumps(line) + "\n")
    except OSError as e:
        logger.warning("suspend: recording %s failed: %s", line, e)
    logger.info(
        "suspend: sandbox %s kept (%s)",
        backend.info.id,
        "paused" if paused else "running",
    )

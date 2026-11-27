"""Pausing a running episode, so an owner that sheds inference load keeps its progress.

A paused episode sends no new model call upstream: each call of its rollouts waits at
the interception server until the episode resumes. A call already upstream finishes.
The boxes keep running, and the time paused does not count against the agents'
deadlines.
"""

from __future__ import annotations

import asyncio
import weakref
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.rollout import Rollout


class Pause:
    """One episode's pause switch (`RunSlot.pause`). Every rollout of the episode,
    retries included, follows it."""

    def __init__(self) -> None:
        self.running = asyncio.Event()
        """Set while the episode runs; its model calls wait for it."""
        self.running.set()
        self._rollouts: weakref.WeakSet[Rollout] = weakref.WeakSet()

    @property
    def paused(self) -> bool:
        return not self.running.is_set()

    def pause(self) -> None:
        if self.paused:
            return
        self.running.clear()
        for rollout in list(self._rollouts):
            rollout.set_paused(True)

    def resume(self) -> None:
        if not self.paused:
            return
        self.running.set()
        for rollout in list(self._rollouts):
            rollout.set_paused(False)

    def attach(self, rollout: Rollout) -> None:
        self._rollouts.add(rollout)
        if self.paused:
            rollout.set_paused(True)


pausing: ContextVar[Pause | None] = ContextVar("vf_pausing", default=None)
"""The pause switch of the episode this task runs (`Env.run_slot`)."""

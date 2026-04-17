"""Contract for multi-actor rubrics.

Replaces flat ``state["metrics"]["reward/{mid}"]`` string-keyed convention with
a structured three-slot schema (member_rewards / member_metrics /
episode_metrics) that the orchestrator bridge can read without string matching.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import State


class MultiAgentRubric(Rubric):
    """Base class for multi-actor scoring.

    Subclasses MUST implement ``score_rollout(state)`` and populate:

      state["member_rewards"]:  dict[str, float]
          Per-member scalar reward.
      state["member_metrics"]:  dict[str, dict[str, float]]
          Per-member named metrics (commits count lives here, keyed
          ``member_metrics[mid]["commits"]``).
      state["episode_metrics"]: dict[str, float]
          Episode-level metrics (agreement, winner index, ...).

    The base class provides a defensive ``score_group`` boundary: per-rollout
    ``try/except`` catching ``vf.Error`` (including ``KernelProtocolError``),
    writing the error onto ``state["error"]`` with default-initialized
    structured keys. A failure in one rollout must not prevent scoring of
    any other rollout in the group.
    """

    members: list[str]

    @abstractmethod
    async def score_rollout(self, state: State) -> None: ...

    async def score_group(self, states: list[State]) -> None:
        async def _safe_score(s: State) -> None:
            try:
                await self.score_rollout(s)
            except vf.Error as e:
                s["error"] = e
                s.setdefault("member_rewards", {m: 0.0 for m in self.members})
                s.setdefault("member_metrics", {m: {} for m in self.members})
                s.setdefault("episode_metrics", {})

        await asyncio.gather(*(_safe_score(s) for s in states))

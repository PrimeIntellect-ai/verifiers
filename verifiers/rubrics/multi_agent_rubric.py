"""Contract for multi-actor rubrics.

Adds per-member reward attribution to the base ``Rubric`` contract via a
single additional state key, ``state["member_rewards"]``. The flat
``state["metrics"]`` dict (populated by the subclass) remains the logging
channel consumed by the orchestrator; this contract only carves out the
one piece the bridge needs without string-matching ``reward/{mid}``.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import State


class MultiAgentRubric(Rubric):
    """Base class for multi-actor scoring.

    Subclasses MUST implement ``score_rollout(state)`` and populate
    ``state["member_rewards"]: dict[str, float]`` covering every member.

    The base class provides a defensive ``score_group`` boundary: per-
    rollout ``try/except`` catching ``vf.Error`` (including
    ``KernelProtocolError``), recording the error onto ``state["error"]``
    with zero-filled member rewards so the bridge doesn't KeyError. A
    failure in one rollout must not prevent scoring any other rollout in
    the group.
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

        await asyncio.gather(*(_safe_score(s) for s in states))

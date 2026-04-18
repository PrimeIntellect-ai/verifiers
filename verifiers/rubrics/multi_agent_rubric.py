"""Contract for multi-actor rubrics.

Subclasses MUST populate ``state["mar_score"]: MARScore`` covering every
member. The bridge reads ``mar_score`` directly; ``state_to_output``
projects it to legacy keys (``output["reward"]``, flat top-level metrics)
at the serialization boundary so downstream consumers (wandb, GRPO
advantage) see the same shape as single-actor rubrics.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import MARScore, MemberScore, State


class MultiAgentRubric(Rubric):
    """Base class for multi-actor scoring.

    Subclasses MUST implement ``score_rollout(state)`` and write
    ``state["mar_score"]: MARScore`` covering every member.

    The base class provides a defensive ``score_group`` boundary: per-
    rollout ``try/except`` catching ``vf.Error`` (including
    ``KernelProtocolError``), recording the error onto ``state["error"]``
    with a zero-reward MARScore so the bridge doesn't KeyError. A
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
                if "mar_score" not in s:
                    s["mar_score"] = MARScore(
                        members=[
                            MemberScore(member_id=mid, role_id=mid, reward=0.0)
                            for mid in self.members
                        ],
                        episode_scalar=0.0,
                        episode_metrics={
                            "errored_rollout": 1.0,
                            "error_type": type(e).__name__,
                            "error_phase": "scoring",
                        },
                    )

        await asyncio.gather(*(_safe_score(s) for s in states))

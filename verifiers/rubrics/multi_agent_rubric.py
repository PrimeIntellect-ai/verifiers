"""Contract for multi-agent rubrics.

Subclasses build and return a ``MARScore`` covering every member. The
base class writes it onto ``state["mar_score"]`` and owns the rollout /
group error boundary. The bridge reads ``mar_score`` directly;
``state_to_output`` projects it to legacy keys (``output["reward"]``,
flat top-level metrics) at the serialization boundary so downstream
consumers (wandb, GRPO advantage) see the same shape as single-agent
rubrics.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Mapping

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import MARScore, MemberScore, State, TrajectoryStep


class MultiAgentRubric(Rubric):
    """Base class for multi-agent scoring.

    Subclasses implement ``build_marscore(state) -> MARScore``. The base
    class owns the rollout/group boundary:

    - rollout-layer broken states (``prompt_too_long`` or existing
      ``state["error"]``) short-circuit to a zero-reward MARScore
    - scoring-time ``vf.Error`` is recorded onto ``state["error"]`` and
      converted into a zero-reward MARScore
    - non-``vf.Error`` exceptions propagate loud
    """

    members: list[str]

    @abstractmethod
    async def build_marscore(self, state: State) -> MARScore: ...

    @staticmethod
    def split_by_member(
        state_or_output: Mapping[str, Any],
    ) -> dict[str, list[TrajectoryStep]]:
        """Group trajectory steps by ``extras['member_id']``.

        Shared primitive used by the rubric (scoring) and the bridge
        (per-member training split). Every trajectory step in a
        multi-agent rollout carries ``extras['member_id']`` —
        ``MultiAgentEnv._build_step`` guarantees this at rollout time.
        A missing value signals a framework-invariant violation.
        """
        out: dict[str, list[TrajectoryStep]] = defaultdict(list)
        for step in state_or_output.get("trajectory", []):
            mid = step.get("extras", {}).get("member_id")
            if mid is None:
                raise ValueError(
                    f"TrajectoryStep missing extras['member_id']: {step!r}"
                )
            out[mid].append(step)
        return dict(out)

    def build_errored_marscore(
        self, state: State, *, error_type: str, error_phase: str
    ) -> MARScore:
        return MARScore(
            members=[MemberScore(member_id=mid, reward=0.0) for mid in self.members],
            episode_scalar=0.0,
            episode_metrics={"errored_rollout": 1.0},
            episode_error={"error_type": error_type, "error_phase": error_phase},
        )

    async def score_rollout(self, state: State) -> None:
        if state.get("prompt_too_long", False):
            state["mar_score"] = self.build_errored_marscore(
                state, error_type="prompt_too_long", error_phase="rollout"
            )
            return

        existing_error = state.get("error")
        if existing_error is not None:
            state["mar_score"] = self.build_errored_marscore(
                state,
                error_type=type(existing_error).__name__,
                error_phase="rollout",
            )
            return

        try:
            state["mar_score"] = await self.build_marscore(state)
        except vf.Error as error:
            state["error"] = error
            state["mar_score"] = self.build_errored_marscore(
                state,
                error_type=type(error).__name__,
                error_phase="scoring",
            )

    async def score_group(self, states: list[State]) -> None:
        await asyncio.gather(*(self.score_rollout(state) for state in states))

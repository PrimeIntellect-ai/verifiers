"""Split a multi-agent RolloutOutput into per-member training rollouts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from .types import MARScore, MemberRollout, RolloutOutput, TrajectoryStep


def rollout_to_member_rollouts(
    output: RolloutOutput | Mapping[str, Any],
    env_name: str,
) -> list[MemberRollout]:
    """Project one episode-level rollout into one rollout per member."""
    mar_raw = output["mar_score"]
    mar = (
        mar_raw if isinstance(mar_raw, MARScore) else MARScore.model_validate(mar_raw)
    )

    sampling_args = output["sampling_args"]
    temperature = sampling_args["temperature"]
    example_id = output["example_id"]
    episode_id = output.get("trajectory_id", "")
    rollout_error = output.get("error")
    trajectory: list[TrajectoryStep] = output.get("trajectory", [])

    steps_by_member: dict[str, list[TrajectoryStep]] = defaultdict(list)
    for step in trajectory:
        extras = step.get("extras", {})
        member_id = extras.get("member_id")
        if member_id is None:
            raise ValueError(
                f"TrajectoryStep missing extras['member_id']: {step!r}"
            )
        steps_by_member[member_id].append(step)

    return [
        MemberRollout(
            example_id=example_id,
            task=env_name,
            trajectory=steps_by_member.get(member.member_id, []),
            sampling_args={"temperature": temperature},
            error=rollout_error,
            reward=member.reward,
            episode_id=episode_id,
            member_id=member.member_id,
            role_id=member.role_id,
        )
        for member in mar.members
    ]

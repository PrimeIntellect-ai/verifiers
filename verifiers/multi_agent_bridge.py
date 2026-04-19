"""Split a multi-agent RolloutOutput into per-member training rollouts."""

from __future__ import annotations

from typing import Any, Mapping

from .types import MARScore, MemberRollout, RolloutOutput


def rollout_to_member_rollouts(
    output: RolloutOutput | Mapping[str, Any],
) -> list[MemberRollout]:
    """Project one episode-level rollout into one rollout per member.

    Inputs are expected to be a ``RolloutOutput`` produced by
    ``state_to_output``, which guarantees ``task``, ``example_id``,
    ``sampling_args``, ``trajectory_id`` and ``mar_score``. Missing any
    of these is a contract violation upstream — we ``KeyError`` rather
    than silently substituting defaults that would corrupt training
    identity (RAE baselines key on ``(task, example_id, member_id)``).

    ``trajectory`` is ALSO required and must be present as a key on the
    output. ``state_to_output`` only includes it when the caller passes
    ``state_columns=["trajectory", ...]``; absent trajectory would cause
    ``split_by_member`` to silently return ``{}`` and the bridge to emit
    empty-trajectory member rollouts — silently discarding all
    token-level training data. Fail loud instead.

    Trajectory member ids must match ``MARScore.members`` exactly: any
    extra ``extras['member_id']`` would be silently dropped by the
    per-member projection, masking a rubric/env wiring bug.
    """
    # Lazy import: verifiers/__init__.py loads this module before Parser
    # is bound, and MultiAgentRubric transitively pulls Rubric which
    # resolves vf.Parser at class-body evaluation time.
    from .rubrics.multi_agent_rubric import MultiAgentRubric

    mar_raw = output["mar_score"]
    mar = mar_raw if isinstance(mar_raw, MARScore) else MARScore.model_validate(mar_raw)

    task = output["task"]
    example_id = output["example_id"]
    sampling_args = output["sampling_args"]
    episode_id = output["trajectory_id"]
    rollout_error = output.get("error")

    if "trajectory" not in output:
        raise KeyError(
            "trajectory missing from RolloutOutput — the multi-agent "
            "bridge requires per-step trajectory data. Add 'trajectory' "
            "to state_columns when saving outputs for training (e.g. "
            "``--state-columns trajectory,mar_score``)."
        )

    # vf-eval doesn't always populate temperature in saved metadata when
    # the user passes --sampling-args without it. Default to OpenAI's
    # canonical 1.0 rather than crashing on the first such rollout.
    member_sampling_args = dict(sampling_args)
    member_sampling_args.setdefault("temperature", 1.0)

    steps_by_member = MultiAgentRubric.split_by_member(output)
    expected_members = {m.member_id for m in mar.members}
    extra_members = set(steps_by_member) - expected_members
    if extra_members:
        raise ValueError(
            f"trajectory contains member_id(s) {sorted(extra_members)} not in "
            f"MARScore.members={sorted(expected_members)}; rubric/env "
            "out-of-sync — silently dropping these steps would corrupt "
            "training data"
        )

    return [
        MemberRollout(
            example_id=example_id,
            task=task,
            trajectory=steps_by_member.get(member.member_id, []),
            sampling_args=member_sampling_args,
            error=rollout_error,
            reward=member.reward,
            episode_id=episode_id,
            member_id=member.member_id,
        )
        for member in mar.members
    ]

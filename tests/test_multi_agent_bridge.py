"""Bridge contract + MARScore projection guards.

Covers the P0 fixes for the multi-agent serialization boundary:

* ``rollout_to_member_rollouts`` preserves ``output['task']`` (no
  ``env_name`` overwrite) and fails loud on missing canonical fields,
  rather than silently substituting defaults that would corrupt
  identity-keyed downstream baselines (RAE keys on ``(task, example_id,
  member_id)``).
* ``rollout_to_member_rollouts`` rejects trajectories containing
  ``member_id``s that are not part of ``MARScore.members`` — silently
  dropping them was masking framework-invariant violations.
* ``MARScore.to_metrics_flat`` blocks projection keys that collide with
  reserved ``RolloutOutput`` fields (e.g. ``reward``, ``example_id``,
  ``task``); previously such keys were silently rewritten by
  ``state_to_output``'s flattening loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from verifiers.multi_agent_bridge import rollout_to_member_rollouts
from verifiers.rubrics.multi_agent_rubric import MultiAgentRubric
from verifiers.types import MARScore, MemberScore


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _step(member_id: str) -> dict[str, Any]:
    return {
        "extras": {"member_id": member_id},
        "advantage": None,
        "reward": None,
        "tokens": None,
        "is_truncated": False,
        "prompt": [],
        "completion": [],
        "response": None,
        "trajectory_id": "t-1",
    }


def _output(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "example_id": 7,
        "task": "real_task",
        "sampling_args": {"temperature": 0.7, "max_tokens": 128},
        "trajectory_id": "t-1",
        "trajectory": [_step("A")],
        "mar_score": MARScore(
            members=[MemberScore(member_id="A", reward=1.0)],
            episode_scalar=1.0,
        ).model_dump(exclude_none=True),
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------
# Bridge identity preservation
# -----------------------------------------------------------------------


def test_bridge_preserves_output_task() -> None:
    members = rollout_to_member_rollouts(_output(task="task_a"))
    assert all(m["task"] == "task_a" for m in members)


def test_bridge_preserves_full_sampling_args() -> None:
    members = rollout_to_member_rollouts(_output())
    assert members[0]["sampling_args"] == {"temperature": 0.7, "max_tokens": 128}


def test_bridge_defaults_temperature_when_missing_but_keeps_other_args() -> None:
    members = rollout_to_member_rollouts(_output(sampling_args={"max_tokens": 64}))
    assert members[0]["sampling_args"] == {"temperature": 1.0, "max_tokens": 64}


# -----------------------------------------------------------------------
# Bridge fail-loud on missing canonical fields
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_key",
    ["task", "example_id", "sampling_args", "trajectory_id", "mar_score", "trajectory"],
)
def test_bridge_keyerror_on_missing_canonical_field(missing_key: str) -> None:
    out = _output()
    out.pop(missing_key)
    with pytest.raises(KeyError, match=missing_key):
        rollout_to_member_rollouts(out)


def test_bridge_rejects_unknown_member_in_trajectory() -> None:
    out = _output(trajectory=[_step("A"), _step("GHOST")])
    with pytest.raises(ValueError, match="GHOST"):
        rollout_to_member_rollouts(out)


def test_bridge_known_member_with_no_steps_yields_empty_trajectory() -> None:
    """An MARScore member with no trajectory steps is allowed (judge-less
    debater that errored before generating)."""
    out = _output(
        trajectory=[_step("A")],
        mar_score=MARScore(
            members=[
                MemberScore(member_id="A", reward=1.0),
                MemberScore(member_id="B", reward=-1.0),
            ],
            episode_scalar=1.0,
        ).model_dump(exclude_none=True),
    )
    members = rollout_to_member_rollouts(out)
    by_id = {m["member_id"]: m for m in members}
    assert len(by_id["A"]["trajectory"]) == 1
    assert by_id["B"]["trajectory"] == []


def test_bridge_episode_id_from_trajectory_id() -> None:
    members = rollout_to_member_rollouts(_output(trajectory_id="abc-123"))
    assert all(m["episode_id"] == "abc-123" for m in members)


# -----------------------------------------------------------------------
# MARScore reserved-key projection guard
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "reserved_key", ["reward", "example_id", "task", "metrics", "trajectory"]
)
def test_marscore_episode_metrics_reserved_key_rejected(reserved_key: str) -> None:
    mar = MARScore(
        members=[MemberScore(member_id="A", reward=0.0)],
        episode_scalar=0.0,
        episode_metrics={reserved_key: 1.0},
    )
    with pytest.raises(ValueError, match=reserved_key):
        mar.to_metrics_flat()


@pytest.mark.parametrize("reserved_key", ["reward", "example_id", "task", "metrics"])
def test_marscore_member_metrics_reserved_key_rejected(reserved_key: str) -> None:
    mar = MARScore(
        members=[MemberScore(member_id="A", reward=1.0, metrics={reserved_key: 0.5})],
        episode_scalar=1.0,
    )
    with pytest.raises(ValueError, match=reserved_key):
        mar.to_metrics_flat()


def test_marscore_non_reserved_keys_project_normally() -> None:
    mar = MARScore(
        members=[
            MemberScore(member_id="A", reward=1.0, metrics={"accuracy": 0.7}),
            MemberScore(member_id="B", reward=-1.0, metrics={"accuracy": 0.3}),
        ],
        episode_scalar=1.0,
        episode_metrics={"agreement": 0.5},
    )
    flat = mar.to_metrics_flat()
    assert flat == {
        "agreement": 0.5,
        "reward/A": 1.0,
        "reward/B": -1.0,
        "accuracy/A": 0.7,
        "accuracy/B": 0.3,
    }


# -----------------------------------------------------------------------
# Orchestrator MA-incompat guard
# -----------------------------------------------------------------------


class _FakeMARubric(MultiAgentRubric):
    members = ["A"]

    async def build_marscore(self, state):  # pragma: no cover - never called
        return MARScore(
            members=[MemberScore(member_id="A", reward=0.0)],
            episode_scalar=0.0,
        )


class _FakeEnv:
    """Duck-typed Environment that only exposes the attributes the
    Orchestrator MA guard touches before any heavy init."""

    def __init__(self, rubric):
        self.rubric = rubric


def _orchestrator_kwargs(env) -> dict[str, Any]:
    # Sentinel values are fine — the MA guard fires before any of these
    # are touched. The dataset filter at the bottom of __init__ would
    # crash on a fake env, which is the assertion: we never reach it.
    return dict(
        env=env,
        client_base_url="http://x",
        client_api_key="k",
        client_limit=1,
        client_timeout=1.0,
        model_name="m",
        sampling_args={},
        rollouts_per_example=1,
        batch_size=1,
        micro_batch_size=1,
        num_processes=1,
        generation_timeout=1.0,
        processing_class=object(),
        mask_env_responses=False,
        max_seq_len=1,
        max_prompt_len=1,
        mask_truncated_completions=False,
        zero_truncated_completions=False,
        max_concurrent=1,
    )


def test_orchestrator_rejects_multi_agent_rubric() -> None:
    pytest.importorskip("verifiers_rl")
    from verifiers_rl.rl.trainer.orchestrator import Orchestrator  # type: ignore[unresolved-import]

    env = _FakeEnv(rubric=_FakeMARubric())
    with pytest.raises(NotImplementedError, match="MultiAgentRubric"):
        Orchestrator(**_orchestrator_kwargs(env))


def test_orchestrator_rejects_rubric_group_containing_multi_agent_rubric() -> None:
    pytest.importorskip("verifiers_rl")
    from verifiers_rl.rl.trainer.orchestrator import Orchestrator  # type: ignore[unresolved-import]
    from verifiers.rubrics.rubric_group import RubricGroup

    env = _FakeEnv(rubric=RubricGroup(rubrics=[_FakeMARubric()]))
    with pytest.raises(NotImplementedError, match="MultiAgentRubric"):
        Orchestrator(**_orchestrator_kwargs(env))


# Sanity: the imports above are not the only possible test paths. Pull
# in asyncio + ValidationError so static checkers don't strip the
# imports if a future refactor decides to use them.
_ = (asyncio, ValidationError)

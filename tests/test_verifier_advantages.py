from unittest.mock import MagicMock

import pytest

from verifiers import Rubric
from verifiers.types import (
    AssistantMessage,
    RolloutInput,
    State,
    TrajectoryStep,
    TrajectoryStepTokens,
    UserMessage,
)
from verifiers.utils.save_utils import states_to_outputs


def _make_step(
    *,
    completion_ids: list[int],
    advantage: float | None,
) -> TrajectoryStep:
    completion_length = len(completion_ids)
    return TrajectoryStep(
        prompt=[UserMessage(content="q")],
        completion=[AssistantMessage(content="a")],
        response=MagicMock(),
        tokens=TrajectoryStepTokens(
            prompt_ids=[1],
            prompt_mask=[0],
            completion_ids=completion_ids,
            completion_mask=[1] * completion_length,
            completion_logprobs=[-0.1] * completion_length,
            overlong_prompt=False,
            is_truncated=False,
            routed_experts=None,
        ),
        reward=None,
        advantage=advantage,
        is_truncated=False,
        trajectory_id="trajectory-1",
        extras={},
    )


@pytest.mark.asyncio
async def test_score_group_populates_step_completion_advantages_from_state_advantage():
    def reward_by_length(completion, **kwargs):
        return float(len(str(completion)))

    rubric = Rubric(funcs=[reward_by_length], weights=[1.0])

    state_a = State(
        input=RolloutInput(
            prompt=[UserMessage(content="prompt-a")],
            answer="",
            task="task",
            example_id=0,
        )
    )
    state_a["completion"] = "a"
    state_a["trajectory"] = [_make_step(completion_ids=[11, 12], advantage=None)]
    state_a["timing"] = {
        "generation_ms": 0.0,
        "scoring_ms": 0.0,
        "total_ms": 0.0,
        "start_time": 0.0,
    }

    state_b = State(
        input=RolloutInput(
            prompt=[UserMessage(content="prompt-b")],
            answer="",
            task="task",
            example_id=1,
        )
    )
    state_b["completion"] = "bbb"
    state_b["trajectory"] = [_make_step(completion_ids=[21, 22, 23], advantage=None)]
    state_b["timing"] = {
        "generation_ms": 0.0,
        "scoring_ms": 0.0,
        "total_ms": 0.0,
        "start_time": 0.0,
    }

    await rubric.score_group([state_a, state_b])

    step_a = state_a["trajectory"][0]
    step_b = state_b["trajectory"][0]

    assert state_a["advantage"] == -1.0
    assert state_b["advantage"] == 1.0
    assert step_a["advantage"] == state_a["advantage"]
    assert step_b["advantage"] == state_b["advantage"]
    assert step_a["completion_advantages"] == [-1.0, -1.0]
    assert step_b["completion_advantages"] == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_score_group_preserves_existing_step_advantage_for_completion_advantages():
    def constant_reward(completion, **kwargs):
        return 1.0

    rubric = Rubric(funcs=[constant_reward], weights=[1.0])

    state = State(
        input=RolloutInput(
            prompt=[UserMessage(content="prompt")],
            answer="",
            task="task",
            example_id=0,
        )
    )
    state["completion"] = "answer"
    state["trajectory"] = [_make_step(completion_ids=[31, 32], advantage=0.25)]
    state["timing"] = {
        "generation_ms": 0.0,
        "scoring_ms": 0.0,
        "total_ms": 0.0,
        "start_time": 0.0,
    }

    await rubric.score_group([state])

    step = state["trajectory"][0]
    assert state["advantage"] == 0.0
    assert step["advantage"] == 0.25
    assert step["completion_advantages"] == [0.25, 0.25]


def test_states_to_outputs_includes_use_verifiers_advantages(make_state):
    state = make_state()
    state["use_verifiers_advantages"] = True

    output = states_to_outputs([state], state_columns=["use_verifiers_advantages"])[0]

    assert output["use_verifiers_advantages"] is True

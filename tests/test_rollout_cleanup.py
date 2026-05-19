import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.envs.experimental.composable.tasksets.swe.r2e_gym import R2ERubric
from verifiers.types import RolloutInput, SamplingArgs, State


def _dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "question": ["q0", "q1"],
            "answer": ["a0", "a1"],
        }
    )


def _input(example_id: int) -> RolloutInput:
    return {
        "prompt": [{"role": "user", "content": f"q{example_id}"}],
        "answer": f"a{example_id}",
        "example_id": example_id,
    }


class RecordingRubric(vf.Rubric):
    def __init__(
        self,
        *,
        score_rollout_error: Exception | None = None,
        score_group_error: Exception | None = None,
    ):
        super().__init__()
        self.cleaned: list[int] = []
        self.score_rollout_error = score_rollout_error
        self.score_group_error = score_group_error

    async def score_rollout(self, state: State):
        if self.score_rollout_error is not None:
            raise self.score_rollout_error
        state["reward"] = 1.0
        state["metrics"] = {}

    async def score_group(self, states: list[State]):
        if self.score_group_error is not None:
            raise self.score_group_error
        for state in states:
            state["reward"] = 1.0
            state["metrics"] = {}

    async def cleanup(self, state: State):
        self.cleaned.append(state["example_id"])


class StaticRolloutEnv(vf.Environment):
    async def rollout(
        self,
        input: RolloutInput,
        client: vf.Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        state["sandbox_id"] = f"sb-{state['example_id']}"
        return state


def _env(rubric: vf.Rubric) -> StaticRolloutEnv:
    return StaticRolloutEnv(dataset=_dataset(), parser=vf.Parser(), rubric=rubric)


@pytest.mark.asyncio
async def test_run_rollout_state_cleans_up_when_scoring_raises(mock_client):
    rubric = RecordingRubric(score_rollout_error=RuntimeError("score failed"))
    env = _env(rubric)

    with pytest.raises(RuntimeError, match="score failed"):
        await env._run_rollout_state(_input(0), mock_client, "test-model", {})

    assert rubric.cleaned == [0]


@pytest.mark.asyncio
async def test_run_group_states_cleans_completed_states_when_gather_raises(
    mock_client,
):
    first_rollout_finished = asyncio.Event()

    class PartiallyFailingEnv(StaticRolloutEnv):
        async def rollout(
            self,
            input: RolloutInput,
            client: vf.Client,
            model: str,
            sampling_args: SamplingArgs | None = None,
        ) -> State:
            if input["example_id"] == 1:
                await first_rollout_finished.wait()
                raise RuntimeError("rollout failed")
            state = await super().rollout(input, client, model, sampling_args)
            first_rollout_finished.set()
            return state

    rubric = RecordingRubric()
    env = PartiallyFailingEnv(dataset=_dataset(), parser=vf.Parser(), rubric=rubric)

    with pytest.raises(RuntimeError, match="rollout failed"):
        await env._run_group_states(
            [_input(0), _input(1)],
            mock_client,
            "test-model",
            {},
        )

    assert rubric.cleaned == [0]


@pytest.mark.asyncio
async def test_run_group_states_cleans_all_states_when_group_scoring_raises(
    mock_client,
):
    rubric = RecordingRubric(score_group_error=RuntimeError("group score failed"))
    env = _env(rubric)

    with pytest.raises(RuntimeError, match="group score failed"):
        await env._run_group_states(
            [_input(0), _input(1)],
            mock_client,
            "test-model",
            {},
        )

    assert rubric.cleaned == [0, 1]


@pytest.mark.asyncio
async def test_environment_cleanup_failure_does_not_skip_later_handler(mock_client):
    class FailingCleanupEnv(StaticRolloutEnv):
        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.destroyed = False

        @vf.cleanup(priority=1)
        async def failing_cleanup(self, state: State):
            raise RuntimeError("early cleanup failed")

        @vf.cleanup(priority=0)
        async def destroy_sandbox(self, state: State):
            self.destroyed = True

    env = FailingCleanupEnv(dataset=_dataset(), parser=vf.Parser(), rubric=vf.Rubric())
    state = await env.init_state(_input(0), mock_client, "test-model")

    with pytest.raises(RuntimeError, match="early cleanup failed"):
        await env.cleanup(state)

    assert env.destroyed is True


@pytest.mark.asyncio
async def test_rubric_group_cleanup_failure_does_not_skip_later_rubric():
    class FailingRubric(vf.Rubric):
        async def cleanup(self, state: State):
            raise RuntimeError("rubric cleanup failed")

    class DestroyingRubric(vf.Rubric):
        async def cleanup(self, state: State):
            state["destroyed"] = True

    state: State = vf.State(input={})
    rubric = vf.RubricGroup([FailingRubric(), DestroyingRubric()])

    with pytest.raises(RuntimeError, match="rubric cleanup failed"):
        await rubric.cleanup(state)

    assert state["destroyed"] is True


@pytest.mark.asyncio
async def test_cli_agent_destroy_sandbox_deletes_when_post_rollout_fails():
    class FailingPostRolloutEnv(CliAgentEnv):
        async def post_rollout(self, state: State):
            raise RuntimeError("post rollout failed")

    env = FailingPostRolloutEnv(
        run_command="echo done",
        dataset=_dataset(),
        parser=vf.Parser(),
        rubric=vf.Rubric(),
        keep_sandbox_for_scoring=True,
    )
    env.delete_sandbox = AsyncMock()  # type: ignore[method-assign]
    state: State = vf.State(input={})
    state.update({"is_completed": True, "sandbox_id": "sb-post-rollout-failed"})

    try:
        with pytest.raises(RuntimeError, match="post rollout failed"):
            await env.destroy_sandbox(state)
        env.delete_sandbox.assert_awaited_once_with("sb-post-rollout-failed")
    finally:
        env.teardown_sandbox_client()


@pytest.mark.asyncio
async def test_swe_rubric_model_error_skips_sandbox_scoring():
    class StubTaskSet:
        def __init__(self):
            self.ran_tests = False

        async def _run_tests(self, *args: Any, **kwargs: Any) -> str:
            self.ran_tests = True
            return "PASS"

        def _calculate_reward(self, test_output: str, info: dict[str, Any]) -> float:
            return 1.0

    taskset = StubTaskSet()
    rubric = R2ERubric(taskset)  # type: ignore[arg-type]
    state: State = vf.State(input={})
    state.update(
        {
            "error": vf.ModelError("No available workers"),
            "sandbox_client": object(),
            "sandbox_id": "sb-leaked-without-short-circuit",
        }
    )

    reward = await rubric.solved(state, info={})

    assert reward == 0.0
    assert taskset.ran_tests is False

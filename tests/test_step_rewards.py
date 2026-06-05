import pytest

from verifiers import MultiTurnEnv, Parser, Rubric, State, step_reward
from verifiers.rubrics.step_reward_rubric import StepRewardRubric
from verifiers.utils.step_reward_utils import (
    apply_step_advantages,
    compute_discounted_returns,
)


class TestComputeDiscountedReturns:
    def test_empty(self):
        assert compute_discounted_returns([]) == []

    def test_single_step(self):
        assert compute_discounted_returns([5.0]) == [5.0]

    def test_no_discounting(self):
        result = compute_discounted_returns([1.0, 2.0, 3.0], gamma=1.0)
        assert result == [6.0, 5.0, 3.0]

    def test_full_discounting(self):
        result = compute_discounted_returns([1.0, 2.0, 3.0], gamma=0.0)
        assert result == [1.0, 2.0, 3.0]

    def test_partial_discounting(self):
        result = compute_discounted_returns([1.0, 1.0, 1.0], gamma=0.5)
        # R_2 = 1.0
        # R_1 = 1.0 + 0.5*1.0 = 1.5
        # R_0 = 1.0 + 0.5*1.5 = 1.75
        assert result[2] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.5)
        assert result[0] == pytest.approx(1.75)


class TestApplyStepAdvantages:
    def _make_state(self, step_rewards):
        trajectory = [
            {"reward": r, "advantage": None, "prompt": [], "completion": []}
            for r in step_rewards
        ]
        return State(trajectory=trajectory, reward=None, advantage=None)

    def test_single_state_uniform_rewards(self):
        states = [self._make_state([1.0, 1.0, 1.0])]
        apply_step_advantages(states, gamma=1.0)
        # All returns equal (3.0, 2.0, 1.0) -> normalized advantages
        for step in states[0]["trajectory"]:
            assert step["advantage"] is not None

    def test_two_states_advantage_normalization(self):
        states = [
            self._make_state([1.0, 0.0]),
            self._make_state([0.0, 1.0]),
        ]
        apply_step_advantages(states, gamma=1.0)
        # All steps should have advantages that sum to ~0
        all_advantages = []
        for state in states:
            for step in state["trajectory"]:
                all_advantages.append(step["advantage"])
        assert sum(all_advantages) == pytest.approx(0.0, abs=1e-6)

    def test_sets_rollout_level_advantage(self):
        states = [
            self._make_state([2.0, 1.0]),
            self._make_state([0.0, 0.0]),
        ]
        apply_step_advantages(states, gamma=1.0)
        assert states[0]["advantage"] is not None
        assert states[1]["advantage"] is not None
        # State with higher rewards should have higher advantage
        assert states[0]["advantage"] > states[1]["advantage"]

    def test_empty_trajectory(self):
        states = [State(trajectory=[], reward=None, advantage=None)]
        apply_step_advantages(states, gamma=1.0)
        assert states[0].get("advantage") is None

    def test_gamma_affects_returns(self):
        states = [self._make_state([0.0, 0.0, 10.0])]
        apply_step_advantages(states, gamma=0.5)
        traj = states[0]["trajectory"]
        # With gamma=0.5, earlier steps get less credit for future reward
        assert traj[0]["reward"] < traj[1]["reward"] < traj[2]["reward"]


class TestStepRewardDecorator:
    def test_bare_decorator(self):
        @step_reward
        def my_func(state):
            return 1.0

        assert getattr(my_func, "step_reward") is True
        assert getattr(my_func, "step_reward_priority") == 0
        assert getattr(my_func, "step_reward_weight") == 1.0

    def test_parameterized_decorator(self):
        @step_reward(weight=0.5, priority=10)
        def my_func(state):
            return 1.0

        assert getattr(my_func, "step_reward") is True
        assert getattr(my_func, "step_reward_priority") == 10
        assert getattr(my_func, "step_reward_weight") == 0.5


class TestStepRewardInRollout:
    @pytest.mark.asyncio
    async def test_step_reward_called_per_turn(
        self, mock_client, sample_chat_dataset, make_input
    ):
        class StepRewardEnv(MultiTurnEnv):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.step_reward_calls = 0

            @step_reward
            async def count_reward(self, state: State) -> float:
                self.step_reward_calls += 1
                return 1.0

            async def env_response(self, messages, state, **kwargs):
                return [{"role": "user", "content": "Continue"}]

        mock_client.set_default_response("response")
        env = StepRewardEnv(
            client=mock_client,
            model="test-model",
            dataset=sample_chat_dataset,
            max_turns=3,
            parser=Parser(),
            rubric=Rubric(),
        )

        state = await env.rollout(
            input=make_input(
                prompt=[{"role": "user", "content": "Start"}],
                answer="answer",
            ),
            client=mock_client,
            model="test-model",
        )

        assert env.step_reward_calls == 3
        for step in state["trajectory"]:
            assert step["reward"] == 1.0

    @pytest.mark.asyncio
    async def test_weighted_step_rewards(
        self, mock_client, sample_chat_dataset, make_input
    ):
        class WeightedEnv(MultiTurnEnv):
            @step_reward(weight=2.0)
            async def double_reward(self, state: State) -> float:
                return 1.0

            async def env_response(self, messages, state, **kwargs):
                return [{"role": "user", "content": "Continue"}]

        mock_client.set_default_response("response")
        env = WeightedEnv(
            client=mock_client,
            model="test-model",
            dataset=sample_chat_dataset,
            max_turns=2,
            parser=Parser(),
            rubric=Rubric(),
        )

        state = await env.rollout(
            input=make_input(
                prompt=[{"role": "user", "content": "Start"}],
                answer="answer",
            ),
            client=mock_client,
            model="test-model",
        )

        for step in state["trajectory"]:
            assert step["reward"] == 2.0

    @pytest.mark.asyncio
    async def test_multiple_step_rewards(
        self, mock_client, sample_chat_dataset, make_input
    ):
        class MultiRewardEnv(MultiTurnEnv):
            @step_reward(priority=10)
            async def reward_a(self, state: State) -> float:
                return 1.0

            @step_reward(weight=0.5)
            async def reward_b(self, state: State) -> float:
                return 2.0

            async def env_response(self, messages, state, **kwargs):
                return [{"role": "user", "content": "Continue"}]

        mock_client.set_default_response("response")
        env = MultiRewardEnv(
            client=mock_client,
            model="test-model",
            dataset=sample_chat_dataset,
            max_turns=1,
            parser=Parser(),
            rubric=Rubric(),
        )

        state = await env.rollout(
            input=make_input(
                prompt=[{"role": "user", "content": "Start"}],
                answer="answer",
            ),
            client=mock_client,
            model="test-model",
        )

        # 1.0 * 1.0 + 2.0 * 0.5 = 2.0
        assert state["trajectory"][0]["reward"] == 2.0

    @pytest.mark.asyncio
    async def test_no_step_reward_handlers(
        self, mock_client, sample_chat_dataset, make_input
    ):
        class PlainEnv(MultiTurnEnv):
            async def env_response(self, messages, state, **kwargs):
                return [{"role": "user", "content": "Continue"}]

        mock_client.set_default_response("response")
        env = PlainEnv(
            client=mock_client,
            model="test-model",
            dataset=sample_chat_dataset,
            max_turns=2,
            parser=Parser(),
            rubric=Rubric(),
        )

        state = await env.rollout(
            input=make_input(
                prompt=[{"role": "user", "content": "Start"}],
                answer="answer",
            ),
            client=mock_client,
            model="test-model",
        )

        for step in state["trajectory"]:
            assert step["reward"] is None


class TestStepRewardRubric:
    @pytest.mark.asyncio
    async def test_score_group(self):
        rubric = StepRewardRubric(gamma=1.0)
        states = [
            State(
                trajectory=[
                    {"reward": 1.0, "advantage": None},
                    {"reward": 2.0, "advantage": None},
                ],
                reward=None,
                advantage=None,
            ),
            State(
                trajectory=[
                    {"reward": 0.0, "advantage": None},
                    {"reward": 0.0, "advantage": None},
                ],
                reward=None,
                advantage=None,
            ),
        ]

        await rubric.score_group(states)

        assert states[0]["reward"] == 3.0
        assert states[1]["reward"] == 0.0
        assert states[0]["advantage"] > states[1]["advantage"]
        for state in states:
            for step in state["trajectory"]:
                assert step["advantage"] is not None

    @pytest.mark.asyncio
    async def test_score_group_with_discount(self):
        rubric = StepRewardRubric(gamma=0.5)
        states = [
            State(
                trajectory=[
                    {"reward": 0.0, "advantage": None},
                    {"reward": 10.0, "advantage": None},
                ],
                reward=None,
                advantage=None,
            ),
        ]

        await rubric.score_group(states)

        traj = states[0]["trajectory"]
        # step 0 return: 0.0 + 0.5*10.0 = 5.0
        # step 1 return: 10.0
        assert traj[0]["reward"] == pytest.approx(5.0)
        assert traj[1]["reward"] == pytest.approx(10.0)

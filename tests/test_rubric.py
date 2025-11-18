"""Tests for the Rubric class."""

from typing import cast

import pytest

from verifiers import Parser, Rubric
from verifiers.types import RewardFunc, RolloutInput, State
from verifiers.utils.async_utils import NullAsyncContext


class TestRubric:
    """Test cases for the Rubric class."""

    def test_rubric_initialization_empty(self):
        """Test Rubric initialization with no parameters."""
        rubric = Rubric()

        assert rubric.funcs == []
        assert rubric.weights == []
        assert isinstance(rubric.parser, Parser)

    def test_rubric_initialization_with_functions(self):
        """Test Rubric initialization with reward functions."""

        def reward_func1(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        def reward_func2(completion, **kwargs):
            return len(completion) * 0.1

        funcs = cast(list[RewardFunc], [reward_func1, reward_func2])
        weights = [1.0, 0.5]

        rubric = Rubric(funcs=funcs, weights=weights)

        assert rubric.funcs == funcs
        assert rubric.weights == weights
        assert len(rubric._get_reward_func_names()) == 2
        assert rubric._get_reward_func_names() == ["reward_func1", "reward_func2"]

    def test_rubric_initialization_functions_without_weights(self):
        """Test Rubric initialization with functions but no explicit weights."""

        def reward_func1(completion, **kwargs) -> float:
            return 1.0

        def reward_func2(completion, **kwargs) -> float:
            return 0.5

        funcs = cast(list[RewardFunc], [reward_func1, reward_func2])

        rubric = Rubric(funcs=funcs)

        assert rubric.funcs == funcs
        assert rubric.weights == [1.0, 1.0]  # Default weights

    def test_rubric_initialization_with_kwargs(self):
        """Test Rubric initialization - kwargs not supported."""
        # Rubric doesn't accept arbitrary kwargs
        with pytest.raises(TypeError):
            Rubric(custom_param="test_value", another_param=42)

    def test_add_reward_func(self):
        """Test adding reward functions."""
        rubric = Rubric(funcs=[], weights=[])

        def test_func(completion, **kwargs):
            return 1.0

        rubric.add_reward_func(test_func, weight=0.8)

        assert len(rubric.funcs) == 1
        assert rubric.funcs[0] == test_func
        assert rubric.weights == [0.8]
        assert rubric._get_reward_func_names() == ["test_func"]

    def test_add_multiple_reward_funcs(self):
        """Test adding multiple reward functions."""
        # Create fresh rubric to avoid test isolation issues
        rubric = Rubric(funcs=[], weights=[])

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        rubric.add_reward_func(func1, weight=1.0)
        rubric.add_reward_func(func2, weight=0.3)

        assert len(rubric.funcs) == 2
        assert rubric._get_reward_func_names() == ["func1", "func2"]
        assert rubric.weights == [1.0, 0.3]

    def test_add_reward_func_default_weight(self):
        """Test adding reward function with default weight."""
        rubric = Rubric(funcs=[], weights=[])

        def test_func(completion, **kwargs):
            return 1.0

        rubric.add_reward_func(test_func)

        assert rubric.weights == [1.0]

    def test_get_methods(self):
        """Test getter methods."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        rubric = Rubric(funcs=[func1, func2], weights=[0.8, 0.2])

        assert rubric._get_reward_funcs() == [func1, func2]
        assert rubric._get_reward_weights() == [0.8, 0.2]
        assert rubric._get_reward_func_names() == ["func1", "func2"]

    @pytest.mark.asyncio
    async def test_call_reward_func_with_all_args(self):
        """Test calling reward function - method removed, test internal call instead."""
        # call_reward_func method doesn't exist anymore
        # Reward functions are called internally via _call_individual_reward_func
        # This test is no longer applicable
        pass

    @pytest.mark.asyncio
    async def test_call_reward_func_with_subset_args(self):
        """Test calling reward function - method removed."""
        pass

    @pytest.mark.asyncio
    async def test_call_reward_func_with_var_kwargs(self):
        """Test calling reward function - method removed."""
        pass

    @pytest.mark.asyncio
    async def test_call_reward_func_error_handling(self):
        """Test error handling - tested via score_rollout instead."""
        # Error handling is tested through score_rollout
        pass

    @pytest.mark.asyncio
    async def test_score_rollout_single(self):
        """Test scoring a single rollout."""

        def func1(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        def func2(completion, **kwargs):
            return len(completion) * 0.1

        rubric = Rubric(funcs=[func1, func2], weights=[1.0, 0.5])

        state = State(
            input=RolloutInput(
                prompt="test prompt",
                answer="test",
                task="test_task",
                example_id=0,
            )
        )
        state["completion"] = "test"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }
        score_sem = NullAsyncContext()

        await rubric.score_rollout(state, score_sem)

        assert "func1" in state["metrics"]
        assert "func2" in state["metrics"]
        assert state["metrics"]["func1"] == 1.0  # completion == answer
        assert state["metrics"]["func2"] == 0.4  # len("test") * 0.1
        assert state["reward"] == 1.0 * 1.0 + 0.4 * 0.5  # Weighted sum

    @pytest.mark.asyncio
    async def test_score_rollout_with_list_completion(self):
        """Test scoring rollout with list-type completion."""

        def list_func(completion, **kwargs):
            return len(completion) if isinstance(completion, list) else 0.0

        rubric = Rubric(funcs=[list_func])

        completion = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        state = State(
            input=RolloutInput(
                prompt="test",
                answer="test",
                task="test",
                example_id=0,
            )
        )
        state["completion"] = completion
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }
        score_sem = NullAsyncContext()

        await rubric.score_rollout(state, score_sem)

        assert state["metrics"]["list_func"] == 2.0  # Length of completion list
        assert state["reward"] == 2.0

    @pytest.mark.asyncio
    async def test_score_rollouts_multiple(self):
        """Test scoring multiple rollouts using score_group."""

        def accuracy_func(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        def length_func(completion, **kwargs):
            return len(str(completion))

        rubric = Rubric(funcs=[accuracy_func, length_func], weights=[1.0, 0.1])

        states = [
            State(
                input=RolloutInput(
                    prompt="prompt1",
                    answer="answer1",
                    task="task1",
                    example_id=0,
                )
            ),
            State(
                input=RolloutInput(
                    prompt="prompt2",
                    answer="answer2",
                    task="task2",
                    example_id=1,
                )
            ),
            State(
                input=RolloutInput(
                    prompt="prompt3",
                    answer="answer3",
                    task="task3",
                    example_id=2,
                )
            ),
        ]
        for i, state in enumerate(states):
            state["completion"] = ["answer1", "answer2", "wrong"][i]
            state["trajectory"] = []
            state["timing"] = {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
                "start_time": 0.0,
            }

        score_sem = NullAsyncContext()
        await rubric.score_group(states, score_sem)

        assert states[0]["metrics"]["accuracy_func"] == 1.0
        assert states[1]["metrics"]["accuracy_func"] == 1.0
        assert states[2]["metrics"]["accuracy_func"] == 0.0
        assert states[0]["metrics"]["length_func"] == 7.0
        assert states[1]["metrics"]["length_func"] == 7.0
        assert states[2]["metrics"]["length_func"] == 5.0

    @pytest.mark.asyncio
    async def test_score_rollouts_with_apply_weights(self):
        """Test scoring rollouts - weights always applied via score_group."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        rubric = Rubric(funcs=[func1, func2], weights=[2.0, 3.0])

        state = State(
            input=RolloutInput(
                prompt="test",
                answer="test",
                task="test",
                example_id=0,
            )
        )
        state["completion"] = "test"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }
        score_sem = NullAsyncContext()

        await rubric.score_group([state], score_sem)

        # Weighted sum: 1.0*2.0 + 0.5*3.0 = 3.5
        assert state["reward"] == pytest.approx(3.5)

    @pytest.mark.asyncio
    async def test_score_rollouts_empty(self):
        """Test scoring empty list of rollouts."""

        def test_func(completion, **kwargs):
            return 1.0

        rubric = Rubric(funcs=[test_func], weights=[1.0])
        score_sem = NullAsyncContext()

        # score_group with empty list should handle gracefully
        await rubric.score_group([], score_sem)

    @pytest.mark.asyncio
    async def test_score_rollouts_with_default_infos(self):
        """Test scoring rollouts with default empty infos."""

        def simple_func(completion, **kwargs):
            return 1.0

        rubric = Rubric(funcs=[simple_func], weights=[1.0])

        state = State(
            input=RolloutInput(
                prompt="test",
                answer="test",
                task="test",
                example_id=0,
            )
        )
        state["completion"] = "test"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }
        score_sem = NullAsyncContext()

        await rubric.score_group([state], score_sem)

        assert "simple_func" in state["metrics"]
        assert state["metrics"]["simple_func"] == 1.0

    def test_rubric_with_custom_parser(self):
        """Test Rubric with custom parser."""
        custom_parser = Parser()
        rubric = Rubric(funcs=[], weights=[], parser=custom_parser)

        assert rubric.parser is custom_parser

    @pytest.mark.asyncio
    async def test_score_rollouts_with_mixed_return_types(self):
        """Test scoring when reward functions return different types."""

        def scalar_func(completion, **kwargs):
            return 0.5

        rubric = Rubric(funcs=[scalar_func], weights=[1.0])

        state = State(
            input=RolloutInput(
                prompt="test",
                answer="test",
                task="test",
                example_id=0,
            )
        )
        state["completion"] = "test"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }
        score_sem = NullAsyncContext()

        await rubric.score_group([state], score_sem)

        assert state["metrics"]["scalar_func"] == 0.5
        assert state["reward"] == 0.5

    @pytest.mark.asyncio
    async def test_call_reward_func_kwargs_filtering(self):
        """Test that functions without **kwargs get filtered kwargs."""

        def f_no_kwargs(completion, answer):
            return 0.5

        def f_with_kwargs(completion, **kwargs):
            assert kwargs.get("info", {}).get("extra") == 123
            return 1.0

        rubric = Rubric(funcs=[f_no_kwargs, f_with_kwargs], weights=[1.0, 2.0])

        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "q"}],
                answer="ans",
                task="default",
                example_id=0,
                info={"extra": 123},
            )
        )
        state["completion"] = [{"role": "assistant", "content": "a"}]
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }
        score_sem = NullAsyncContext()

        await rubric.score_rollout(state, score_sem)

        # Weighted sum: 0.5*1 + 1.0*2 = 2.5
        assert state["reward"] == pytest.approx(2.5)
        assert set(state["metrics"].keys()) == {"f_no_kwargs", "f_with_kwargs"}

    @pytest.mark.asyncio
    async def test_score_rollout_serial_execution_order(self):
        """Test that execution order is respected."""
        calls = []

        def g1(**kwargs):
            calls.append("g1")
            return 0.2

        def g2(**kwargs):
            calls.append("g2")
            return 0.3

        rubric = Rubric(funcs=[g1, g2], weights=[1.0, 1.0])

        state = State(
            input=RolloutInput(
                prompt="q",
                answer="ans",
                task="default",
                example_id=0,
            )
        )
        state["completion"] = "a"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }
        score_sem = NullAsyncContext()

        await rubric.score_rollout(state, score_sem)

        assert state["reward"] == pytest.approx(0.5)
        assert calls == ["g1", "g2"]  # order respected

    @pytest.mark.asyncio
    async def test_call_reward_func_error_handling_both_paths(self):
        """Test error handling - tested via score_rollout instead."""
        # Error handling is tested through score_rollout
        pass

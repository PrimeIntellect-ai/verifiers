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


class TestGDPO:
    """Test cases for GDPO (Group Reward-Decoupled Policy Optimization) mode."""

    def test_gdpo_initialization(self):
        """Test Rubric initialization with GDPO parameters."""
        gates = {"style_score": {"func": "correct_answer", "op": ">=", "value": 1.0}}
        rubric = Rubric(
            advantage_mode="gdpo",
            gates=gates,
            epsilon=1e-6,
        )

        assert rubric.advantage_mode == "gdpo"
        assert rubric.gates == gates
        assert rubric.epsilon == 1e-6

    def test_default_advantage_mode_is_grpo(self):
        """Test that default advantage_mode is 'grpo'."""
        rubric = Rubric()
        assert rubric.advantage_mode == "grpo"
        assert rubric.gates == {}
        assert rubric.epsilon == 1e-8

    def test_zscore_normalize_group_normal(self):
        """Test group z-score normalization with normal values (no epsilon)."""
        rubric = Rubric(epsilon=1e-8)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = rubric._zscore_normalize_group(values)

        # Mean = 3.0, Std ≈ 1.414
        assert len(result) == 5
        assert result[2] == pytest.approx(0.0, abs=1e-6)  # Middle value
        assert result[0] < 0  # Below mean
        assert result[4] > 0  # Above mean

    def test_zscore_normalize_group_zero_std(self):
        """Test group z-score normalization when σ=0 (returns zeros, no epsilon)."""
        rubric = Rubric(epsilon=1e-8)
        values = [5.0, 5.0, 5.0, 5.0]
        result = rubric._zscore_normalize_group(values)

        # When std=0, per paper Eq. 4: return zeros (no signal)
        for r in result:
            assert r == pytest.approx(0.0, abs=1e-6)

    def test_zscore_normalize_batch_with_epsilon(self):
        """Test batch z-score normalization uses epsilon in denominator."""
        rubric = Rubric(epsilon=0.1)  # Large epsilon for testing
        values = [1.0, 1.0, 1.0]  # All same -> std=0
        result = rubric._zscore_normalize_batch(values)

        # With epsilon: (0) / (0 + 0.1) = 0
        for r in result:
            assert r == pytest.approx(0.0, abs=1e-6)

    def test_zscore_normalize_empty(self):
        """Test z-score normalization with empty list."""
        rubric = Rubric(epsilon=1e-8)
        assert rubric._zscore_normalize_group([]) == []
        assert rubric._zscore_normalize_batch([]) == []

    def test_evaluate_gate_simple_condition(self):
        """Test gate evaluation with simple condition."""
        rubric = Rubric()
        metrics = {"correct_answer": 1.0, "style_score": 0.8}

        # Test >= operator
        gate = {"func": "correct_answer", "op": ">=", "value": 1.0}
        assert rubric._evaluate_gate(gate, metrics) is True

        gate = {"func": "correct_answer", "op": ">=", "value": 1.5}
        assert rubric._evaluate_gate(gate, metrics) is False

        # Test == operator
        gate = {"func": "correct_answer", "op": "==", "value": 1.0}
        assert rubric._evaluate_gate(gate, metrics) is True

        # Test < operator
        gate = {"func": "style_score", "op": "<", "value": 0.9}
        assert rubric._evaluate_gate(gate, metrics) is True

    def test_evaluate_gate_and_expression(self):
        """Test gate evaluation with AND expression."""
        rubric = Rubric()
        metrics = {"correct_answer": 1.0, "style_score": 0.8, "length": 0.5}

        gate = {
            "AND": [
                {"func": "correct_answer", "op": "==", "value": 1.0},
                {"func": "style_score", "op": ">=", "value": 0.5},
            ]
        }
        assert rubric._evaluate_gate(gate, metrics) is True

        # One condition fails
        gate = {
            "AND": [
                {"func": "correct_answer", "op": "==", "value": 1.0},
                {"func": "style_score", "op": ">=", "value": 0.9},  # Fails
            ]
        }
        assert rubric._evaluate_gate(gate, metrics) is False

    def test_evaluate_gate_or_expression(self):
        """Test gate evaluation with OR expression."""
        rubric = Rubric()
        metrics = {"correct_answer": 0.0, "style_score": 0.8}

        gate = {
            "OR": [
                {"func": "correct_answer", "op": "==", "value": 1.0},  # Fails
                {"func": "style_score", "op": ">=", "value": 0.5},  # Passes
            ]
        }
        assert rubric._evaluate_gate(gate, metrics) is True

        # Both fail
        gate = {
            "OR": [
                {"func": "correct_answer", "op": "==", "value": 1.0},
                {"func": "style_score", "op": ">=", "value": 0.9},
            ]
        }
        assert rubric._evaluate_gate(gate, metrics) is False

    def test_evaluate_gate_not_expression(self):
        """Test gate evaluation with NOT expression."""
        rubric = Rubric()
        metrics = {"verbose": 0.3}

        gate = {"NOT": {"func": "verbose", "op": ">", "value": 0.5}}
        assert rubric._evaluate_gate(gate, metrics) is True

        gate = {"NOT": {"func": "verbose", "op": "<", "value": 0.5}}
        assert rubric._evaluate_gate(gate, metrics) is False

    def test_evaluate_gate_nested_expression(self):
        """Test gate evaluation with nested AND/OR/NOT expressions."""
        rubric = Rubric()
        metrics = {"correct": 1.0, "format": 0.9, "verbose": 0.3}

        # (correct == 1.0) AND ((format >= 0.8) OR (NOT verbose > 0.5))
        gate = {
            "AND": [
                {"func": "correct", "op": "==", "value": 1.0},
                {
                    "OR": [
                        {"func": "format", "op": ">=", "value": 0.8},
                        {"NOT": {"func": "verbose", "op": ">", "value": 0.5}},
                    ]
                },
            ]
        }
        assert rubric._evaluate_gate(gate, metrics) is True

    def test_evaluate_gate_missing_func(self):
        """Test gate evaluation when referenced function is missing."""
        rubric = Rubric()
        metrics = {"correct_answer": 1.0}

        gate = {"func": "nonexistent", "op": ">=", "value": 0.5}
        # Should return False and log warning
        assert rubric._evaluate_gate(gate, metrics) is False

    @pytest.mark.asyncio
    async def test_score_group_grpo_mode(self):
        """Test that GRPO mode computes advantages as reward - mean."""

        def func1(completion, **kwargs):
            return float(len(completion))

        rubric = Rubric(funcs=[func1], weights=[1.0], advantage_mode="grpo")

        states = []
        for i, comp in enumerate(["a", "bb", "ccc"]):
            state = State(
                input=RolloutInput(
                    prompt="test",
                    answer="test",
                    task="test",
                    example_id=i,
                )
            )
            state["completion"] = comp
            state["trajectory"] = []
            state["timing"] = {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
                "start_time": 0.0,
            }
            states.append(state)

        score_sem = NullAsyncContext()
        await rubric.score_group(states, score_sem)

        # Rewards: [1.0, 2.0, 3.0], Mean = 2.0
        # Advantages: [-1.0, 0.0, 1.0]
        assert states[0]["advantage"] == pytest.approx(-1.0)
        assert states[1]["advantage"] == pytest.approx(0.0)
        assert states[2]["advantage"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_score_group_gdpo_mode_no_gates(self):
        """Test GDPO mode without gates (per-reward group normalization)."""

        def func1(completion, **kwargs):
            return float(len(completion))

        def func2(completion, **kwargs):
            return 1.0 if "x" in completion else 0.0

        rubric = Rubric(
            funcs=[func1, func2],
            weights=[1.0, 1.0],
            advantage_mode="gdpo",
        )

        # All states share the same example_id (same prompt, multiple rollouts)
        # This is the typical GDPO setup
        states = []
        for comp in ["a", "bx", "ccc"]:
            state = State(
                input=RolloutInput(
                    prompt="test",
                    answer="test",
                    task="test",
                    example_id=0,  # Same example_id for all
                )
            )
            state["completion"] = comp
            state["trajectory"] = []
            state["timing"] = {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
                "start_time": 0.0,
            }
            states.append(state)

        score_sem = NullAsyncContext()
        await rubric.score_group(states, score_sem)

        # Rewards still computed traditionally
        assert states[0]["reward"] == pytest.approx(1.0)  # len=1, no x
        assert states[1]["reward"] == pytest.approx(3.0)  # len=2, has x
        assert states[2]["reward"] == pytest.approx(3.0)  # len=3, no x

        # func1 rewards: [1, 2, 3] -> within-group z-score -> [-1.22, 0, 1.22]
        # func2 rewards: [0, 1, 0] -> within-group z-score -> [-0.71, 1.41, -0.71]
        # Sum: [-1.93, 1.41, 0.51]
        # Batch z-score of sum gives final advantages
        # The middle state (bx) should have highest advantage
        assert states[1]["advantage"] > states[0]["advantage"]
        assert states[1]["advantage"] > states[2]["advantage"]

    @pytest.mark.asyncio
    async def test_score_group_gdpo_mode_with_gates(self):
        """Test GDPO mode with gating conditions."""

        def correct_answer(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        def style_score(completion, **kwargs):
            return 0.8 if len(completion) > 3 else 0.2

        gates = {
            # style_score only counts when correct_answer >= 1.0
            "style_score": {"func": "correct_answer", "op": ">=", "value": 1.0}
        }

        rubric = Rubric(
            funcs=[correct_answer, style_score],
            weights=[1.0, 1.0],
            advantage_mode="gdpo",
            gates=gates,
        )

        # All states share same example_id (same prompt, multiple rollouts)
        states = []
        completions = ["correct", "wrong", "correct"]
        answers = ["correct", "correct", "correct"]
        for i in range(3):
            state = State(
                input=RolloutInput(
                    prompt="test",
                    answer=answers[i],
                    task="test",
                    example_id=0,  # Same example_id for all
                )
            )
            state["completion"] = completions[i]
            state["trajectory"] = []
            state["timing"] = {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
                "start_time": 0.0,
            }
            states.append(state)

        score_sem = NullAsyncContext()
        await rubric.score_group(states, score_sem)

        # State 0: correct=1.0, style=0.8 (gated, passes) -> rewards [1.0, 0.8]
        # State 1: correct=0.0, style=0.2 (gated to 0 because correct=0) -> rewards [0.0, 0.0]
        # State 2: correct=1.0, style=0.8 (gated, passes) -> rewards [1.0, 0.8]
        #
        # For correct_answer: [1, 0, 1] -> z-score within group
        # For style_score after gating: [0.8, 0, 0.8] -> z-score within group
        #
        # The "wrong" state should have worst advantage
        assert states[1]["advantage"] < states[0]["advantage"]
        assert states[1]["advantage"] < states[2]["advantage"]

    @pytest.mark.asyncio
    async def test_gdpo_backward_compatibility(self):
        """Test that GRPO mode (default) behavior is unchanged."""

        def simple_func(completion, **kwargs):
            return float(len(completion))

        # Without GDPO parameters (backward compatible)
        rubric = Rubric(funcs=[simple_func], weights=[1.0])

        states = []
        for i, comp in enumerate(["a", "bb", "ccc"]):
            state = State(
                input=RolloutInput(
                    prompt="test",
                    answer="test",
                    task="test",
                    example_id=i,
                )
            )
            state["completion"] = comp
            state["trajectory"] = []
            state["timing"] = {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
                "start_time": 0.0,
            }
            states.append(state)

        score_sem = NullAsyncContext()
        await rubric.score_group(states, score_sem)

        # GRPO: advantage = reward - mean
        # Rewards: [1, 2, 3], Mean = 2
        assert states[0]["advantage"] == pytest.approx(-1.0)
        assert states[1]["advantage"] == pytest.approx(0.0)
        assert states[2]["advantage"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_gdpo_group_wise_normalization(self):
        """Test that GDPO normalizes within groups (same example_id) per paper Eq. 4."""

        def reward_func(completion, **kwargs):
            return float(len(completion))

        rubric = Rubric(
            funcs=[reward_func],
            weights=[1.0],
            advantage_mode="gdpo",
        )

        # Create 2 groups with 3 rollouts each (6 total states)
        # Group 0: completions "a", "bb", "ccc" -> rewards [1, 2, 3]
        # Group 1: completions "x", "yy", "zzz" -> rewards [1, 2, 3]
        states = []
        for group_id in [0, 1]:
            for comp in ["a", "bb", "ccc"] if group_id == 0 else ["x", "yy", "zzz"]:
                state = State(
                    input=RolloutInput(
                        prompt=f"prompt_{group_id}",
                        answer="test",
                        task="test",
                        example_id=group_id,
                    )
                )
                state["completion"] = comp
                state["trajectory"] = []
                state["timing"] = {
                    "generation_ms": 0.0,
                    "scoring_ms": 0.0,
                    "total_ms": 0.0,
                    "start_time": 0.0,
                }
                states.append(state)

        score_sem = NullAsyncContext()
        await rubric.score_group(states, score_sem)

        # Within each group: rewards [1, 2, 3] -> z-score -> [-1.22, 0, 1.22] (approx)
        # Both groups have identical z-scores, so after batch normalization
        # the final advantages should be symmetric around 0

        # Group 0: indices 0, 1, 2
        # Group 1: indices 3, 4, 5
        # Within-group advantages before batch norm should be equal
        group0_advs = [
            states[0]["advantage"],
            states[1]["advantage"],
            states[2]["advantage"],
        ]
        group1_advs = [
            states[3]["advantage"],
            states[4]["advantage"],
            states[5]["advantage"],
        ]

        # Both groups should have same pattern (low, mid, high)
        assert group0_advs[0] < group0_advs[1] < group0_advs[2]
        assert group1_advs[0] < group1_advs[1] < group1_advs[2]

        # Since both groups are identical, corresponding positions should be equal
        for i in range(3):
            assert group0_advs[i] == pytest.approx(group1_advs[i], abs=1e-6)

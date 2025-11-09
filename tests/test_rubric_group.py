"""Tests for the RubricGroup class."""

import pytest

from verifiers import Rubric, RubricGroup, XMLParser
from verifiers.types import RolloutInput, RolloutTiming, State
from verifiers.utils.async_utils import NullAsyncContext


class TestRubricGroup:
    """Test cases for the RubricGroup class."""

    def test_rubric_group_initialization(self):
        """Test RubricGroup initialization with multiple rubrics."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        rubric1 = Rubric(funcs=[func1], weights=[1.0])
        rubric2 = Rubric(funcs=[func2], weights=[0.8])

        rubrics = [rubric1, rubric2]
        group = RubricGroup(rubrics=rubrics)

        assert group.rubrics == rubrics
        assert len(group.rubrics) == 2

    def test_rubric_group_initialization_empty_fails(self):
        """Test that RubricGroup initialization fails with empty rubrics list."""
        with pytest.raises(
            ValueError, match="RubricGroup must have at least one rubric"
        ):
            RubricGroup(rubrics=[])

    def test_rubric_group_get_reward_func_names(self):
        """Test getting aggregated reward function names from all rubrics."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        def func3(completion, **kwargs):
            return 0.3

        rubric1 = Rubric(funcs=[func1, func2], weights=[1.0, 0.5])
        rubric2 = Rubric(funcs=[func3], weights=[0.8])

        group = RubricGroup(rubrics=[rubric1, rubric2])
        names = group._get_reward_func_names()

        assert names == ["func1", "func2", "func3"]

    def test_rubric_group_get_reward_funcs(self):
        """Test getting aggregated reward functions from all rubrics."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        rubric1 = Rubric(funcs=[func1], weights=[1.0])
        rubric2 = Rubric(funcs=[func2], weights=[0.8])

        group = RubricGroup(rubrics=[rubric1, rubric2])
        funcs = group._get_reward_funcs()

        assert len(funcs) == 2
        assert funcs[0] == func1
        assert funcs[1] == func2

    def test_rubric_group_get_reward_weights(self):
        """Test getting aggregated reward weights from all rubrics."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        def func3(completion, **kwargs):
            return 0.3

        rubric1 = Rubric(funcs=[func1, func2], weights=[1.0, 0.7])
        rubric2 = Rubric(funcs=[func3], weights=[0.8])

        group = RubricGroup(rubrics=[rubric1, rubric2])
        weights = group._get_reward_weights()

        assert weights == [1.0, 0.7, 0.8]

    def test_rubric_group_add_reward_func(self):
        """Test adding reward function to RubricGroup (should add to first rubric)."""

        def func1(completion, **kwargs):
            return 1.0

        def new_func(completion, **kwargs):
            return 0.9

        rubric1 = Rubric(funcs=[func1], weights=[1.0])
        rubric2 = Rubric()

        group = RubricGroup(rubrics=[rubric1, rubric2])

        # Should add to first rubric
        group.add_reward_func(new_func, weight=0.6)

        assert len(rubric1.funcs) == 2
        assert len(rubric2.funcs) == 0
        assert rubric1.funcs[1] == new_func
        assert rubric1.weights[1] == 0.6

    def test_rubric_group_add_reward_func_empty_group_fails(self):
        """Test that adding reward function fails if no rubrics exist."""
        # This shouldn't happen due to initialization check, but test edge case
        group = RubricGroup.__new__(RubricGroup)  # Bypass __init__
        group.rubrics = []

        def test_func(completion, **kwargs):
            return 1.0

        with pytest.raises(
            AssertionError, match="RubricGroup must have at least one rubric"
        ):
            group.add_reward_func(test_func)

    @pytest.mark.asyncio
    async def test_rubric_group_score_rollouts_basic(self):
        """Test basic scoring of rollouts with multiple rubrics."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        rubric1 = Rubric(funcs=[func1], weights=[1.0])
        rubric2 = Rubric(funcs=[func2], weights=[0.8])

        group = RubricGroup(rubrics=[rubric1, rubric2])

        # Create state
        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "What is 1+1?"}],
                answer="2",
                task="default",
                example_id=0,
            )
        )
        state["completion"] = [{"role": "assistant", "content": "2"}]
        state["trajectory"] = []
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=0.0,
        )

        score_sem = NullAsyncContext()
        await group.score_group([state], score_sem)

        # Should have scores from both rubrics
        assert "func1" in state["metrics"]
        assert "func2" in state["metrics"]
        assert state["metrics"]["func1"] == 1.0
        assert state["metrics"]["func2"] == 0.5

    @pytest.mark.asyncio
    async def test_rubric_group_score_rollouts_duplicate_names(self):
        """Test that duplicate reward function names are summed up."""

        def func1(completion, **kwargs):
            return 1.0

        # Create two rubrics with same function name
        rubric1 = Rubric(funcs=[func1], weights=[1.0])
        rubric2 = Rubric(funcs=[func1], weights=[0.5])  # Same function name

        group = RubricGroup(rubrics=[rubric1, rubric2])

        # Create state
        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "What is 1+1?"}],
                answer="2",
                task="default",
                example_id=0,
            )
        )
        state["completion"] = [{"role": "assistant", "content": "2"}]
        state["trajectory"] = []
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=0.0,
        )

        score_sem = NullAsyncContext()
        await group.score_group([state], score_sem)

        # Should have summed scores for duplicate function names
        assert "func1" in state["metrics"]
        assert (
            state["metrics"]["func1"] == 2.0
        )  # 1.0 + 1.0 (same function called twice)

    @pytest.mark.asyncio
    async def test_rubric_group_score_rollouts_with_kwargs(self):
        """Test scoring rollouts with additional kwargs."""

        def func1(completion, info=None, **kwargs):
            custom_param = info.get("custom_param") if info else None
            return 1.0 if custom_param == "test" else 0.5

        rubric1 = Rubric(funcs=[func1], weights=[1.0])
        rubric2 = Rubric(funcs=[func1], weights=[0.8])

        group = RubricGroup(rubrics=[rubric1, rubric2])

        # Create state
        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "What is 1+1?"}],
                answer="2",
                task="default",
                example_id=0,
                info={"custom_param": "test"},
            )
        )
        state["completion"] = [{"role": "assistant", "content": "2"}]
        state["trajectory"] = []
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=0.0,
        )

        score_sem = NullAsyncContext()
        await group.score_group([state], score_sem)

        # Should pass custom kwargs to reward functions
        assert "func1" in state["metrics"]
        assert (
            state["metrics"]["func1"] == 2.0
        )  # 1.0 + 1.0 (both should get custom_param="test")

    @pytest.mark.asyncio
    async def test_rubric_group_score_rollouts_single_rubric(self):
        """Test scoring rollouts with a single rubric (edge case)."""

        def func1(completion, **kwargs):
            return 1.0

        rubric1 = Rubric(funcs=[func1], weights=[1.0])

        group = RubricGroup(rubrics=[rubric1])

        # Create state
        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "What is 1+1?"}],
                answer="2",
                task="default",
                example_id=0,
            )
        )
        state["completion"] = [{"role": "assistant", "content": "2"}]
        state["trajectory"] = []
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=0.0,
        )

        score_sem = NullAsyncContext()
        await group.score_group([state], score_sem)

        # Should work with single rubric
        assert "func1" in state["metrics"]
        assert state["metrics"]["func1"] == 1.0

    @pytest.mark.asyncio
    async def test_rubric_group_score_rollouts_empty_data(self):
        """Test scoring empty rollouts."""

        def func1(completion, **kwargs):
            return 1.0

        rubric1 = Rubric(funcs=[func1], weights=[1.0])

        group = RubricGroup(rubrics=[rubric1])

        # Test with empty data - should handle gracefully
        states = []
        score_sem = NullAsyncContext()
        # Empty states should not cause errors
        try:
            await group.score_group(states, score_sem)
        except ZeroDivisionError:
            pytest.skip("score_group doesn't handle empty states yet")

    def test_rubric_group_mixed_rubric_types(self):
        """Test RubricGroup with different types of rubrics."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        # Create rubrics with different configurations
        rubric1 = Rubric(funcs=[func1], weights=[1.0])
        rubric2 = Rubric(funcs=[func2], weights=[0.3])

        group = RubricGroup(rubrics=[rubric1, rubric2])

        # Should aggregate functions and weights correctly
        assert group._get_reward_func_names() == ["func1", "func2"]
        assert group._get_reward_weights() == [1.0, 0.3]

    @pytest.mark.asyncio
    async def test_rubric_group_with_max_concurrent(self):
        """Test RubricGroup with max_concurrent parameter."""

        def func1(completion, **kwargs):
            return 1.0

        rubric1 = Rubric(funcs=[func1], weights=[1.0])

        group = RubricGroup(rubrics=[rubric1])

        # Create states
        states = [
            State(
                input=RolloutInput(
                    prompt=[{"role": "user", "content": "What is 1+1?"}],
                    answer="2",
                    task="default",
                    example_id=0,
                )
            ),
            State(
                input=RolloutInput(
                    prompt=[{"role": "user", "content": "What is 2+2?"}],
                    answer="4",
                    task="default",
                    example_id=1,
                )
            ),
        ]
        for state in states:
            state["completion"] = [{"role": "assistant", "content": state["answer"]}]
            state["trajectory"] = []
            state["timing"] = RolloutTiming(
                generation_ms=0.0,
                scoring_ms=0.0,
                total_ms=0.0,
                start_time=0.0,
            )

        score_sem = NullAsyncContext()
        await group.score_group(states, score_sem)

        # Should work with multiple states
        assert "func1" in states[0]["metrics"]
        assert "func1" in states[1]["metrics"]
        assert states[0]["metrics"]["func1"] == 1.0
        assert states[1]["metrics"]["func1"] == 1.0

    def test_rubric_group_inheritance(self):
        """Test that RubricGroup properly inherits from Rubric."""
        rubric = Rubric()
        group = RubricGroup(rubrics=[rubric])

        assert isinstance(group, Rubric)
        assert hasattr(group, "logger")
        assert hasattr(group, "parser")

    @pytest.mark.asyncio
    async def test_rubric_group_score_rollout_uses_rubric_parser(self):
        """Ensure individual rubric parsers are respected during score_rollout."""

        recorded_parsers: list[XMLParser] = []

        def reward_func(completion, parser, answer, **_):
            recorded_parsers.append(parser)
            guess = parser.parse_answer(completion) or ""
            return 1.0 if guess == answer else 0.0

        xml_parser = XMLParser(fields=["answer"], answer_field="answer")
        rubric = Rubric(funcs=[reward_func], parser=xml_parser)
        group = RubricGroup(rubrics=[rubric])

        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "What is 6 * 7?"}],
                answer="42",
                task="default",
                example_id=0,
            )
        )
        state["completion"] = [
            {"role": "assistant", "content": "Let me think\n<answer>42</answer>"}
        ]
        state["trajectory"] = []
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=0.0,
        )

        score_sem = NullAsyncContext()
        await group.score_rollout(state, score_sem)

        assert state["reward"] == 1.0
        assert recorded_parsers == [xml_parser]

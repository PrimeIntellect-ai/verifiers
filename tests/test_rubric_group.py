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


class TestRubricGroupGDPO:
    """Test cases for RubricGroup GDPO (Group Reward-Decoupled Policy Optimization) support."""

    def test_rubric_group_inherits_gdpo_mode(self):
        """Test that RubricGroup inherits advantage_mode from first GDPO rubric."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        gates = {"func2": {"func": "func1", "op": ">=", "value": 1.0}}

        # First rubric is GDPO
        rubric_gdpo = Rubric(
            funcs=[func1, func2],
            weights=[1.0, 0.5],
            advantage_mode="gdpo",
            gates=gates,
            epsilon=1e-6,
        )
        # Second rubric is default GRPO (like a monitor)
        rubric_grpo = Rubric(funcs=[], weights=[])

        group = RubricGroup(rubrics=[rubric_gdpo, rubric_grpo])

        # RubricGroup should inherit GDPO settings from first GDPO rubric
        assert group.advantage_mode == "gdpo"
        assert group.gates == gates
        assert group.epsilon == 1e-6

    def test_rubric_group_stays_grpo_when_no_gdpo_rubrics(self):
        """Test that RubricGroup stays GRPO when no GDPO rubrics are present."""

        def func1(completion, **kwargs):
            return 1.0

        rubric1 = Rubric(funcs=[func1], weights=[1.0])
        rubric2 = Rubric(funcs=[], weights=[])

        group = RubricGroup(rubrics=[rubric1, rubric2])

        # Should remain GRPO (default)
        assert group.advantage_mode == "grpo"
        assert group.gates == {}

    def test_rubric_group_finds_gdpo_in_second_position(self):
        """Test that RubricGroup finds GDPO rubric even if not first."""

        def func1(completion, **kwargs):
            return 1.0

        gates = {"func1": {"func": "func1", "op": ">=", "value": 0.5}}

        # First rubric is GRPO
        rubric_grpo = Rubric(funcs=[], weights=[])
        # Second rubric is GDPO
        rubric_gdpo = Rubric(
            funcs=[func1],
            weights=[1.0],
            advantage_mode="gdpo",
            gates=gates,
        )

        group = RubricGroup(rubrics=[rubric_grpo, rubric_gdpo])

        # Should find and inherit from the GDPO rubric
        assert group.advantage_mode == "gdpo"
        assert group.gates == gates

    @pytest.mark.asyncio
    async def test_rubric_group_preserves_gdpo_advantages(self):
        """Test that RubricGroup preserves GDPO advantages and doesn't let monitors overwrite."""

        def correct_answer(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        def length_reward(completion, **kwargs):
            # Short responses get reward even if wrong (the problem GDPO solves)
            return 1.0 if len(str(completion)) < 10 else 0.0

        gates = {
            # length_reward only counts when correct_answer >= 1.0
            "length_reward": {"func": "correct_answer", "op": ">=", "value": 1.0}
        }

        gdpo_rubric = Rubric(
            funcs=[correct_answer, length_reward],
            weights=[1.0, 0.5],
            advantage_mode="gdpo",
            gates=gates,
        )

        # Monitor rubric (like MultiTurnMonitorRubric) - GRPO mode, weight 0
        def num_turns(state, **kwargs):
            return 1.0

        monitor_rubric = Rubric(funcs=[num_turns], weights=[0.0])

        group = RubricGroup(rubrics=[gdpo_rubric, monitor_rubric])

        # Create states: one correct, one wrong (but short, so would get length reward in GRPO)
        states = [
            State(
                input=RolloutInput(
                    prompt="test",
                    answer="correct",
                    task="test",
                    example_id=0,
                )
            ),
            State(
                input=RolloutInput(
                    prompt="test",
                    answer="correct",
                    task="test",
                    example_id=0,
                )
            ),
        ]
        states[0]["completion"] = "correct"  # Correct answer
        states[1]["completion"] = "wrong"  # Wrong but short

        for state in states:
            state["trajectory"] = [{"advantage": None, "reward": None}]
            state["timing"] = RolloutTiming(
                generation_ms=0.0, scoring_ms=0.0, total_ms=0.0, start_time=0.0
            )

        score_sem = NullAsyncContext()
        await group.score_group(states, score_sem)

        # GDPO should give correct answer much higher advantage than wrong answer
        # because length_reward is gated on correctness
        assert states[0]["advantage"] > states[1]["advantage"]

        # The gap should be significant (not just the 0.5 from length reward)
        # In GRPO without gating, the gap would be smaller
        advantage_gap = states[0]["advantage"] - states[1]["advantage"]
        assert advantage_gap > 1.0  # Significant gap due to gating

    @pytest.mark.asyncio
    async def test_rubric_group_grpo_computes_advantages_from_aggregated_rewards(self):
        """Test that GRPO mode computes advantages as reward - mean from aggregated rewards."""

        def func1(completion, **kwargs):
            return float(len(str(completion)))

        def func2(completion, **kwargs):
            return 1.0

        rubric1 = Rubric(funcs=[func1], weights=[1.0])
        rubric2 = Rubric(funcs=[func2], weights=[0.5])

        group = RubricGroup(rubrics=[rubric1, rubric2])

        # Verify it's GRPO mode
        assert group.advantage_mode == "grpo"

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
            state["trajectory"] = [{"advantage": None, "reward": None}]
            state["timing"] = RolloutTiming(
                generation_ms=0.0, scoring_ms=0.0, total_ms=0.0, start_time=0.0
            )
            states.append(state)

        score_sem = NullAsyncContext()
        await group.score_group(states, score_sem)

        # Rewards from rubric1: [1.0, 2.0, 3.0] (len of completion)
        # Rewards from rubric2: [1.0, 1.0, 1.0] * 0.5 = [0.5, 0.5, 0.5]
        # Aggregated: [1.5, 2.5, 3.5], Mean = 2.5
        # Advantages: [-1.0, 0.0, 1.0]
        assert states[0]["advantage"] == pytest.approx(-1.0)
        assert states[1]["advantage"] == pytest.approx(0.0)
        assert states[2]["advantage"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_rubric_group_gdpo_vs_grpo_advantage_difference(self):
        """Test that GDPO and GRPO produce different advantages for same inputs."""

        def correct_answer(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        def length_reward(completion, **kwargs):
            return 1.0 if len(str(completion)) < 20 else 0.0

        gates = {"length_reward": {"func": "correct_answer", "op": ">=", "value": 1.0}}

        def make_states():
            states = []
            completions = ["4", "wrong"]  # One correct (short), one wrong (short)
            for i, comp in enumerate(completions):
                state = State(
                    input=RolloutInput(
                        prompt="What is 2+2?",
                        answer="4",
                        task="math",
                        example_id=0,
                    )
                )
                state["completion"] = comp
                state["trajectory"] = [{"advantage": None, "reward": None}]
                state["timing"] = RolloutTiming(
                    generation_ms=0.0, scoring_ms=0.0, total_ms=0.0, start_time=0.0
                )
                states.append(state)
            return states

        # GDPO rubric
        gdpo_rubric = Rubric(
            funcs=[correct_answer, length_reward],
            weights=[1.0, 0.5],
            advantage_mode="gdpo",
            gates=gates,
        )
        gdpo_group = RubricGroup(rubrics=[gdpo_rubric])

        # GRPO rubric (same funcs, no gating)
        grpo_rubric = Rubric(
            funcs=[correct_answer, length_reward],
            weights=[1.0, 0.5],
            advantage_mode="grpo",
        )
        grpo_group = RubricGroup(rubrics=[grpo_rubric])

        score_sem = NullAsyncContext()

        # Score with GDPO
        gdpo_states = make_states()
        await gdpo_group.score_group(gdpo_states, score_sem)
        gdpo_gap = gdpo_states[0]["advantage"] - gdpo_states[1]["advantage"]

        # Score with GRPO
        grpo_states = make_states()
        await grpo_group.score_group(grpo_states, score_sem)
        grpo_gap = grpo_states[0]["advantage"] - grpo_states[1]["advantage"]

        # GDPO should have larger advantage gap due to gating
        # (wrong answer doesn't get length reward in GDPO)
        assert gdpo_gap > grpo_gap

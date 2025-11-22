"""
Tests for GEPA integration: Rubric feedback support and GEPAAdapter.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import verifiers as vf
from verifiers.types import RewardResult, State


def require_gepa_adapter():
    """Import GEPAAdapter or skip tests if the module is unavailable."""
    module = pytest.importorskip("verifiers.adapters.gepa")
    return module.GEPAAdapter


class TestRubricFeedback:
    """Tests for Rubric class feedback support."""

    def test_rubric_with_dict_return(self):
        """Test Rubric with reward function returning dict."""

        def reward_with_feedback(completion, answer, **kwargs) -> RewardResult:
            correct = completion == answer
            return {
                "score": 1.0 if correct else 0.0,
                "feedback": f"Expected: {answer}, Got: {completion}",
            }

        rubric = vf.Rubric()
        rubric.add_reward_func(reward_with_feedback)

        assert len(rubric.funcs) == 1
        assert rubric.funcs[0] == reward_with_feedback

    def test_rubric_with_float_return(self):
        """Test Rubric with reward function returning float (backward compat)."""

        def simple_reward(completion, answer, **kwargs) -> float:
            return 1.0 if completion == answer else 0.0

        rubric = vf.Rubric()
        rubric.add_reward_func(simple_reward)

        assert len(rubric.funcs) == 1
        assert rubric.funcs[0] == simple_reward

    def test_rubric_mixed_functions(self):
        """Test Rubric with mix of dict and float returning functions."""

        def reward_with_feedback(completion, answer, **kwargs) -> RewardResult:
            return {
                "score": 1.0 if completion == answer else 0.0,
                "feedback": "Detailed feedback",
            }

        def simple_reward(completion, **kwargs) -> float:
            return 0.5

        rubric = vf.Rubric()
        rubric.add_reward_func(reward_with_feedback, weight=1.0)
        rubric.add_reward_func(simple_reward, weight=0.5)

        assert len(rubric.funcs) == 2

    @pytest.mark.asyncio
    async def test_get_feedback_with_feedbacks(self):
        """Test get_feedback when state has feedbacks."""
        rubric = vf.Rubric()

        state = State(input={})
        state["reward"] = 0.75
        state["feedbacks"] = [
            "reward_1: Good job!",
            "reward_2: Could be better",
        ]

        feedback = rubric.get_feedback(state)

        assert "0.75" in feedback or "75" in feedback  # Score percentage
        assert "Good job!" in feedback
        assert "Could be better" in feedback

    @pytest.mark.asyncio
    async def test_get_feedback_without_feedbacks(self):
        """Test get_feedback when state has no feedbacks (fallback)."""
        rubric = vf.Rubric()

        state = State(input={})
        state["reward"] = 0.5

        feedback = rubric.get_feedback(state)

        assert "0.5" in feedback or "50" in feedback
        assert "no detailed feedback" in feedback.lower()


class TestGEPAAdapter:
    """Tests for GEPAAdapter class."""

    def test_gepa_adapter_initialization(self):
        """Test GEPAAdapter initializes correctly."""
        GEPAAdapter = require_gepa_adapter()

        # Create mock environment
        env = MagicMock(spec=vf.SingleTurnEnv)
        env.system_prompt = "Test prompt"
        env.dataset = None
        env.eval_dataset = None
        env.parser = vf.Parser()
        env.rubric = vf.Rubric()
        env.sampling_args = {}
        env.message_type = "chat"
        env.max_workers = 512

        client = AsyncMock()

        adapter = GEPAAdapter(
            env=env,
            client=client,
            model="gpt-4o-mini",
            sampling_args={"temperature": 1.0},
            components_to_optimize=["system_prompt"],
        )

        assert adapter.base_env == env
        assert adapter.model == "gpt-4o-mini"
        assert "system_prompt" in adapter.components_to_optimize

    def test_gepa_adapter_tool_descriptions_validation(self):
        """Test GEPAAdapter validates tool_descriptions component."""
        GEPAAdapter = require_gepa_adapter()

        # Create mock environment WITHOUT tools
        env = MagicMock(spec=vf.SingleTurnEnv)
        env.system_prompt = "Test prompt"
        env.oai_tools = None

        client = AsyncMock()

        # Should raise error when trying to optimize tool_descriptions without tools
        with pytest.raises(ValueError, match="no tools"):
            GEPAAdapter(
                env=env,
                client=client,
                model="gpt-4o-mini",
                sampling_args={},
                components_to_optimize=["tool_descriptions"],
            )

    def test_gepa_adapter_build_program(self):
        """Test GEPAAdapter.build_program creates new environment with updated components."""
        GEPAAdapter = require_gepa_adapter()

        # Create real environment
        dataset = vf.load_example_dataset(n=5)
        env = vf.SingleTurnEnv(
            dataset=dataset,
            system_prompt="Original prompt",
            rubric=vf.Rubric(),
        )

        client = AsyncMock()

        adapter = GEPAAdapter(
            env=env,
            client=client,
            model="gpt-4o-mini",
            sampling_args={},
            components_to_optimize=["system_prompt"],
        )

        # Build new program with updated system_prompt
        candidate = {"system_prompt": "Optimized prompt"}
        new_env = adapter.build_program(candidate)

        assert new_env.system_prompt == "Optimized prompt"
        assert new_env.system_prompt != env.system_prompt

    def test_gepa_adapter_extract_seed_candidate(self):
        """Test extracting seed candidate from environment."""
        dataset = vf.load_example_dataset(n=5)
        env = vf.SingleTurnEnv(
            dataset=dataset,
            system_prompt="Test prompt",
            rubric=vf.Rubric(),
        )

        # Verify we can extract the system_prompt
        assert hasattr(env, "system_prompt")
        assert env.system_prompt == "Test prompt"

    def test_gepa_adapter_evaluate_uses_generate(self):
        """Integration test ensuring evaluate() calls env.generate correctly."""
        GEPAAdapter = require_gepa_adapter()

        base_env = MagicMock(spec=vf.Environment)
        base_env.dataset = None
        base_env.eval_dataset = None
        base_env.parser = vf.Parser()
        base_env.rubric = vf.Rubric()
        base_env.sampling_args = {}
        base_env.message_type = "chat"
        base_env.max_workers = 1
        base_env.system_prompt = "Base system"
        base_env.few_shot = None
        base_env.env_id = "stub-env"
        base_env.oai_tools = []

        adapter = GEPAAdapter(
            env=base_env,
            client=AsyncMock(),
            model="stub-model",
            sampling_args={"temperature": 0.1},
            components_to_optimize=["system_prompt"],
            num_rollouts_per_example=1,
        )

        class StubEnv:
            def __init__(self):
                self.dataset = None
                self.eval_dataset = None
                self.parser = base_env.parser
                self.rubric = base_env.rubric
                self.sampling_args = {}
                self.message_type = "chat"
                self.system_prompt = "Stub system"
                self.few_shot = None
                self.env_id = "stub-env"
                self.max_workers = 1
                self.oai_tools = []
                self.last_inputs = None

            async def generate(
                self,
                inputs,
                client,
                model,
                sampling_args=None,
                max_concurrent=-1,
                use_tqdm=True,
            ):
                self.last_inputs = inputs
                return {
                    "completion": [[{"role": "assistant", "content": "42"}]],
                    "state": [
                        {
                            "prompt": [
                                {"role": "system", "content": "Stub system"},
                                {"role": "user", "content": "What is 6*7?"},
                            ],
                            "completion": [{"role": "assistant", "content": "42"}],
                            "reward": 0.9,
                        }
                    ],
                    "reward": [0.9],
                }

        stub_env = StubEnv()
        batch = [
            {
                "question": "What is 6*7?",
                "answer": "42",
                "task": "math",
                "info": {},
            }
        ]

        with patch.object(adapter, "build_program", return_value=stub_env):
            result = adapter.evaluate(
                batch, candidate={"system_prompt": "Stub system"}, capture_traces=True
            )

        assert stub_env.last_inputs is not None
        assert stub_env.last_inputs[0]["task"] == "math"
        # Prompt should include system + user messages
        assert isinstance(stub_env.last_inputs[0]["prompt"], list)
        assert stub_env.last_inputs[0]["prompt"][-1]["content"] == "What is 6*7?"

        assert result.scores == [0.9]
        assert result.outputs == [[{"role": "assistant", "content": "42"}]]
        assert result.trajectories is not None
        assert result.trajectories[0]["score"] == 0.9


class TestRubricDictSupport:
    """Tests for base Rubric class dict return support."""

    @pytest.mark.asyncio
    async def test_rubric_score_rollout_with_dict_return(self):
        """Test that score_rollout handles dict returns from reward functions."""

        def reward_with_feedback(completion, answer, **kwargs) -> RewardResult:
            return {
                "score": 0.8,
                "feedback": "Good answer",
            }

        rubric = vf.Rubric()
        rubric.add_reward_func(reward_with_feedback)

        # Create minimal state
        state = State(
            input={
                "prompt": [{"role": "user", "content": "test"}],
                "example_id": 0,
                "task": "test",
                "answer": "correct",
            }
        )
        state["prompt"] = [{"role": "user", "content": "test"}]
        state["completion"] = [{"role": "assistant", "content": "response"}]
        state["task"] = "test"
        state["timing"] = {"scoring_ms": 0.0, "total_ms": 0.0}

        # Mock score_sem
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_sem():
            yield

        await rubric.score_rollout(state, score_sem=mock_sem())

        # Check that reward was extracted correctly
        assert state["reward"] == 0.8
        assert "reward_with_feedback" in state["metrics"]
        assert state["metrics"]["reward_with_feedback"] == 0.8

        # Check that feedback was stored
        assert "feedbacks" in state
        assert len(state["feedbacks"]) == 1
        assert "Good answer" in state["feedbacks"][0]

    @pytest.mark.asyncio
    async def test_rubric_score_rollout_with_float_return(self):
        """Test that score_rollout still handles float returns (backward compat)."""

        def simple_reward(completion, answer, **kwargs) -> float:
            return 0.5

        rubric = vf.Rubric()
        rubric.add_reward_func(simple_reward)

        # Create minimal state
        state = State(
            input={
                "prompt": [{"role": "user", "content": "test"}],
                "example_id": 0,
                "task": "test",
                "answer": "correct",
            }
        )
        state["prompt"] = [{"role": "user", "content": "test"}]
        state["completion"] = [{"role": "assistant", "content": "response"}]
        state["task"] = "test"
        state["timing"] = {"scoring_ms": 0.0, "total_ms": 0.0}

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_sem():
            yield

        await rubric.score_rollout(state, score_sem=mock_sem())

        # Check that reward was extracted correctly
        assert state["reward"] == 0.5
        assert "simple_reward" in state["metrics"]
        assert state["metrics"]["simple_reward"] == 0.5

        # Feedbacks should be empty for float returns
        assert "feedbacks" in state
        assert len(state["feedbacks"]) == 0

"""
Tests for GEPA integration: Rubric feedback support and GEPAAdapter.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import verifiers as vf
from verifiers.types import RewardResult, State


def require_gepa_adapter():
    """Import GEPAAdapter or skip tests if the module is unavailable."""
    module = pytest.importorskip("verifiers.gepa.adapter")
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
        """Test GEPAAdapter.build_program creates new environment with updated components.

        Important: datasets are shared (not copied) for efficiency via shallow copy.
        The adapter provides inputs directly via _build_rollout_inputs.
        """
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

        # Verify component was updated
        assert new_env.system_prompt == "Optimized prompt"
        assert new_env.system_prompt != env.system_prompt

        # Verify dataset is shared (shallow copy - most efficient)
        assert new_env.dataset is not None
        assert new_env.dataset is env.dataset  # Same reference (shared)

        # Verify rubric is also shared (preserves feedback functions)
        assert new_env.rubric is env.rubric

    def test_gepa_adapter_build_program_multiturn_env(self):
        """Test build_program with MultiTurnEnv (uses **kwargs)."""
        GEPAAdapter = require_gepa_adapter()

        # Create a simple MultiTurnEnv
        dataset = vf.load_example_dataset(n=5)

        class TestMultiTurnEnv(vf.MultiTurnEnv):
            async def env_response(self, messages, state, **kwargs):
                return [{"role": "user", "content": "test"}]

        env = TestMultiTurnEnv(
            dataset=dataset,
            system_prompt="Original prompt",
            rubric=vf.Rubric(),
            max_turns=3,
        )

        client = AsyncMock()
        adapter = GEPAAdapter(
            env=env,
            client=client,
            model="gpt-4o-mini",
            sampling_args={},
            components_to_optimize=["system_prompt"],
        )

        candidate = {"system_prompt": "Optimized prompt"}
        new_env = adapter.build_program(candidate)

        # Verify component was updated
        assert new_env.system_prompt == "Optimized prompt"
        # Verify dataset is shared (shallow copy)
        assert new_env.dataset is not None
        assert new_env.dataset is env.dataset

    def test_gepa_adapter_build_program_tool_env(self):
        """Test build_program with ToolEnv."""
        GEPAAdapter = require_gepa_adapter()

        def example_tool(x: int) -> int:
            return x * 2

        dataset = vf.load_example_dataset(n=5)

        class TestToolEnv(vf.ToolEnv):
            def __init__(self, **kwargs):
                super().__init__(tools=[example_tool], **kwargs)

        env = TestToolEnv(
            dataset=dataset,
            system_prompt="Use the tool",
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

        candidate = {"system_prompt": "Use the tool wisely"}
        new_env = adapter.build_program(candidate)

        # Verify component was updated
        assert new_env.system_prompt == "Use the tool wisely"
        # Verify dataset is shared (shallow copy)
        assert new_env.dataset is not None
        assert new_env.dataset is env.dataset
        assert new_env.oai_tools is not None  # Tools preserved

    def test_gepa_adapter_build_program_stateful_tool_env(self):
        """Test build_program with StatefulToolEnv."""
        GEPAAdapter = require_gepa_adapter()

        def stateful_tool(x: int, state_val: int) -> int:
            return x + state_val

        dataset = vf.load_example_dataset(n=5)

        class TestStatefulToolEnv(vf.StatefulToolEnv):
            def __init__(self, **kwargs):
                super().__init__(tools=[stateful_tool], **kwargs)

            def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
                return {**tool_args, "state_val": 10}

        env = TestStatefulToolEnv(
            dataset=dataset,
            system_prompt="Stateful tool env",
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

        candidate = {"system_prompt": "Updated stateful prompt"}
        new_env = adapter.build_program(candidate)

        # Verify component was updated
        assert new_env.system_prompt == "Updated stateful prompt"
        # Verify dataset is shared (shallow copy)
        assert new_env.dataset is not None
        assert new_env.dataset is env.dataset

    def test_gepa_adapter_build_program_internal_dataset_env(self):
        """Test build_program with env that creates dataset internally."""
        GEPAAdapter = require_gepa_adapter()

        class InternalDatasetEnv(vf.SingleTurnEnv):
            """Mock env that creates dataset internally like TextArenaEnv."""

            def __init__(
                self,
                num_train_examples: int = 10,
                num_eval_examples: int = 0,
                system_prompt: str | None = None,
                **kwargs,
            ):
                # Create dataset internally (like TextArenaEnv does)
                from datasets import Dataset

                rows = [
                    {"question": f"q{i}", "answer": f"a{i}"}
                    for i in range(num_train_examples)
                ]
                dataset = Dataset.from_list(rows)

                self.num_train_examples = num_train_examples
                self.num_eval_examples = num_eval_examples

                super().__init__(
                    dataset=dataset,
                    system_prompt=system_prompt,
                    rubric=vf.Rubric(),
                    **kwargs,
                )

        env = InternalDatasetEnv(
            num_train_examples=100,
            system_prompt="Internal dataset env",
        )

        client = AsyncMock()
        adapter = GEPAAdapter(
            env=env,
            client=client,
            model="gpt-4o-mini",
            sampling_args={},
            components_to_optimize=["system_prompt"],
        )

        candidate = {"system_prompt": "Updated internal prompt"}
        new_env = adapter.build_program(candidate)

        # Verify component was updated
        assert new_env.system_prompt == "Updated internal prompt"
        # Verify dataset is shared (shallow copy preserves all attributes)
        assert new_env.dataset is not None
        assert new_env.dataset is env.dataset  # Shared reference
        assert len(new_env.dataset) == 100  # Original dataset preserved
        assert new_env.num_train_examples == 100

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

    def test_gepa_adapter_tool_metadata_extraction(self):
        """Test that GEPAAdapter extracts tool metadata for tool_descriptions."""
        GEPAAdapter = require_gepa_adapter()

        def search_tool(query: str, max_results: int = 10) -> str:
            """Search for information about a query.

            Args:
                query: The search query string
                max_results: Maximum number of results to return
            """
            return f"Results for: {query}"

        dataset = vf.load_example_dataset(n=5)
        env = vf.ToolEnv(
            dataset=dataset,
            tools=[search_tool],
            system_prompt="Use the search tool",
            rubric=vf.Rubric(),
        )

        client = AsyncMock()
        adapter = GEPAAdapter(
            env=env,
            client=client,
            model="gpt-4o-mini",
            sampling_args={},
            components_to_optimize=["tool_descriptions"],
        )

        # Verify tool metadata was extracted
        assert "tool_0_description" in adapter._tool_metadata
        assert adapter._tool_metadata["tool_0_description"]["name"] == "search_tool"
        assert "parameters" in adapter._tool_metadata["tool_0_description"]

        # Verify parameters include the function arguments
        params = adapter._tool_metadata["tool_0_description"]["parameters"]
        assert "properties" in params
        assert "query" in params["properties"]
        assert "max_results" in params["properties"]

    def test_gepa_adapter_propose_new_texts_tool_descriptions(self):
        """Test that propose_new_texts uses tool-specific template for tool descriptions."""
        GEPAAdapter = require_gepa_adapter()

        def calculate(x: int, y: int) -> int:
            """Add two numbers together."""
            return x + y

        dataset = vf.load_example_dataset(n=5)
        env = vf.ToolEnv(
            dataset=dataset,
            tools=[calculate],
            system_prompt="Use the calculator",
            rubric=vf.Rubric(),
        )

        client = AsyncMock()
        adapter = GEPAAdapter(
            env=env,
            client=client,
            model="gpt-4o-mini",
            sampling_args={},
            components_to_optimize=["tool_descriptions"],
        )

        # Mock reflection_lm
        reflection_output = "```\nImproved tool description that adds two numbers with better clarity.\n```"
        adapter.reflection_lm = MagicMock(return_value=reflection_output)

        # Create mock candidate and reflective dataset
        candidate = {"tool_0_description": "Add two numbers together."}
        reflective_dataset = {
            "tool_0_description": [
                {
                    "Inputs": {"Task": "Calculate 2 + 3"},
                    "Generated Outputs": "Tool Call: calculate(x=2, y=3)",
                    "Feedback": "Correct usage",
                }
            ]
        }

        # Call propose_new_texts
        new_texts = adapter.propose_new_texts(
            candidate=candidate,
            reflective_dataset=reflective_dataset,
            components_to_update=["tool_0_description"],
        )

        # Verify the reflection_lm was called
        assert adapter.reflection_lm.called
        called_prompt = adapter.reflection_lm.call_args[0][0]

        # Verify tool name is in the prompt
        assert "calculate" in called_prompt

        # Verify tool parameters are in the prompt (JSON schema)
        assert "parameters" in called_prompt.lower()
        assert '"x"' in called_prompt or "'x'" in called_prompt
        assert '"y"' in called_prompt or "'y'" in called_prompt

        # Verify current description is in the prompt
        assert "Add two numbers together" in called_prompt

        # Verify new text was extracted correctly
        assert "tool_0_description" in new_texts
        assert "Improved tool description" in new_texts["tool_0_description"]

    def test_gepa_adapter_propose_new_texts_system_prompt(self):
        """Test that propose_new_texts uses default GEPA template for system_prompt."""
        GEPAAdapter = require_gepa_adapter()

        dataset = vf.load_example_dataset(n=5)
        env = vf.SingleTurnEnv(
            dataset=dataset,
            system_prompt="Original system prompt",
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

        # Mock reflection_lm
        reflection_output = "```\nImproved system prompt with better instructions.\n```"
        adapter.reflection_lm = MagicMock(return_value=reflection_output)

        # Create mock candidate and reflective dataset
        candidate = {"system_prompt": "Original system prompt"}
        reflective_dataset = {
            "system_prompt": [
                {
                    "Inputs": {"Task": "Solve this problem"},
                    "Generated Outputs": "Here's the solution",
                    "Feedback": "Good response",
                }
            ]
        }

        # Call propose_new_texts
        new_texts = adapter.propose_new_texts(
            candidate=candidate,
            reflective_dataset=reflective_dataset,
            components_to_update=["system_prompt"],
        )

        # Verify the reflection_lm was called
        assert adapter.reflection_lm.called
        called_prompt = adapter.reflection_lm.call_args[0][0]

        # Verify it uses the default GEPA template (should NOT contain tool-specific language)
        assert "TOOL NAME" not in called_prompt
        assert "TOOL PARAMETERS" not in called_prompt

        # Should contain the default GEPA language about "assistant" and "instructions"
        assert (
            "assistant" in called_prompt.lower()
            or "instruction" in called_prompt.lower()
        )

        # Verify new text was extracted correctly
        assert "system_prompt" in new_texts


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

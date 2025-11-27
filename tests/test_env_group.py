"""Tests for the EnvGroup class."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from datasets import Dataset

from verifiers import EnvGroup, Rubric, SingleTurnEnv
from verifiers.envs.env_group import EnvGroupRubric
from verifiers.types import RolloutInput, State
from verifiers.utils.async_utils import NullAsyncContext


class TestEnvGroupRubric:
    """Test cases for the EnvGroupRubric class."""

    def test_env_group_rubric_initialization(self, mock_openai_client):
        """Test EnvGroupRubric initialization with multiple environments."""

        # Create test environments with different rubrics
        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        def func3(completion, **kwargs):
            return 0.8

        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(funcs=[func1, func2], weights=[1.0, 0.5]),
        )

        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["q2"], "answer": ["a2"]}),
            rubric=Rubric(funcs=[func2, func3], weights=[0.7, 1.0]),
        )

        env_map = {"task1": env1, "task2": env2}
        rubric = EnvGroupRubric(env_map)

        assert rubric.env_map == env_map
        # Should have all unique reward function names
        assert set(rubric.all_reward_names) == {"func1", "func2", "func3"}

    @pytest.mark.asyncio
    async def test_env_group_rubric_score_rollout(self, mock_openai_client):
        """Test scoring a rollout with EnvGroupRubric."""

        # Create test environments
        def func1(completion, **kwargs):
            return 0.8

        def func2(completion, **kwargs):
            return 0.6

        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(funcs=[func1], weights=[1.0]),
        )

        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["q2"], "answer": ["a2"]}),
            rubric=Rubric(funcs=[func2], weights=[1.0]),
        )

        env_map = {"math": env1, "code": env2}
        rubric = EnvGroupRubric(env_map)

        # Test scoring for "math" task
        state = State(
            input=RolloutInput(
                prompt="Test prompt",
                answer="Test answer",
                task="math",
                example_id=0,
            )
        )
        state["completion"] = "Test completion"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }
        state["is_completed"] = False
        state["stop_condition"] = None
        state["oai_tools"] = []
        state["reward"] = None
        state["metrics"] = None
        score_sem = NullAsyncContext()

        await rubric.score_rollout(state, score_sem)

        assert "func1" in state["metrics"]
        assert "func2" in state["metrics"]
        assert state["metrics"]["func1"] == 0.8  # From env1
        assert state["metrics"]["func2"] == 0.0  # Not in env1, so 0.0
        assert state["reward"] == 0.8

    @pytest.mark.asyncio
    async def test_env_group_rubric_unknown_task(self, mock_openai_client):
        """Test scoring with unknown task returns zeros."""
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(),
        )

        env_map = {"known_task": env1}
        rubric = EnvGroupRubric(env_map)

        state = State(
            input=RolloutInput(
                prompt="Test",
                task="unknown_task",
                example_id=0,
            )
        )
        state["completion"] = "Test"
        state["trajectory"] = []
        state["timing"] = {
            "generation_ms": 0.0,
            "scoring_ms": 0.0,
            "total_ms": 0.0,
            "start_time": 0.0,
        }
        state["is_completed"] = False
        state["stop_condition"] = None
        state["oai_tools"] = []
        state["reward"] = None
        state["metrics"] = None
        score_sem = NullAsyncContext()

        await rubric.score_rollout(state, score_sem)

        assert state["reward"] == 0.0


class TestEnvGroup:
    """Test cases for the EnvGroup class."""

    def test_env_group_initialization(self, mock_openai_client):
        """Test EnvGroup initialization with multiple environments."""
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(),
        )

        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q2"], "answer": ["a2"]}),
            rubric=Rubric(),
        )

        env_group = EnvGroup(envs=[env1, env2])

        assert len(env_group.envs) == 2
        assert env_group.env_names == ["env_0", "env_1"]
        assert env_group.env_map["env_0"] == env1
        assert env_group.env_map["env_1"] == env2

    def test_env_group_unique_example_ids(self, mock_openai_client):
        """Test EnvGroup initialization with multiple environments."""
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(),
        )

        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q2"], "answer": ["a2"]}),
            rubric=Rubric(),
        )

        env_group = EnvGroup(envs=[env1, env2])
        dataset = env_group.get_dataset()
        assert "example_id" in dataset.column_names
        assert len(set(dataset["example_id"])) == len(dataset)

    def test_env_group_with_custom_names(self, mock_openai_client):
        """Test EnvGroup with custom environment names."""
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(),
        )

        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q2"], "answer": ["a2"]}),
            rubric=Rubric(),
        )

        env_group = EnvGroup(envs=[env1, env2], env_names=["math", "code"])

        assert env_group.env_names == ["math", "code"]
        assert env_group.env_map["math"] == env1
        assert env_group.env_map["code"] == env2

    def test_env_group_empty_envs_fails(self):
        """Test that EnvGroup fails with empty environments list."""
        with pytest.raises(
            ValueError, match="EnvGroup requires at least one environment"
        ):
            EnvGroup(envs=[])

    def test_env_group_mismatched_names_fails(self, mock_openai_client):
        """Test that EnvGroup fails when env_names length doesn't match envs."""
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(),
        )

        with pytest.raises(
            ValueError, match="Number of env_names must match number of envs"
        ):
            EnvGroup(envs=[env1], env_names=["math", "code"])

    def test_env_group_dataset_concatenation(self, mock_openai_client):
        """Test that EnvGroup properly concatenates datasets with task labels."""
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict(
                {"question": ["q1", "q2"], "answer": ["a1", "a2"]}
            ),
            rubric=Rubric(),
        )

        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q3"], "answer": ["a3"]}),
            rubric=Rubric(),
        )

        env_group = EnvGroup(envs=[env1, env2], env_names=["math", "code"])

        # Check concatenated dataset
        dataset = env_group.get_dataset()
        assert len(dataset) == 3
        assert "task" in dataset.column_names

        # Check task labels
        tasks = dataset["task"]
        assert tasks[0] == "math"
        assert tasks[1] == "math"
        assert tasks[2] == "code"

    def test_env_group_rubric_type(self, mock_openai_client):
        """Test that EnvGroup creates EnvGroupRubric."""
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(),
        )

        env_group = EnvGroup(envs=[env1])

        assert isinstance(env_group.rubric, EnvGroupRubric)
        assert env_group.rubric.env_map["env_0"] == env1

    @pytest.mark.asyncio
    async def test_env_group_rollout_routing(self, mock_openai_client):
        """Test that rollout is properly routed to the correct sub-environment."""
        # Create environments with different behaviors
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(),
        )

        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q2"], "answer": ["a2"]}),
            rubric=Rubric(),
        )

        # Mock the rollout methods to return different values
        async def env1_rollout(*args, **kwargs):
            state = State(
                input=RolloutInput(prompt="Test prompt", task="math", example_id=0)
            )
            state["env"] = "env1"
            return state

        async def env2_rollout(*args, **kwargs):
            state = State(
                input=RolloutInput(prompt="Test prompt", task="code", example_id=0)
            )
            state["env"] = "env2"
            return state

        # Explicitly mark shadowing as intentional for the type checker, and keep references to mocks
        env1_rollout_mock = AsyncMock(side_effect=env1_rollout)
        env2_rollout_mock = AsyncMock(side_effect=env2_rollout)
        env1.rollout = env1_rollout_mock  # type: ignore[method-assign]
        env2.rollout = env2_rollout_mock  # type: ignore[method-assign]

        env_group = EnvGroup(envs=[env1, env2], env_names=["math", "code"])

        # Test routing to math environment
        state1 = await env_group.rollout(
            input=RolloutInput(prompt="Test prompt", task="math", example_id=0),
            client=mock_openai_client,
            model="test-model",
        )

        assert state1["env"] == "env1"
        env1_rollout_mock.assert_called_once()
        env2_rollout_mock.assert_not_called()

        # Reset mocks
        env1_rollout_mock.reset_mock()
        env2_rollout_mock.reset_mock()

        # Test routing to code environment
        state2 = await env_group.rollout(
            input=RolloutInput(prompt="Test prompt", task="code", example_id=0),
            client=mock_openai_client,
            model="test-model",
        )

        assert state2["env"] == "env2"
        env1_rollout_mock.assert_not_called()
        env2_rollout_mock.assert_called_once()

    def test_get_env_for_task(self, mock_openai_client):
        """Test getting environment for a specific task."""
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(),
        )

        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q2"], "answer": ["a2"]}),
            rubric=Rubric(),
        )

        env_group = EnvGroup(envs=[env1, env2], env_names=["math", "code"])

        assert env_group.get_env_for_task("math") == env1
        assert env_group.get_env_for_task("code") == env2
        # Unknown task returns first environment as fallback
        assert env_group.get_env_for_task("unknown") == env1

    @pytest.mark.asyncio
    async def test_env_group_generate(self, mock_openai_client):
        """Test generate method with EnvGroup."""
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            rubric=Rubric(),
        )

        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q2"], "answer": ["a2"]}),
            rubric=Rubric(),
        )

        env_group = EnvGroup(envs=[env1, env2], env_names=["math", "code"])

        # Mock the scoring with a properly-typed cast
        from typing import cast

        async def mock_score_group(states, score_sem=None):
            for state in states:
                state["reward"] = 0.8 if state["task"] == "math" else 0.9
                state["metrics"] = {}

        cast(Any, env_group.rubric).score_group = mock_score_group

        inputs = [
            RolloutInput(
                prompt=[{"role": "user", "content": "Math question"}],
                answer="math_answer",
                task="math",
                example_id=0,
            ),
            RolloutInput(
                prompt=[{"role": "user", "content": "Code question"}],
                answer="code_answer",
                task="code",
                example_id=1,
            ),
        ]

        results = await env_group.generate(
            inputs, client=mock_openai_client, model="test-model"
        )

        assert "completion" in results
        assert "state" in results
        assert "reward" in results
        assert len(results["completion"]) == 2

    def test_env_group_with_mixed_datasets(self, mock_openai_client):
        """Test EnvGroup with environments having different dataset configurations."""
        # Environment with both train and eval datasets
        env1 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q1"], "answer": ["a1"]}),
            eval_dataset=Dataset.from_dict({"question": ["eq1"], "answer": ["ea1"]}),
            rubric=Rubric(),
        )

        # Environment with only eval dataset
        env2 = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=Dataset.from_dict({"question": ["q2"], "answer": ["a2"]}),
            eval_dataset=Dataset.from_dict({"question": ["eq2"], "answer": ["ea2"]}),
            rubric=Rubric(),
        )

        env_group = EnvGroup(envs=[env1, env2], env_names=["task1", "task2"])

        # Should have concatenated train dataset from both envs
        train_dataset = env_group.get_dataset()
        assert len(train_dataset) == 2
        assert train_dataset["task"][0] == "task1"
        assert train_dataset["task"][1] == "task2"

        # Should have concatenated eval datasets from both
        eval_dataset = env_group.get_eval_dataset()
        assert eval_dataset is not None
        assert len(eval_dataset) == 2
        assert eval_dataset["task"][0] == "task1"
        assert eval_dataset["task"][1] == "task2"

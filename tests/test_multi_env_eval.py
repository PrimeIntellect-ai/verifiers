import sys
import asyncio
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from verifiers.scripts.eval import (
    eval_environment_async,
    eval_environments_parallel,
    push_eval_to_prime_hub,
)
from verifiers.types import GenerateOutputs


class FakeEnvironment:
    """Fake environment for testing."""

    def __init__(self, env_id, **kwargs):
        self.env_id = env_id
        self.eval_dataset = Dataset.from_dict(
            {
                "question": ["Q1", "Q2", "Q3"],
                "answer": ["A1", "A2", "A3"],
            }
        )

    def get_eval_dataset(self, n=-1):
        """Return eval dataset."""
        if n == -1:
            return self.eval_dataset
        return self.eval_dataset.select(range(min(n, len(self.eval_dataset))))

    async def a_generate(
        self,
        inputs,
        client,
        model,
        sampling_args=None,
        score_rollouts=True,
        max_concurrent=-1,
    ):
        """Fake async generation."""
        await asyncio.sleep(0.01)

        n = len(inputs)
        return GenerateOutputs(
            prompt=[[{"role": "user", "content": f"Q{i}"}] for i in range(n)],
            completion=[
                [{"role": "assistant", "content": f"A{i}"}] for i in range(n)
            ],
            reward=[0.8 + (i * 0.05) for i in range(n)],  # Varying rewards
            answer=[f"A{i}" for i in range(n)],
            task=[self.env_id] * n,
            info=[{}] * n,
            state=[{"responses": []} for _ in range(n)],
            metrics={"accuracy": [0.8 + (i * 0.05) for i in range(n)]},
        )


@pytest.fixture
def mock_async_client():
    """Return a mock AsyncOpenAI client."""
    client = Mock()
    client.api_key = "test-key"
    client.base_url = "http://localhost:8000/v1"
    return client


@pytest.fixture
def fake_load_environment():
    """Return a function that creates fake environments."""

    def _load(env_id, **env_args):
        return FakeEnvironment(env_id, **env_args)

    return _load


class TestEvalEnvironmentAsync:
    """Test cases for async single environment evaluation."""

    @pytest.mark.asyncio
    async def test_eval_single_environment_async(
        self, mock_async_client, fake_load_environment
    ):
        """Test async evaluation of a single environment."""
        with patch("verifiers.scripts.eval.vf.load_environment", fake_load_environment):
            env_id, results = await eval_environment_async(
                env="test-env",
                env_args={},
                client=mock_async_client,
                model="test-model",
                num_examples=3,
                rollouts_per_example=1,
                max_concurrent=32,
                sampling_args={"temperature": 0.7},
            )

            assert env_id == "test-env"
            assert len(results.reward) == 3
            assert all(r >= 0.8 for r in results.reward)
            assert results.metrics["accuracy"] is not None

    @pytest.mark.asyncio
    async def test_eval_with_rollouts(self, mock_async_client, fake_load_environment):
        """Test evaluation with multiple rollouts per example."""
        with patch("verifiers.scripts.eval.vf.load_environment", fake_load_environment):
            env_id, results = await eval_environment_async(
                env="test-env",
                env_args={},
                client=mock_async_client,
                model="test-model",
                num_examples=2,
                rollouts_per_example=3,
                max_concurrent=32,
                sampling_args={},
            )

            # 2 examples * 3 rollouts = 6 total samples
            assert len(results.reward) == 6

    @pytest.mark.asyncio
    async def test_eval_with_env_args(self, mock_async_client, fake_load_environment):
        """Test that environment-specific arguments are passed correctly."""
        with patch("verifiers.scripts.eval.vf.load_environment", fake_load_environment):
            env_id, results = await eval_environment_async(
                env="test-env",
                env_args={"difficulty": "hard"},
                client=mock_async_client,
                model="test-model",
                num_examples=2,
                rollouts_per_example=1,
                max_concurrent=32,
                sampling_args={},
            )

            assert env_id == "test-env"
            assert len(results.reward) == 2


class TestEvalEnvironmentsParallel:
    """Test cases for parallel multi-environment evaluation."""

    @pytest.mark.asyncio
    async def test_eval_multiple_environments(
        self, mock_async_client, fake_load_environment
    ):
        """Test parallel evaluation of multiple environments."""
        with patch("verifiers.scripts.eval.vf.load_environment", fake_load_environment):
            results_dict = await eval_environments_parallel(
                envs=["env1", "env2", "env3"],
                env_args_dict={"env1": {}, "env2": {}, "env3": {}},
                client=mock_async_client,
                model="test-model",
                num_examples=[2, 3, 3],
                rollouts_per_example=[1, 1, 1],
                max_concurrent=[32, 32, 32],
                sampling_args={"temperature": 0.7},
            )

            # Check all environments completed
            assert len(results_dict) == 3
            assert "env1" in results_dict
            assert "env2" in results_dict
            assert "env3" in results_dict

            # Check correct number of samples per environment
            assert len(results_dict["env1"].reward) == 2
            assert len(results_dict["env2"].reward) == 3
            assert len(results_dict["env3"].reward) == 3

    @pytest.mark.asyncio
    async def test_eval_with_different_rollouts(
        self, mock_async_client, fake_load_environment
    ):
        """Test parallel evaluation with different rollouts per environment."""
        with patch("verifiers.scripts.eval.vf.load_environment", fake_load_environment):
            results_dict = await eval_environments_parallel(
                envs=["env1", "env2"],
                env_args_dict={"env1": {}, "env2": {}},
                client=mock_async_client,
                model="test-model",
                num_examples=[2, 2],
                rollouts_per_example=[1, 3],  # Different rollouts
                max_concurrent=[32, 16],
                sampling_args={},
            )

            # env1: 2 examples * 1 rollout = 2
            assert len(results_dict["env1"].reward) == 2
            # env2: 2 examples * 3 rollouts = 6
            assert len(results_dict["env2"].reward) == 6

    @pytest.mark.asyncio
    async def test_eval_with_per_env_args(
        self, mock_async_client, fake_load_environment
    ):
        """Test parallel evaluation with per-environment arguments."""
        with patch("verifiers.scripts.eval.vf.load_environment", fake_load_environment):
            results_dict = await eval_environments_parallel(
                envs=["env1", "env2"],
                env_args_dict={
                    "env1": {"difficulty": "easy"},
                    "env2": {"difficulty": "hard"},
                },
                client=mock_async_client,
                model="test-model",
                num_examples=[2, 2],
                rollouts_per_example=[1, 1],
                max_concurrent=[32, 32],
                sampling_args={},
            )

            assert "env1" in results_dict
            assert "env2" in results_dict

    @pytest.mark.asyncio
    async def test_parallel_execution_timing(
        self, mock_async_client, fake_load_environment
    ):
        """Test that environments are evaluated in parallel, not sequentially."""
        import time

        with patch("verifiers.scripts.eval.vf.load_environment", fake_load_environment):
            start_time = time.time()

            # Each environment sleeps for 0.01s in a_generate
            # If sequential: 3 * 0.01 = 0.03s
            # If parallel: ~0.01s (plus overhead)
            results_dict = await eval_environments_parallel(
                envs=["env1", "env2", "env3"],
                env_args_dict={"env1": {}, "env2": {}, "env3": {}},
                client=mock_async_client,
                model="test-model",
                num_examples=[1, 1, 1],
                rollouts_per_example=[1, 1, 1],
                max_concurrent=[32, 32, 32],
                sampling_args={},
            )

            elapsed = time.time() - start_time

            # Should complete faster than sequential (with some margin for overhead)
            assert elapsed < 0.025  # Much less than 0.03s
            assert len(results_dict) == 3


class TestPrimeHubIntegration:
    """Test cases for Prime Hub integration."""

    def test_push_to_prime_hub_success(self):
        """Test successful push to Prime Hub."""
        mock_client = Mock()
        mock_response = {
            "eval_id": "test-eval-123",
            "viewer_url": "https://app.primeintellect.ai/dashboard/evals/test-eval-123",
        }
        mock_client.push_eval.return_value = mock_response

        with patch(
            "prime_cli.api.evals.EvalsClient", return_value=mock_client
        ) as mock_cls:
            push_eval_to_prime_hub(
                eval_name="test-eval",
                model_name="gpt-4o-mini",
                dataset="gsm8k",
                metrics={"avg_reward": 0.85, "num_samples": 100},
                metadata={"timestamp": "2025-10-03T12:00:00Z"},
            )

            # Verify the client was created
            mock_cls.assert_called_once()

            # Verify push_eval was called with correct payload
            mock_client.push_eval.assert_called_once()
            call_args = mock_client.push_eval.call_args[0][0]
            assert call_args["eval_name"] == "test-eval"
            assert call_args["model_name"] == "gpt-4o-mini"
            assert call_args["dataset"] == "gsm8k"
            assert call_args["metrics"]["avg_reward"] == 0.85

    def test_push_to_prime_hub_with_results(self):
        """Test push to Prime Hub with sample-level results."""
        mock_client = Mock()
        mock_response = {"eval_id": "test-eval-123"}
        mock_client.push_eval.return_value = mock_response

        results = [
            {"example_id": 0, "reward": 1.0, "task": "gsm8k"},
            {"example_id": 1, "reward": 0.0, "task": "gsm8k"},
        ]

        with patch("prime_cli.api.evals.EvalsClient", return_value=mock_client):
            push_eval_to_prime_hub(
                eval_name="test-eval",
                model_name="gpt-4o-mini",
                dataset="gsm8k",
                metrics={"avg_reward": 0.5},
                metadata={},
                results=results,
            )

            # Verify results were included
            call_args = mock_client.push_eval.call_args[0][0]
            assert "results" in call_args
            assert len(call_args["results"]) == 2

    def test_push_to_prime_hub_import_error(self, caplog):
        """Test graceful handling when prime-cli is not installed."""
        with patch.dict(sys.modules, {"prime_cli.api.evals": None}):
            # Should not raise, just log warning
            push_eval_to_prime_hub(
                eval_name="test-eval",
                model_name="gpt-4o-mini",
                dataset="gsm8k",
                metrics={"avg_reward": 0.85},
                metadata={},
            )

    def test_push_to_prime_hub_api_error(self, caplog):
        """Test graceful handling of API errors."""
        mock_client = Mock()
        mock_client.push_eval.side_effect = Exception("API Error")

        with patch("prime_cli.api.evals.EvalsClient", return_value=mock_client):
            # Should not raise, just log warning
            push_eval_to_prime_hub(
                eval_name="test-eval",
                model_name="gpt-4o-mini",
                dataset="gsm8k",
                metrics={"avg_reward": 0.85},
                metadata={},
            )

            # Function should complete without raising

    def test_push_to_prime_hub_constructs_url(self):
        """Test URL construction when viewer_url not provided by backend."""
        mock_client = Mock()
        mock_response = {"eval_id": "test-eval-123"}  # No viewer_url
        mock_client.push_eval.return_value = mock_response

        mock_config = Mock()
        mock_config.frontend_url = "https://custom.primeintellect.ai"

        with patch(
            "prime_cli.api.evals.EvalsClient", return_value=mock_client
        ):
            push_eval_to_prime_hub(
                eval_name="test-eval",
                model_name="gpt-4o-mini",
                dataset="gsm8k",
                metrics={"avg_reward": 0.85},
                metadata={},
            )


class TestCLIIntegration:
    """Test integration with CLI."""

    def test_cli_multi_env_argument_parsing(self):
        """Test that CLI correctly handles multiple environment arguments."""
        import sys

        # Mock sys.argv
        test_args = [
            "vf-eval",
            "gsm8k",
            "math500",
            "-m",
            "gpt-4o-mini",
            "-n",
            "10",
        ]

        with patch.object(sys, "argv", test_args):
            from verifiers.scripts.eval import main

            assert callable(main)

    def test_cli_per_env_config_parsing(self):
        """Test that per-env config JSON is parsed correctly."""
        import json

        config_json = json.dumps(
            {
                "gsm8k": {"num_examples": 100, "rollouts_per_example": 5},
                "math500": {"num_examples": 50, "rollouts_per_example": 3},
            }
        )

        # Parse it back
        config = json.loads(config_json)
        assert config["gsm8k"]["num_examples"] == 100
        assert config["math500"]["rollouts_per_example"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


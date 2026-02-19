"""Tests for post_rollout safety with empty/missing trajectory data."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset

from verifiers import State
from verifiers.envs.sandbox_env import SandboxEnv


@pytest.fixture
def sandbox_env():
    """Fixture to create a SandboxEnv instance with mocked dataset."""
    mock_dataset = Dataset.from_dict({"question": ["mock question"], "info": [{}]})

    mock_request_patcher = patch("verifiers.envs.sandbox_env.CreateSandboxRequest")

    mock_request_patcher.start()

    try:
        env = SandboxEnv(dataset=mock_dataset, max_retries=1, base_delay=0.1)
        env.logger = MagicMock()
        env.active_sandboxes = set()
        yield env
    finally:
        mock_request_patcher.stop()


class TestPostRolloutEmptyTrajectory:
    """Test that post_rollout handles empty/missing trajectory gracefully."""

    @pytest.mark.asyncio
    async def test_post_rollout_with_empty_trajectory(self, sandbox_env):
        """Test post_rollout when trajectory is empty list."""
        # Create state with empty trajectory
        state = State()
        state["trajectory"] = []
        state["sandbox_id"] = "test-sandbox"

        # Should not raise IndexError
        await sandbox_env.post_rollout(state)

    @pytest.mark.asyncio
    async def test_post_rollout_with_missing_trajectory(self, sandbox_env):
        """Test post_rollout when trajectory key is missing."""
        state = State()
        state["sandbox_id"] = "test-sandbox"
        # No trajectory key at all

        # Should not raise KeyError
        await sandbox_env.post_rollout(state)

    @pytest.mark.asyncio
    async def test_cleanup_with_empty_trajectory(self, sandbox_env):
        """Test cleanup handler runs safely with empty trajectory."""
        # Mock the sandbox client delete method
        sandbox_env.sandbox_client.delete = AsyncMock()

        state = State()
        state["trajectory"] = []
        state["sandbox_id"] = "test-sandbox"

        # Should not raise - cleanup includes post_rollout
        await sandbox_env.destroy_sandbox(state)

        # Verify sandbox was still cleaned up
        assert "test-sandbox" not in sandbox_env.active_sandboxes
        sandbox_env.sandbox_client.delete.assert_called_once_with("test-sandbox")

    @pytest.mark.asyncio
    async def test_destroy_sandbox_handles_post_rollout_error(self, sandbox_env):
        """Test destroy_sandbox continues even if post_rollout fails."""
        # Create a subclass with a broken post_rollout
        class BrokenPostRolloutEnv(SandboxEnv):
            async def post_rollout(self, state):
                # Simulate the bug - access empty trajectory
                if not state.get("trajectory"):
                    raise IndexError("trajectory is empty")

        # Create environment with broken post_rollout
        mock_dataset = Dataset.from_dict({"question": ["mock question"], "info": [{}]})
        with patch("verifiers.envs.sandbox_env.CreateSandboxRequest"):
            env = BrokenPostRolloutEnv(dataset=mock_dataset, max_retries=1, base_delay=0.1)
            env.logger = MagicMock()
            env.active_sandboxes.add("test-sandbox")
            env.sandbox_client.delete = AsyncMock()

        state = State()
        state["trajectory"] = []
        state["sandbox_id"] = "test-sandbox"

        # Should not raise - error is caught and logged
        await env.destroy_sandbox(state)

        # Verify warning was logged
        env.logger.warning.assert_called()
        warning_calls = [str(call) for call in env.logger.warning.call_args_list]
        assert any("post_rollout failed" in str(call) for call in warning_calls)

        # Verify sandbox was still cleaned up
        assert "test-sandbox" not in env.active_sandboxes
        env.sandbox_client.delete.assert_called_once_with("test-sandbox")


class TestCustomPostRolloutSafety:
    """Test custom post_rollout implementations use safe patterns."""

    @pytest.mark.asyncio
    async def test_custom_post_rollout_with_trajectory_access(self):
        """Test custom post_rollout that accesses trajectory[-1]."""
        # Create a custom env with safe trajectory access
        class CustomEnv(SandboxEnv):
            async def post_rollout(self, state):
                # Safe pattern - check before accessing
                if state.get("trajectory"):
                    last_step = state["trajectory"][-1]
                    state["last_prompt"] = last_step.get("prompt")

        mock_dataset = Dataset.from_dict({"question": ["mock question"], "info": [{}]})
        with patch("verifiers.envs.sandbox_env.CreateSandboxRequest"):
            env = CustomEnv(dataset=mock_dataset, max_retries=1, base_delay=0.1)
            env.logger = MagicMock()

        # Test with empty trajectory
        state = State()
        state["trajectory"] = []

        await env.post_rollout(state)
        assert state.get("last_prompt") is None  # Should not crash

        # Test with populated trajectory
        state["trajectory"] = [{"prompt": "test", "completion": "response"}]

        await env.post_rollout(state)
        assert state.get("last_prompt") == "test"

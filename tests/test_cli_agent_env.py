"""Tests for CliAgentEnv and HarborEnv."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset

import verifiers as vf


@pytest.fixture
def mock_sandbox_client():
    """Mock AsyncSandboxClient for testing."""
    with patch(
        "verifiers.envs.experimental.cli_agent_env.AsyncSandboxClient"
    ) as mock_client_cls:
        mock_client = MagicMock()
        mock_client.create = AsyncMock(return_value=MagicMock(id="test-sandbox-123"))
        mock_client.wait_for_creation = AsyncMock()
        mock_client.execute_command = AsyncMock(
            return_value=MagicMock(stdout="running")
        )
        mock_client.delete = AsyncMock()
        mock_client_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_sandbox_request():
    """Mock CreateSandboxRequest."""
    with patch(
        "verifiers.envs.experimental.cli_agent_env.CreateSandboxRequest"
    ) as mock_req:
        yield mock_req


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Test task"}]],
            "answer": ["expected"],
            "example_id": [0],
        }
    )


class TestCliAgentEnv:
    """Tests for CliAgentEnv."""

    def test_init_basic(self, sample_dataset):
        """Test basic initialization."""
        env = vf.CliAgentEnv(
            run_command="python agent.py",
            dataset=sample_dataset,
            rubric=vf.Rubric(),
        )
        assert env.run_command == "python agent.py"
        assert env.docker_image == "python:3.11-slim"
        assert env.interception_port == 8765
        assert env.timeout_seconds == 3600.0

    def test_init_custom_config(self, sample_dataset):
        """Test initialization with custom configuration."""
        env = vf.CliAgentEnv(
            run_command="bash run.sh",
            dataset=sample_dataset,
            rubric=vf.Rubric(),
            docker_image="node:18",
            interception_port=9000,
            timeout_seconds=1800.0,
            cpu_cores=2,
            memory_gb=4,
        )
        assert env.run_command == "bash run.sh"
        assert env.docker_image == "node:18"
        assert env.interception_port == 9000
        assert env.timeout_seconds == 1800.0
        assert env.cpu_cores == 2
        assert env.memory_gb == 4

    @pytest.mark.asyncio
    async def test_build_env_vars(self, sample_dataset):
        """Test environment variable building."""
        env = vf.CliAgentEnv(
            run_command="python agent.py",
            dataset=sample_dataset,
            rubric=vf.Rubric(),
            environment_vars={"CUSTOM_VAR": "value"},
        )
        state = {
            "interception_base_url": "https://test.trycloudflare.com/v1",
            "model": "gpt-4",
        }
        env_vars = await env.build_env_vars(state)

        assert env_vars["OPENAI_BASE_URL"] == "https://test.trycloudflare.com/v1"
        assert env_vars["OPENAI_MODEL"] == "gpt-4"
        assert env_vars["CUSTOM_VAR"] == "value"

    @pytest.mark.asyncio
    async def test_get_docker_image(self, sample_dataset):
        """Test docker image retrieval."""
        env = vf.CliAgentEnv(
            run_command="python agent.py",
            dataset=sample_dataset,
            rubric=vf.Rubric(),
            docker_image="python:3.11-slim",
        )
        state = {}
        image = await env.get_docker_image(state)
        assert image == "python:3.11-slim"

    @pytest.mark.asyncio
    async def test_agent_completed_stop_condition(self, sample_dataset):
        """Test the agent_completed stop condition."""
        env = vf.CliAgentEnv(
            run_command="python agent.py",
            dataset=sample_dataset,
            rubric=vf.Rubric(),
        )

        state = {"agent_completed": False}
        assert await env.agent_completed(state) is False

        state = {"agent_completed": True}
        assert await env.agent_completed(state) is True

    @pytest.mark.asyncio
    async def test_timeout_reached_stop_condition(self, sample_dataset):
        """Test the timeout_reached stop condition."""
        env = vf.CliAgentEnv(
            run_command="python agent.py",
            dataset=sample_dataset,
            rubric=vf.Rubric(),
            timeout_seconds=10.0,
        )
        import time

        state = {"timing": {"start_time": time.time()}}
        assert await env.timeout_reached(state) is False

        state = {"timing": {"start_time": time.time() - 20}}
        assert await env.timeout_reached(state) is True

    def test_extract_tunnel_url_from_line(self, sample_dataset):
        """Test tunnel URL extraction from cloudflared output."""
        env = vf.CliAgentEnv(
            run_command="python agent.py",
            dataset=sample_dataset,
            rubric=vf.Rubric(),
        )

        line = "2024-01-01 INFO: https://abc-xyz.trycloudflare.com is ready"
        url = env.extract_tunnel_url_from_line(line)
        assert url == "https://abc-xyz.trycloudflare.com"

        line = "Some other log line without URL"
        url = env.extract_tunnel_url_from_line(line)
        assert url is None

    @pytest.mark.asyncio
    async def test_env_response_returns_empty(self, sample_dataset):
        """Test that env_response returns empty list."""
        env = vf.CliAgentEnv(
            run_command="python agent.py",
            dataset=sample_dataset,
            rubric=vf.Rubric(),
        )
        messages = [{"role": "assistant", "content": "test"}]
        state = {}
        response = await env.env_response(messages, state)
        assert response == []


class TestHarborEnv:
    """Tests for HarborEnv."""

    @pytest.fixture
    def harbor_task_dir(self):
        """Create a temporary Harbor task directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "test_task"
            task_path.mkdir()

            # Create task.toml
            task_toml = task_path / "task.toml"
            task_toml.write_text('[environment]\ndocker_image = "python:3.11"\n')

            # Create instruction.md
            instruction = task_path / "instruction.md"
            instruction.write_text("# Test Task\n\nDo something.")

            yield Path(tmpdir)

    def test_init_loads_dataset(self, harbor_task_dir):
        """Test that HarborEnv loads tasks from directory."""
        env = vf.HarborEnv(
            run_command="python agent.py",
            dataset_path=harbor_task_dir,
        )
        assert len(env.dataset) == 1
        assert env.dataset[0]["task"] == "test_task"

    def test_init_filters_tasks(self, harbor_task_dir):
        """Test that HarborEnv can filter tasks by name."""
        # Create a second task
        task2_path = harbor_task_dir / "task2"
        task2_path.mkdir()
        (task2_path / "task.toml").write_text("[environment]\n")
        (task2_path / "instruction.md").write_text("Task 2")

        env = vf.HarborEnv(
            run_command="python agent.py",
            dataset_path=harbor_task_dir,
            tasks=["test_task"],
        )
        assert len(env.dataset) == 1
        assert env.dataset[0]["task"] == "test_task"

    def test_init_raises_on_empty_dataset(self):
        """Test that HarborEnv raises when no valid tasks found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No valid Harbor tasks"):
                vf.HarborEnv(
                    run_command="python agent.py",
                    dataset_path=tmpdir,
                )

    def test_init_raises_on_missing_path(self):
        """Test that HarborEnv raises when path doesn't exist."""
        with pytest.raises(FileNotFoundError):
            vf.HarborEnv(
                run_command="python agent.py",
                dataset_path="/nonexistent/path",
            )

    @pytest.mark.asyncio
    async def test_get_docker_image_from_task(self, harbor_task_dir):
        """Test docker image retrieval from task config."""
        env = vf.HarborEnv(
            run_command="python agent.py",
            dataset_path=harbor_task_dir,
        )
        state = {
            "info": {"docker_image": "custom:latest"},
        }
        image = await env.get_docker_image(state)
        assert image == "custom:latest"

    @pytest.mark.asyncio
    async def test_get_docker_image_fallback(self, harbor_task_dir):
        """Test docker image fallback to default."""
        env = vf.HarborEnv(
            run_command="python agent.py",
            dataset_path=harbor_task_dir,
            docker_image="fallback:latest",
        )
        state = {"info": {}}
        image = await env.get_docker_image(state)
        assert image == "fallback:latest"

    @pytest.mark.asyncio
    async def test_build_env_vars_harbor_specific(self, harbor_task_dir):
        """Test Harbor-specific environment variables."""
        env = vf.HarborEnv(
            run_command="python agent.py",
            dataset_path=harbor_task_dir,
            agent_workdir="/workspace",
        )
        state = {
            "interception_base_url": "https://test.trycloudflare.com/v1",
            "task": "my_task",
        }
        env_vars = await env.build_env_vars(state)

        assert env_vars["HARBOR_TASK_NAME"] == "my_task"
        assert env_vars["HARBOR_TASK_DIR"] == "/task"
        assert env_vars["HARBOR_INSTRUCTION_PATH"] == "/task/instruction.md"
        assert env_vars["AGENT_WORKDIR"] == "/workspace"

    @pytest.mark.asyncio
    async def test_harbor_reward_returns_cached(self, harbor_task_dir):
        """Test harbor_reward returns cached reward from state."""
        env = vf.HarborEnv(
            run_command="python agent.py",
            dataset_path=harbor_task_dir,
        )
        state = {"reward": 0.75}
        reward = await env.harbor_reward(state)
        assert reward == 0.75

    @pytest.mark.asyncio
    async def test_harbor_reward_default_zero(self, harbor_task_dir):
        """Test harbor_reward returns 0 when no reward cached."""
        env = vf.HarborEnv(
            run_command="python agent.py",
            dataset_path=harbor_task_dir,
        )
        state = {}
        reward = await env.harbor_reward(state)
        assert reward == 0.0

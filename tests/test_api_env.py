"""Tests for ApiEnv."""

import asyncio
import time

import pytest

import verifiers as vf


def noop_agent(base_url: str, state: vf.State):
    """Sync agent that does nothing."""
    pass


async def async_noop_agent(base_url: str, state: vf.State):
    """Async agent that does nothing."""
    pass


class TestApiEnv:
    """Tests for ApiEnv."""

    def test_init_basic(self, sample_chat_dataset):
        """Test basic initialization."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
            interception_port=8765,
        )
        assert env.agent_fn is noop_agent
        assert env.interception_port == 8765
        assert env.timeout_seconds == 3600.0
        assert env.poll_interval == 1.0
        assert env.use_tunnel is False

    def test_init_custom_config(self, sample_chat_dataset):
        """Test initialization with custom configuration."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
            interception_port=9000,
            timeout_seconds=120.0,
            poll_interval=0.5,
            use_tunnel=True,
        )
        assert env.interception_port == 9000
        assert env.timeout_seconds == 120.0
        assert env.poll_interval == 0.5
        assert env.use_tunnel is True

    def test_init_auto_port(self, sample_chat_dataset):
        """Test that a free port is auto-assigned when not specified."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )
        assert env.interception_port > 0

    @pytest.mark.asyncio
    async def test_agent_completed_stop_condition(self, sample_chat_dataset):
        """Test the agent_completed stop condition."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )

        state = {"agent_completed": False}
        assert await env.agent_completed(state) is False

        state = {"agent_completed": True}
        assert await env.agent_completed(state) is True

    @pytest.mark.asyncio
    async def test_timeout_reached_stop_condition(self, sample_chat_dataset):
        """Test the timeout_reached stop condition."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
            timeout_seconds=10.0,
        )

        state: dict = {"timing": {"start_time": time.time()}}
        assert await env.timeout_reached(state) is False
        assert state.get("agent_timed_out") is None

        state = {"timing": {"start_time": time.time() - 20}}
        assert await env.timeout_reached(state) is True
        assert state["agent_timed_out"] is True

    @pytest.mark.asyncio
    async def test_env_response_returns_empty(self, sample_chat_dataset):
        """Test that env_response returns empty list."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )
        response = await env.env_response([], {})
        assert response == []

    @pytest.mark.asyncio
    async def test_normalize_intercepted_tools_oai_format(self, sample_chat_dataset):
        """Test that OpenAI-format tools are normalized to vf.Tool."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "echo tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        normalized = env.normalize_intercepted_tools(oai_tools)
        assert normalized is not None
        assert len(normalized) == 1
        assert normalized[0].name == "echo"
        assert normalized[0].description == "echo tool"

    @pytest.mark.asyncio
    async def test_normalize_intercepted_tools_passthrough(self, sample_chat_dataset):
        """Test that vf.Tool objects pass through normalization unchanged."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )
        tool = vf.Tool(name="echo", description="echo", parameters={})
        normalized = env.normalize_intercepted_tools([tool])
        assert normalized is not None
        assert normalized[0] is tool

    @pytest.mark.asyncio
    async def test_non_streaming_intercept_tools_use_oai_schema(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """OpenAI-formatted intercepted tools should work for non-streaming requests."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )
        state = await env.init_state(
            input=make_input(),
            client=mock_client,
            model="test-model",
        )
        request_id = "req-test"
        state["current_request_id"] = request_id
        env._interception_server.intercepts[request_id] = {
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "echo tool",
                        "parameters": {},
                    },
                }
            ],
        }

        response = await env.get_model_response(
            state=state,
            prompt=make_input()["prompt"],
            client=mock_client,
            model="test-model",
        )

        assert isinstance(response, vf.Response)
        kwargs = mock_client.last_call_kwargs
        assert kwargs["tools"] is not None
        assert kwargs["tools"][0].name == "echo"

    @pytest.mark.asyncio
    async def test_setup_state_local_base_url(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """Test that local base URL is computed correctly."""
        env = vf.ApiEnv(
            agent_fn=async_noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
            interception_port=9999,
        )
        state = await env.init_state(
            input=make_input(),
            client=mock_client,
            model="test-model",
        )
        state = await env.setup_state(state)

        try:
            assert "rollout_id" in state
            assert state["interception_base_url"].startswith(
                "http://localhost:9999/rollout/"
            )
            assert state["interception_base_url"].endswith("/v1")
            assert state["agent_completed"] is False
            assert "agent_task" in state
        finally:
            agent_task = state.get("agent_task")
            if agent_task and not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass
            rollout_id = state.get("rollout_id")
            if rollout_id:
                env._interception_server.unregister_rollout(rollout_id)
            await env._interception_server.stop()

    @pytest.mark.asyncio
    async def test_run_agent_fn_sync(self, sample_chat_dataset):
        """Test that sync agent_fn runs via to_thread."""
        results = {}

        def sync_agent(base_url: str, state: vf.State):
            results["base_url"] = base_url
            return "sync_result"

        env = vf.ApiEnv(
            agent_fn=sync_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )

        state: dict = {"agent_completed": False}
        await env._run_agent_fn(state, "http://localhost:1234/v1")

        assert state["agent_completed"] is True
        assert state["agent_result"] == "sync_result"
        assert results["base_url"] == "http://localhost:1234/v1"
        assert "agent_error" not in state

    @pytest.mark.asyncio
    async def test_run_agent_fn_async(self, sample_chat_dataset):
        """Test that async agent_fn is awaited directly."""
        results = {}

        async def async_agent(base_url: str, state: vf.State):
            results["base_url"] = base_url
            return "async_result"

        env = vf.ApiEnv(
            agent_fn=async_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )

        state: dict = {"agent_completed": False}
        await env._run_agent_fn(state, "http://localhost:1234/v1")

        assert state["agent_completed"] is True
        assert state["agent_result"] == "async_result"
        assert results["base_url"] == "http://localhost:1234/v1"
        assert "agent_error" not in state

    @pytest.mark.asyncio
    async def test_run_agent_fn_error_handling(self, sample_chat_dataset):
        """Test that agent_fn errors are caught and stored."""

        def failing_agent(base_url: str, state: vf.State):
            raise ValueError("agent exploded")

        env = vf.ApiEnv(
            agent_fn=failing_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )

        state: dict = {"agent_completed": False}
        await env._run_agent_fn(state, "http://localhost:1234/v1")

        assert state["agent_completed"] is True
        assert "agent exploded" in state["agent_error"]
        assert "agent_result" not in state

    @pytest.mark.asyncio
    async def test_cleanup_cancels_agent_task(self, sample_chat_dataset):
        """Test that cleanup cancels a running agent task."""

        async def slow_agent(base_url: str, state: vf.State):
            await asyncio.sleep(3600)

        env = vf.ApiEnv(
            agent_fn=slow_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )

        agent_task = asyncio.create_task(slow_agent("http://localhost:1234/v1", {}))
        await asyncio.sleep(0)

        state: dict = {
            "agent_task": agent_task,
            "rollout_id": None,
            "is_completed": True,
            "timing": {"total_ms": 0},
            "trajectory": [],
        }

        assert not agent_task.done()
        await env.cleanup_agent_and_interception(state)
        assert agent_task.done()

    @pytest.mark.asyncio
    async def test_get_model_response_agent_completed(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """Test that empty prompt returns agent-completed response."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )
        state = await env.init_state(
            input=make_input(),
            client=mock_client,
            model="test-model",
        )

        response = await env.get_model_response(state=state, prompt=[])

        assert response.id == "agent-completed"
        assert response.model == "test-model"
        assert response.message.content == ""

    @pytest.mark.asyncio
    async def test_add_model_response_skips_empty(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """Test that add_model_response skips empty prompt (agent completion)."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )
        state = await env.init_state(
            input=make_input(),
            client=mock_client,
            model="test-model",
        )

        dummy_response = vf.Response(
            id="test",
            created=0,
            model="test-model",
            usage=None,
            message=vf.ResponseMessage(
                content="",
                reasoning_content=None,
                tool_calls=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
            ),
        )

        await env.add_model_response(state, [], dummy_response)
        assert len(state["trajectory"]) == 0


class TestApiEnvMonitorRubric:
    """Tests for ApiEnvMonitorRubric."""

    @pytest.mark.asyncio
    async def test_defaults(self):
        """Test ApiEnvMonitorRubric returns 0 for clean state."""
        from verifiers.envs.experimental.api_env import ApiEnvMonitorRubric

        rubric = ApiEnvMonitorRubric()

        state: dict = {}
        assert await rubric.agent_timeout(state) == 0.0
        assert await rubric.agent_error(state) == 0.0

    @pytest.mark.asyncio
    async def test_records_issues(self):
        """Test ApiEnvMonitorRubric tracks timeout and error."""
        from verifiers.envs.experimental.api_env import ApiEnvMonitorRubric

        rubric = ApiEnvMonitorRubric()

        state: dict = {"agent_timed_out": True, "agent_error": "boom"}
        assert await rubric.agent_timeout(state) == 1.0
        assert await rubric.agent_error(state) == 1.0

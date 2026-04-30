"""Tests for ApiEnv."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

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
        # timeout_seconds default of None means no wall-clock cap, matching
        # MultiTurnEnv. --timeout N (CLI) or kwarg sets it.
        assert env.timeout_seconds is None
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
        """Test that port=0 is used when not specified (OS assigns at bind time)."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
        )
        # Port 0 means "let the OS pick a free port at server-start time".
        # The actual assigned port is only known after interception_server.start().
        assert env.interception_port == 0

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
    async def test_explicit_empty_intercept_tools_disables_state_tools(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """An explicit empty tools list overrides env-level state["tool_defs"]."""
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
        state["tool_defs"] = [
            vf.Tool(name="env_tool", description="env-level tool", parameters={})
        ]
        request_id = "req-empty-tools"
        state["current_request_id"] = request_id
        env._interception_server.intercepts[request_id] = {
            "stream": False,
            "tools": [],
        }

        response = await env.get_model_response(
            state=state,
            prompt=make_input()["prompt"],
            client=mock_client,
            model="test-model",
        )

        assert isinstance(response, vf.Response)
        kwargs = mock_client.last_call_kwargs
        assert kwargs["tools"] is None

    @pytest.mark.asyncio
    async def test_absent_intercept_tools_falls_back_to_state_tools(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """An absent tools field falls back to env-level state["tool_defs"]."""
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
        state["tool_defs"] = [
            vf.Tool(name="env_tool", description="env-level tool", parameters={})
        ]
        request_id = "req-no-tools-field"
        state["current_request_id"] = request_id
        env._interception_server.intercepts[request_id] = {
            "stream": False,
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
        assert kwargs["tools"][0].name == "env_tool"

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
        await env.setup_state(state)

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
        await env.cleanup_rollout(state)
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

    @pytest.mark.asyncio
    async def test_compute_base_url_custom_interception_url(self, sample_chat_dataset):
        """Test compute_base_url with a custom interception_url."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
            interception_url="https://my-proxy.example.com",
        )
        url = await env.compute_base_url({}, "rollout_abc")
        assert url == "https://my-proxy.example.com/rollout/rollout_abc/v1"

    @pytest.mark.asyncio
    async def test_compute_base_url_custom_interception_url_trailing_slash(
        self, sample_chat_dataset
    ):
        """Test compute_base_url strips trailing slash from interception_url."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
            interception_url="https://my-proxy.example.com/",
        )
        url = await env.compute_base_url({}, "rollout_xyz")
        assert url == "https://my-proxy.example.com/rollout/rollout_xyz/v1"


class TestPollNextRequest:
    """Tests for the _poll_next_request polling loop."""

    @pytest.mark.asyncio
    async def test_returns_request_id_from_queue(self, sample_chat_dataset):
        """Test that a queued request_id is returned immediately."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
            poll_interval=0.05,
        )
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put("req-123")
        state = {
            "request_id_queue": queue,
            "agent_completed": False,
            "timing": {"start_time": time.time()},
        }
        result = await env._poll_next_request(state)
        assert result == "req-123"

    @pytest.mark.asyncio
    async def test_returns_none_on_agent_completed(self, sample_chat_dataset):
        """Test that None is returned when the agent completes between polls."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
            poll_interval=0.05,
        )
        queue: asyncio.Queue = asyncio.Queue()
        state = {
            "request_id_queue": queue,
            "agent_completed": True,
            "timing": {"start_time": time.time()},
        }
        result = await env._poll_next_request(state)
        assert result is None

    @pytest.mark.asyncio
    async def test_waits_then_returns_request(self, sample_chat_dataset):
        """Test that the poller waits across poll cycles until a request arrives."""
        env = vf.ApiEnv(
            agent_fn=noop_agent,
            dataset=sample_chat_dataset,
            rubric=vf.Rubric(),
            poll_interval=0.05,
        )
        queue: asyncio.Queue = asyncio.Queue()
        state = {
            "request_id_queue": queue,
            "agent_completed": False,
            "timing": {"start_time": time.time()},
        }

        async def enqueue_later():
            await asyncio.sleep(0.1)
            await queue.put("req-delayed")

        asyncio.create_task(enqueue_later())
        result = await env._poll_next_request(state)
        assert result == "req-delayed"


class TestAddModelResponseFirstTurn:
    """Tests for add_model_response first-turn prompt update."""

    @pytest.mark.asyncio
    async def test_first_turn_updates_state_prompt(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """Test that the first turn replaces state['prompt'] with the agent's actual prompt."""
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
        original_prompt = state["prompt"]
        agent_prompt = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write hello world"},
        ]

        response = vf.Response(
            id="resp-1",
            created=0,
            model="test-model",
            usage=None,
            message=vf.ResponseMessage(
                content="print('hello world')",
                reasoning_content=None,
                tool_calls=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
            ),
        )

        assert state["trajectory"] == []
        await env.add_model_response(state, agent_prompt, response)

        # state["prompt"] should now be the agent's prompt, not the original
        assert state["prompt"] == agent_prompt
        assert state["prompt"] != original_prompt
        assert len(state["trajectory"]) == 1

    @pytest.mark.asyncio
    async def test_second_turn_does_not_update_prompt(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """Test that subsequent turns don't overwrite state['prompt']."""
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

        first_prompt = [{"role": "user", "content": "First turn"}]
        second_prompt = [{"role": "user", "content": "Second turn"}]

        response = vf.Response(
            id="resp-1",
            created=0,
            model="test-model",
            usage=None,
            message=vf.ResponseMessage(
                content="response",
                reasoning_content=None,
                tool_calls=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
            ),
        )

        await env.add_model_response(state, first_prompt, response)
        assert state["prompt"] == first_prompt

        await env.add_model_response(state, second_prompt, response)
        # prompt should still be from the first turn
        assert state["prompt"] == first_prompt


class TestGetModelResponseErrorDelivery:
    """Tests for get_model_response error delivery to HTTP handler."""

    @pytest.mark.asyncio
    async def test_error_delivered_to_handler_non_streaming(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """Test that when get_model_response raises, the error is still delivered
        to the HTTP handler via deliver_response so the agent doesn't hang."""
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

        request_id = "req-error-test"
        state["current_request_id"] = request_id
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        env._interception_server.intercepts[request_id] = {
            "stream": False,
            "tools": None,
            "messages": [{"role": "user", "content": "test"}],
            "response_future": future,
        }

        injected_error = RuntimeError("LLM call failed")

        with patch.object(
            vf.MultiTurnEnv,
            "get_model_response",
            new_callable=AsyncMock,
            side_effect=injected_error,
        ):
            with pytest.raises(RuntimeError, match="LLM call failed"):
                await env.get_model_response(
                    state=state,
                    prompt=[{"role": "user", "content": "test"}],
                    client=mock_client,
                    model="test-model",
                )

        # The future should have the error set so the HTTP handler unblocks
        assert future.done()
        with pytest.raises(RuntimeError, match="LLM call failed"):
            future.result()

    @pytest.mark.asyncio
    async def test_error_delivered_to_handler_streaming(
        self, sample_chat_dataset, mock_client, make_input
    ):
        """Test that streaming intercepts also get errors delivered via synthesize_stream."""
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

        request_id = "req-stream-error"
        state["current_request_id"] = request_id
        env._interception_server.intercepts[request_id] = {
            "stream": True,
            "tools": None,
            "messages": [{"role": "user", "content": "test"}],
            "chunk_queue": asyncio.Queue(),
            "response_future": asyncio.get_event_loop().create_future(),
        }

        injected_error = RuntimeError("streaming LLM call failed")

        with (
            patch.object(
                vf.MultiTurnEnv,
                "get_model_response",
                new_callable=AsyncMock,
                side_effect=injected_error,
            ),
            patch(
                "verifiers.envs.experimental.api_env.synthesize_stream",
                new_callable=AsyncMock,
            ) as mock_synth,
        ):
            with pytest.raises(RuntimeError, match="streaming LLM call failed"):
                await env.get_model_response(
                    state=state,
                    prompt=[{"role": "user", "content": "test"}],
                    client=mock_client,
                    model="test-model",
                )

            # synthesize_stream should have been called with the error
            mock_synth.assert_awaited_once()
            call_args = mock_synth.call_args
            assert call_args[0][1] is None  # response is None
            assert call_args[0][2] is injected_error


class TestApiEnvMonitorRubric:
    """Tests for ApiEnvMonitorRubric."""

    @pytest.mark.asyncio
    async def test_defaults(self):
        """Test ApiEnvMonitorRubric returns 0 for clean state."""
        from verifiers.envs.experimental.api_env import ApiEnvMonitorRubric

        rubric = ApiEnvMonitorRubric()

        state: dict = {}
        assert await rubric.agent_timeout(state) == 0.0

    @pytest.mark.asyncio
    async def test_records_issues(self):
        """Test ApiEnvMonitorRubric tracks timeout."""
        from verifiers.envs.experimental.api_env import ApiEnvMonitorRubric

        rubric = ApiEnvMonitorRubric()

        # MultiTurnEnv.mark_timed_out writes state["timed_out"]=True when
        # asyncio.wait_for fires; the rubric mirrors that contract.
        state: dict = {"timed_out": True}
        assert await rubric.agent_timeout(state) == 1.0

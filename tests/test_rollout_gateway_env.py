from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from datasets import Dataset

import verifiers as vf
import verifiers.envs.experimental.rollout_gateway_mixin as rollout_gateway_mixin

pytestmark = [pytest.mark.integration, pytest.mark.environments]


class FakeTunnel:
    instances: list["FakeTunnel"] = []
    _next_id: int = 0

    def __init__(
        self,
        local_port: int,
        local_addr: str = "127.0.0.1",
        log_level: str | None = None,
    ):
        self.local_port = local_port
        self.local_addr = local_addr
        self.log_level = log_level
        self.url: str | None = None
        FakeTunnel._next_id += 1
        self.tunnel_id: str = f"fake-tunnel-{FakeTunnel._next_id}"
        self._is_running: bool = True
        self._recent_output: list[str] = ["frpc log line 1", "frpc log line 2"]
        self.start_calls = 0
        self.stop_calls = 0
        FakeTunnel.instances.append(self)

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def recent_output(self) -> list[str]:
        return list(self._recent_output)

    async def start(self) -> str:
        self.start_calls += 1
        self._is_running = True
        self.url = "https://unit-test.tunnel.prime.ai"
        return self.url

    async def stop(self) -> None:
        self.stop_calls += 1
        self._is_running = False

    def sync_stop(self) -> None:
        self.stop_calls += 1
        self._is_running = False


class GatewayCliAgentEnv(vf.RolloutGatewayMixin, vf.CliAgentEnv):
    def __init__(self, *, gateway_port=8000, use_gateway=True, **kwargs):
        self.use_gateway = use_gateway
        super().__init__(**kwargs)
        if use_gateway:
            self.init_gateway(
                gateway_port=gateway_port, timeout_seconds=self.timeout_seconds
            )

    async def post_rollout(self, state: vf.State):
        state["reward"] = 1.0
        state["test_output"] = "ok"


def _build_gateway_transport(tracker: dict) -> httpx.MockTransport:
    trajectory = [
        {
            "prompt": [{"role": "user", "content": "Hello"}],
            "completion": [{"role": "assistant", "content": "reply-1"}],
            "tokens": {
                "prompt_ids": [1, 2],
                "prompt_mask": [0, 0],
                "completion_ids": [3],
                "completion_mask": [1],
                "completion_logprobs": [-0.1],
                "overlong_prompt": False,
                "is_truncated": False,
            },
            "reward": None,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": "traj-1",
            "extras": {},
        },
        {
            "prompt": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "reply-1"},
                {"role": "user", "content": "Turn 2"},
            ],
            "completion": [{"role": "assistant", "content": "reply-2"}],
            "tokens": {
                "prompt_ids": [1, 2, 3, 4],
                "prompt_mask": [0, 0, 0, 0],
                "completion_ids": [5],
                "completion_mask": [1],
                "completion_logprobs": [-0.2],
                "overlong_prompt": False,
                "is_truncated": False,
            },
            "reward": None,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": "traj-1",
            "extras": {},
        },
        {
            "prompt": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "reply-1"},
                {"role": "user", "content": "Turn 2"},
                {"role": "assistant", "content": "reply-2"},
                {"role": "user", "content": "Turn 3"},
            ],
            "completion": [{"role": "assistant", "content": "reply-3"}],
            "tokens": {
                "prompt_ids": [1, 2, 3, 4, 5, 6],
                "prompt_mask": [0, 0, 0, 0, 0, 0],
                "completion_ids": [7],
                "completion_mask": [1],
                "completion_logprobs": [-0.3],
                "overlong_prompt": False,
                "is_truncated": False,
            },
            "reward": None,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": "traj-1",
            "extras": {},
        },
    ]

    def _handler(request: httpx.Request) -> httpx.Response:
        tracker["hosts"].add(request.url.host)
        tracker["paths"].append(request.url.path)
        path = request.url.path

        if request.method == "POST" and path.endswith("/register"):
            payload = json.loads(request.content.decode("utf-8"))
            tracker["register_payload"] = payload
            tracker["rollout_id"] = path.split("/")[-2]
            return httpx.Response(status_code=200, json={"status": "active"})

        if request.method == "POST" and path.endswith("/unregister"):
            tracker["unregister_calls"] += 1
            return httpx.Response(status_code=200, json={"status": "active"})

        if request.method == "GET" and path.endswith("/trajectory"):
            tracker["trajectory_calls"] += 1
            return httpx.Response(
                status_code=200,
                json={
                    "rollout_id": tracker["rollout_id"],
                    "status": "completed",
                    "num_turns": 3,
                    "model": "Qwen/Qwen3-0.6B",
                    "prompt": trajectory[0]["prompt"],
                    "completion": [
                        {"role": "assistant", "content": "reply-1"},
                        {"role": "user", "content": "Turn 2"},
                        {"role": "assistant", "content": "reply-2"},
                        {"role": "user", "content": "Turn 3"},
                        {"role": "assistant", "content": "reply-3"},
                    ],
                    "is_truncated": False,
                    "trajectory": trajectory,
                },
            )

        return httpx.Response(status_code=404, json={"error": f"Unhandled path {path}"})

    return httpx.MockTransport(_handler)


@pytest.mark.asyncio
async def test_cli_agent_env_rollout_uses_gateway_and_tunnel(monkeypatch):
    FakeTunnel.instances.clear()
    FakeTunnel._next_id = 0
    monkeypatch.setattr(rollout_gateway_mixin, "Tunnel", FakeTunnel)

    tracker = {
        "paths": [],
        "hosts": set(),
        "register_payload": None,
        "rollout_id": None,
        "trajectory_calls": 0,
        "unregister_calls": 0,
    }
    transport = _build_gateway_transport(tracker)
    real_async_client = httpx.AsyncClient
    client = vf.OpenAIChatCompletionsClient(
        vf.ClientConfig(
            api_key_var="UNIT_TEST_API_KEY",
            api_base_url="http://gateway.internal:8000/v1/",
        )
    )

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(rollout_gateway_mixin.httpx, "AsyncClient", _client_factory)

    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": [""],
            "example_id": [0],
        }
    )
    env = GatewayCliAgentEnv(
        run_command="echo run-agent",
        dataset=dataset,
        rubric=vf.Rubric(),
        gateway_port=8000,
        max_turns=10,
        timeout_seconds=30.0,
    )

    env.sandbox_client.create = AsyncMock(return_value=SimpleNamespace(id="sb-123"))
    env.sandbox_client.wait_for_creation = AsyncMock(return_value=None)
    env.sandbox_client.start_background_job = AsyncMock(
        return_value=SimpleNamespace(id="job-1")
    )
    env.sandbox_client.get_background_job = AsyncMock(
        return_value=SimpleNamespace(
            completed=True,
            exit_code=0,
            stdout="agent ok",
            stderr="",
        )
    )
    env.sandbox_client.delete = AsyncMock(return_value=None)

    rollout_input = {
        "prompt": [{"role": "user", "content": "Hello"}],
        "answer": "",
        "example_id": 0,
        "task": "gateway-test",
    }
    state = await env.rollout(
        input=rollout_input,
        client=client,
        model="Qwen/Qwen3-0.6B",
        sampling_args={"temperature": 0.7, "max_completion_tokens": 64},
    )

    assert state.get("error") is None
    assert state["gateway_url"] == "http://gateway.internal:8000"
    assert state["tunnel_url"] == "https://unit-test.tunnel.prime.ai"
    assert state["rollout_base_url"].startswith(
        "https://unit-test.tunnel.prime.ai/v1/rollouts/"
    )
    assert len(state["trajectory"]) == 3
    assert state["prompt"] == [{"role": "user", "content": "Hello"}]
    assert state["completion"][-1]["content"] == "reply-3"
    assert state["reward"] == 1.0

    create_request = env.sandbox_client.create.await_args.args[0]
    assert (
        create_request.environment_vars["OPENAI_BASE_URL"]
        == f"https://unit-test.tunnel.prime.ai/v1/rollouts/{state['rollout_id']}"
    )
    assert create_request.environment_vars["OPENAI_MODEL"] == "Qwen/Qwen3-0.6B"

    assert tracker["register_payload"]["max_turns"] == 10
    assert tracker["register_payload"]["sampling_params"]["temperature"] == 0.7
    assert tracker["register_payload"]["sampling_params"]["max_completion_tokens"] == 64
    assert tracker["trajectory_calls"] == 1
    assert tracker["unregister_calls"] == 1
    assert tracker["hosts"] == {"gateway.internal"}

    assert len(FakeTunnel.instances) == 1
    tunnel = FakeTunnel.instances[0]
    assert tunnel.local_port == 8000
    assert tunnel.local_addr == "gateway.internal"
    assert tunnel.start_calls == 1
    assert await env.get_gateway_tunnel_url() == "https://unit-test.tunnel.prime.ai"
    assert tunnel.start_calls == 1

    await env.teardown_gateway()
    assert tunnel.stop_calls == 1


@pytest.mark.asyncio
async def test_cli_agent_env_maintains_tunnel_per_local_addr(monkeypatch):
    FakeTunnel.instances.clear()
    FakeTunnel._next_id = 0
    monkeypatch.setattr(rollout_gateway_mixin, "Tunnel", FakeTunnel)

    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": [""],
            "example_id": [0],
        }
    )
    env = GatewayCliAgentEnv(
        run_command="echo run-agent",
        dataset=dataset,
        rubric=vf.Rubric(),
        gateway_port=8000,
    )

    url_a = await env.get_gateway_tunnel_url(local_addr="10.20.0.58")
    url_b = await env.get_gateway_tunnel_url(local_addr="10.20.0.59")
    url_a_reuse = await env.get_gateway_tunnel_url(local_addr="10.20.0.58")

    assert url_a == "https://unit-test.tunnel.prime.ai"
    assert url_b == "https://unit-test.tunnel.prime.ai"
    assert url_a_reuse == url_a

    assert len(FakeTunnel.instances) == 2
    assert {t.local_addr for t in FakeTunnel.instances} == {"10.20.0.58", "10.20.0.59"}
    assert sum(t.start_calls for t in FakeTunnel.instances) == 2

    with pytest.raises(
        ValueError, match="local_addr is required when multiple tunnels are active"
    ):
        await env.get_gateway_tunnel_url()

    await env.teardown_gateway()
    assert sum(t.stop_calls for t in FakeTunnel.instances) == 2


@pytest.mark.asyncio
async def test_use_gateway_false_initializes_interception(monkeypatch):
    """With use_gateway=False, interception server is created and gateway is not."""
    FakeTunnel.instances.clear()
    FakeTunnel._next_id = 0
    monkeypatch.setattr(rollout_gateway_mixin, "Tunnel", FakeTunnel)

    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": [""],
            "example_id": [0],
        }
    )
    env = GatewayCliAgentEnv(
        run_command="echo run-agent",
        dataset=dataset,
        rubric=vf.Rubric(),
        use_gateway=False,
        timeout_seconds=30.0,
    )

    # Interception server should be initialized
    assert env._interception_server is not None
    assert env._tunnel is None
    assert env._tunnel_lock is not None

    # Gateway attributes should not exist (init_gateway was never called)
    assert not hasattr(env, "_http_client")
    assert not hasattr(env, "_tunnels")

    # Teardowns should be safe no-ops for the inactive path
    await env.teardown_gateway()  # early return via use_gateway=False
    await env.teardown_resources()  # stops interception (which was never started)

    assert len(FakeTunnel.instances) == 0


@pytest.mark.asyncio
async def test_dead_tunnel_recreated_on_get_gateway_tunnel_url(monkeypatch):
    """Dead tunnel is stopped and replaced when get_gateway_tunnel_url is called."""
    FakeTunnel.instances.clear()
    FakeTunnel._next_id = 0
    monkeypatch.setattr(rollout_gateway_mixin, "Tunnel", FakeTunnel)

    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": [""],
            "example_id": [0],
        }
    )
    env = GatewayCliAgentEnv(
        run_command="echo run-agent",
        dataset=dataset,
        rubric=vf.Rubric(),
        gateway_port=8000,
    )

    # Start a tunnel
    await env.get_gateway_tunnel_url(local_addr="10.0.0.1")
    assert len(FakeTunnel.instances) == 1
    original_tunnel = FakeTunnel.instances[0]
    assert original_tunnel.start_calls == 1

    # Kill the tunnel
    original_tunnel._is_running = False

    # Requesting URL again should recreate
    url2 = await env.get_gateway_tunnel_url(local_addr="10.0.0.1")
    assert url2 == "https://unit-test.tunnel.prime.ai"
    assert len(FakeTunnel.instances) == 2
    assert original_tunnel.stop_calls == 1
    new_tunnel = FakeTunnel.instances[1]
    assert new_tunnel.start_calls == 1
    assert new_tunnel._is_running is True

    await env.teardown_gateway()


@pytest.mark.asyncio
async def test_poll_job_completion_raises_tunnel_error_on_dead_tunnel(monkeypatch):
    """poll_job_completion raises TunnelError when the tunnel dies mid-rollout."""
    FakeTunnel.instances.clear()
    FakeTunnel._next_id = 0
    monkeypatch.setattr(rollout_gateway_mixin, "Tunnel", FakeTunnel)

    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": [""],
            "example_id": [0],
        }
    )
    env = GatewayCliAgentEnv(
        run_command="echo run-agent",
        dataset=dataset,
        rubric=vf.Rubric(),
        gateway_port=8000,
    )

    # Start a tunnel
    await env.get_gateway_tunnel_url(local_addr="10.0.0.1")
    tunnel = FakeTunnel.instances[0]

    # Mock sandbox_client.get_background_job to never complete
    env.sandbox_client = SimpleNamespace(
        get_background_job=AsyncMock(return_value=SimpleNamespace(completed=False)),
    )

    state = {
        "rollout_id": "rollout_test123",
        "tunnel_local_addr": "10.0.0.1",
    }
    background_job = SimpleNamespace(id="job-1")

    # Kill tunnel after a brief delay
    async def kill_tunnel():
        await asyncio.sleep(0.05)
        tunnel._is_running = False

    asyncio.create_task(kill_tunnel())

    with pytest.raises(vf.TunnelError, match="Tunnel process died"):
        await env.poll_job_completion(state, "sb-123", background_job)

    await env.teardown_gateway()


@pytest.mark.asyncio
async def test_health_monitor_restarts_dead_tunnels(monkeypatch):
    """Background health monitor detects and restarts dead tunnels."""
    FakeTunnel.instances.clear()
    FakeTunnel._next_id = 0
    monkeypatch.setattr(rollout_gateway_mixin, "Tunnel", FakeTunnel)

    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": [""],
            "example_id": [0],
        }
    )
    env = GatewayCliAgentEnv(
        run_command="echo run-agent",
        dataset=dataset,
        rubric=vf.Rubric(),
        gateway_port=8000,
    )

    # Start a tunnel (this also starts the health monitor)
    await env.get_gateway_tunnel_url(local_addr="10.0.0.1")
    assert env._tunnel_monitor_task is not None
    assert not env._tunnel_monitor_task.done()

    original_tunnel = FakeTunnel.instances[0]
    original_tunnel._is_running = False

    # Run the health monitor with a short interval
    # Cancel the default one and start one with a short interval
    env._tunnel_monitor_task.cancel()
    try:
        await env._tunnel_monitor_task
    except asyncio.CancelledError:
        pass

    env._tunnel_monitor_task = asyncio.create_task(
        env._tunnel_health_monitor(interval=0.05)
    )

    # Wait for the monitor to detect and restart
    await asyncio.sleep(0.2)

    assert len(FakeTunnel.instances) == 2
    assert original_tunnel.stop_calls == 1
    new_tunnel = FakeTunnel.instances[1]
    assert new_tunnel.start_calls == 1
    assert new_tunnel._is_running is True

    await env.teardown_gateway()


@pytest.mark.asyncio
async def test_teardown_gateway_cancels_health_monitor(monkeypatch):
    """teardown_gateway cancels the health monitor task."""
    FakeTunnel.instances.clear()
    FakeTunnel._next_id = 0
    monkeypatch.setattr(rollout_gateway_mixin, "Tunnel", FakeTunnel)

    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": [""],
            "example_id": [0],
        }
    )
    env = GatewayCliAgentEnv(
        run_command="echo run-agent",
        dataset=dataset,
        rubric=vf.Rubric(),
        gateway_port=8000,
    )

    # Start a tunnel to create the health monitor
    await env.get_gateway_tunnel_url(local_addr="10.0.0.1")
    monitor_task = env._tunnel_monitor_task
    assert monitor_task is not None
    assert not monitor_task.done()

    await env.teardown_gateway()

    assert monitor_task.done()
    assert env._tunnel_monitor_task is None

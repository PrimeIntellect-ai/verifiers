from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from datasets import Dataset

import verifiers as vf
import verifiers.envs.experimental.cli_agent_env as cli_agent_env

pytestmark = [pytest.mark.integration, pytest.mark.environments]


class FakeTunnel:
    instances: list["FakeTunnel"] = []

    def __init__(self, local_port: int, log_level: str | None = None):
        self.local_port = local_port
        self.log_level = log_level
        self.url: str | None = None
        self.start_calls = 0
        self.stop_calls = 0
        FakeTunnel.instances.append(self)

    async def start(self) -> str:
        self.start_calls += 1
        self.url = "https://unit-test.tunnel.prime.ai"
        return self.url

    def sync_stop(self) -> None:
        self.stop_calls += 1


class GatewayCliAgentEnv(vf.CliAgentEnv):
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
    monkeypatch.setattr(cli_agent_env, "Tunnel", FakeTunnel)

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

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(cli_agent_env.httpx, "AsyncClient", _client_factory)

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
    client = cast(Any, SimpleNamespace(base_url="http://gateway.internal:8000/v1/"))
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
    assert tunnel.start_calls == 1
    assert await env.get_tunnel_url() == "https://unit-test.tunnel.prime.ai"
    assert tunnel.start_calls == 1

    await env.teardown_resources()
    assert tunnel.stop_calls == 1

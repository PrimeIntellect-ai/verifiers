from typing import Any

import pytest

import verifiers as vf
from verifiers.envs.integrations import openenv_env
from verifiers.types import UserMessage


class StepResult:
    def __init__(
        self, observation: dict[str, object], reward: float | None, done: bool
    ):
        self.observation = observation
        self.reward = reward
        self.done = done


class FakeGenericEnvClient:
    instances: list["FakeGenericEnvClient"] = []

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.connected = False
        self.closed = False
        self.reset_seeds: list[int] = []
        self.step_actions: list[dict[str, object]] = []
        FakeGenericEnvClient.instances.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def reset(self, *, seed: int) -> StepResult:
        self.reset_seeds.append(seed)
        return StepResult({"prompt": f"seed-{seed}"}, reward=None, done=False)

    async def step(self, action: dict[str, object]) -> StepResult:
        self.step_actions.append(action)
        return StepResult({"prompt": "next"}, reward=1.0, done=True)

    async def close(self) -> None:
        self.closed = True


class FakeMCPToolClient:
    instances: list["FakeMCPToolClient"] = []

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.connected = False
        self.closed = False
        self.step_actions: list[Any] = []
        FakeMCPToolClient.instances.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def reset(self, *, seed: int) -> StepResult:
        return StepResult({"prompt": f"mcp-{seed}"}, reward=None, done=False)

    async def list_tools(self) -> list[dict[str, object]]:
        return [
            {
                "name": "echo",
                "description": "Echo a message",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    async def step(self, action: Any) -> StepResult:
        self.step_actions.append(action)
        return StepResult({"result": {"data": "ok"}}, reward=1.0, done=True)

    async def close(self) -> None:
        self.closed = True


class FakeCallToolAction:
    def __init__(self, tool_name: str, arguments: dict[str, object]):
        self.tool_name = tool_name
        self.arguments = arguments


def render_prompt(observation: Any, **kwargs: Any):
    assert isinstance(observation, dict)
    return [UserMessage(content=str(observation["prompt"]))]


async def test_openenv_uses_public_async_generic_client(monkeypatch, tmp_path):
    FakeGenericEnvClient.instances.clear()
    monkeypatch.setattr(openenv_env, "GenericEnvClient", FakeGenericEnvClient)
    env = vf.OpenEnvEnv(
        openenv_project=tmp_path,
        num_train_examples=1,
        num_eval_examples=0,
        prompt_renderer=render_prompt,
    )

    async def launch_image_server(
        image: str, port: int, start_command: str, contract: str
    ):
        assert (image, port, start_command, contract) == (
            "image",
            8000,
            "run",
            "gym",
        )
        return openenv_env.OpenEnvServer(
            sandbox_id="sandbox",
            exposure_id="exposure",
            base_url="http://localhost:8000",
            port=8000,
            contract="gym",
        )

    async def fetch_schema(base_url: str) -> dict[str, object]:
        assert base_url == "http://localhost:8000"
        return {"action": {"type": "object", "properties": {}}}

    async def cleanup_server(
        server: openenv_env.OpenEnvServer,
        *,
        surface_delete_errors: bool = False,
    ) -> None:
        _ = surface_delete_errors
        env._active_servers.pop(server.sandbox_id, None)

    monkeypatch.setattr(
        env,
        "_resolve_runtime_config",
        lambda project_path: ("image", 8000, "run", "gym"),
    )
    monkeypatch.setattr(env, "_launch_image_server", launch_image_server)
    monkeypatch.setattr(env, "_fetch_schema", fetch_schema)
    monkeypatch.setattr(env, "_cleanup_server", cleanup_server)

    state = vf.State({"info": {"seed": 7}, "trajectory": []})
    try:
        await env.setup_state(state)

        assert state["prompt"] == [UserMessage(content="seed-7")]
        assert len(FakeGenericEnvClient.instances) == 1
        client = FakeGenericEnvClient.instances[0]
        assert client.base_url == "http://localhost:8000"
        assert client.connected is True
        assert client.reset_seeds == [7]
    finally:
        await env.cleanup_openenv(state)


async def test_openenv_uses_public_async_mcp_client(monkeypatch, tmp_path):
    FakeMCPToolClient.instances.clear()
    monkeypatch.setattr(openenv_env, "MCPToolClient", FakeMCPToolClient)
    monkeypatch.setattr(openenv_env, "CallToolAction", FakeCallToolAction)
    env = vf.OpenEnvEnv(
        openenv_project=tmp_path,
        num_train_examples=1,
        num_eval_examples=0,
        prompt_renderer=render_prompt,
    )

    async def launch_image_server(
        image: str, port: int, start_command: str, contract: str
    ):
        assert (image, port, start_command, contract) == (
            "image",
            8000,
            "run",
            "mcp",
        )
        return openenv_env.OpenEnvServer(
            sandbox_id="sandbox",
            exposure_id="exposure",
            base_url="http://localhost:8000",
            port=8000,
            contract="mcp",
        )

    async def fetch_schema(base_url: str) -> dict[str, object]:
        assert base_url == "http://localhost:8000"
        return {
            "action": {
                "type": "object",
                "properties": {"type": {"enum": ["list_tools", "call_tool"]}},
            }
        }

    async def cleanup_server(
        server: openenv_env.OpenEnvServer,
        *,
        surface_delete_errors: bool = False,
    ) -> None:
        _ = surface_delete_errors
        env._active_servers.pop(server.sandbox_id, None)

    monkeypatch.setattr(
        env,
        "_resolve_runtime_config",
        lambda project_path: ("image", 8000, "run", "mcp"),
    )
    monkeypatch.setattr(env, "_launch_image_server", launch_image_server)
    monkeypatch.setattr(env, "_fetch_schema", fetch_schema)
    monkeypatch.setattr(env, "_cleanup_server", cleanup_server)

    state = vf.State({"info": {"seed": 9}, "trajectory": []})
    try:
        await env.setup_state(state)
        state["trajectory"].append({})
        tool_messages = await env._mcp_env_response(
            [
                vf.AssistantMessage(
                    content=None,
                    tool_calls=[
                        vf.ToolCall(
                            id="call-1", name="echo", arguments='{"message": "hi"}'
                        )
                    ],
                )
            ],
            state,
        )

        assert state["prompt"] == [UserMessage(content="mcp-9")]
        assert state["tool_defs"][0].name == "echo"
        assert state["trajectory"][-1]["reward"] == 1.0
        assert tool_messages == [vf.ToolMessage(content="ok", tool_call_id="call-1")]
        client = FakeMCPToolClient.instances[0]
        action = client.step_actions[0]
        assert action.tool_name == "echo"
        assert action.arguments == {"message": "hi"}
    finally:
        await env.cleanup_openenv(state)


async def test_openenv_cleanup_delete_failure_surfaces_error(monkeypatch, tmp_path):
    class FailingSandboxClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def unexpose(self, sandbox_id: str, exposure_id: str) -> None:
            assert (sandbox_id, exposure_id) == ("sandbox", "exposure")

        async def delete(self, sandbox_id: str) -> None:
            assert sandbox_id == "sandbox"
            raise RuntimeError("delete outage")

    monkeypatch.setattr(openenv_env, "AsyncSandboxClient", FailingSandboxClient)
    env = vf.OpenEnvEnv(
        openenv_project=tmp_path,
        num_train_examples=1,
        num_eval_examples=0,
        prompt_renderer=render_prompt,
    )
    server = openenv_env.OpenEnvServer(
        sandbox_id="sandbox",
        exposure_id="exposure",
        base_url="http://localhost:8000",
        port=8000,
        contract="gym",
    )
    env._active_servers[server.sandbox_id] = server
    state = vf.State({"openenv_server": server})

    with pytest.raises(vf.SandboxDeleteError, match="Failed to delete sandbox"):
        await env.cleanup_openenv(state)

    assert state["openenv_server"] is server
    assert env._active_servers[server.sandbox_id] is server
    env._active_servers.clear()

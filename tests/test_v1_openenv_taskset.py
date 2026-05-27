import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

import verifiers as vf
from tasksets import openenv


class OpenEnvStepResult:
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
        self.actions: list[dict[str, object]] = []
        FakeGenericEnvClient.instances.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def reset(self, *, seed: int) -> OpenEnvStepResult:
        self.reset_seeds.append(seed)
        return OpenEnvStepResult({"prompt": f"seed-{seed}"}, None, False)

    async def step(self, action: dict[str, object]) -> OpenEnvStepResult:
        self.actions.append(action)
        return OpenEnvStepResult({"prompt": "done"}, 1.0, True)

    async def close(self) -> None:
        self.closed = True


class FakeMCPToolClient:
    instances: list["FakeMCPToolClient"] = []

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.connected = False
        self.closed = False
        self.actions: list[object] = []
        FakeMCPToolClient.instances.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def reset(self, *, seed: int) -> OpenEnvStepResult:
        return OpenEnvStepResult({"prompt": f"mcp-{seed}"}, None, False)

    async def list_tools(self) -> list[dict[str, object]]:
        return [
            {
                "name": "echo",
                "description": "Echo a message",
                "input_schema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            }
        ]

    async def step(self, action: object) -> OpenEnvStepResult:
        self.actions.append(action)
        return OpenEnvStepResult({"result": {"data": "ok"}}, 0.5, True)

    async def close(self) -> None:
        self.closed = True


class FakeCallToolAction:
    def __init__(self, tool_name: str, arguments: dict[str, object]):
        self.tool_name = tool_name
        self.arguments = arguments


def openenv_prompt_renderer(observation: object, **kwargs: object) -> list[vf.Message]:
    del kwargs
    assert isinstance(observation, dict)
    observation_data = cast(dict[str, object], observation)
    return [vf.UserMessage(content=str(observation_data["prompt"]))]


@pytest.fixture
def fake_openenv_runtime(monkeypatch):
    FakeGenericEnvClient.instances.clear()
    FakeMCPToolClient.instances.clear()

    async def launch_server(spec: openenv.OpenEnvRuntimeSpec) -> openenv.OpenEnvServer:
        return openenv.OpenEnvServer(
            sandbox_id="sandbox",
            exposure_id="exposure",
            base_url="http://localhost:8000",
            port=spec.port,
            contract=spec.contract,
        )

    async def fetch_schema(
        base_url: str, spec: openenv.OpenEnvRuntimeSpec
    ) -> dict[str, object]:
        del base_url
        if spec.contract == "mcp":
            return {
                "action": {
                    "type": "object",
                    "properties": {"type": {"enum": ["list_tools", "call_tool"]}},
                }
            }
        return {
            "action": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }
        }

    async def cleanup_server(server: openenv.OpenEnvServer) -> None:
        del server

    def optional_type(module_name: str, attr: str) -> type[object] | None:
        if (module_name, attr) == (
            "openenv.core.generic_client",
            "GenericEnvClient",
        ):
            return FakeGenericEnvClient
        if (module_name, attr) == ("openenv.core.mcp_client", "MCPToolClient"):
            return FakeMCPToolClient
        if (module_name, attr) == (
            "openenv.core.env_server.mcp_types",
            "CallToolAction",
        ):
            return FakeCallToolAction
        return None

    monkeypatch.setattr(openenv, "launch_openenv_server", launch_server)
    monkeypatch.setattr(openenv, "fetch_openenv_schema", fetch_schema)
    monkeypatch.setattr(openenv, "cleanup_openenv_server", cleanup_server)
    monkeypatch.setattr(openenv, "openenv_type", optional_type)


def write_openenv_manifest(project: Path, contract: str) -> None:
    project.mkdir(exist_ok=True)
    (project / ".build.json").write_text(
        json.dumps(
            {
                "image": "image",
                "port": 8000,
                "start_command": "run",
                "contract": contract,
            }
        )
    )


@pytest.mark.asyncio
async def test_openenv_taskset_runs_gym_rollout_boundary(
    tmp_path, fake_openenv_runtime
):
    write_openenv_manifest(tmp_path, "gym")
    taskset = openenv.OpenEnvTaskset(
        config=openenv.OpenEnvTasksetConfig(
            openenv_project=str(tmp_path),
            prompt_renderer="tests.test_v1_openenv_taskset:openenv_prompt_renderer",
            num_train_examples=1,
            num_eval_examples=0,
            seed=7,
        )
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness())
    task = next(iter(taskset))
    state = vf.State.for_task(task)

    await env.harness.setup_state(task, state)
    await env.harness.runtime.setup_rollout(task, state)

    assert state["prompt"] == [vf.UserMessage(content="seed-7")]
    client = FakeGenericEnvClient.instances[0]
    assert client.connected is True
    assert client.reset_seeds == [7]

    state["completion"] = [vf.AssistantMessage(content="advance")]
    state["trajectory"].append({"reward": None})
    messages = await env.harness.runtime.user_messages(task, state)

    assert client.actions == [{"command": "advance"}]
    assert messages == [{"role": "user", "content": "done"}]
    assert state["trajectory"][-1]["reward"] == 1.0
    assert state["openenv_done"] is True

    await env.harness.runtime.cleanup_rollout(task, state)
    assert client.closed is True


@pytest.mark.asyncio
async def test_openenv_taskset_exposes_mcp_tools(tmp_path, fake_openenv_runtime):
    write_openenv_manifest(tmp_path, "mcp")
    taskset = openenv.OpenEnvTaskset(
        config=openenv.OpenEnvTasksetConfig(
            openenv_project=str(tmp_path),
            prompt_renderer="tests.test_v1_openenv_taskset:openenv_prompt_renderer",
            num_train_examples=1,
            num_eval_examples=0,
            seed=9,
        )
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness())
    task = next(iter(taskset))
    state = vf.State.for_task(task)

    await env.harness.setup_state(task, state)
    await env.harness.runtime.setup_rollout(task, state)

    assert state["prompt"] == [vf.UserMessage(content="mcp-9")]
    assert state["tools"] == ["echo"]
    client = FakeMCPToolClient.instances[0]
    state["trajectory"].append({"reward": None})
    tool = cast(
        Callable[..., Awaitable[object]],
        env.harness.runtime.tool_calls(task, state)["echo"],
    )
    result = await tool(message="hello")

    action = cast(FakeCallToolAction, client.actions[0])
    assert action.tool_name == "echo"
    assert action.arguments == {"message": "hello"}
    assert result == "ok"
    assert state["trajectory"][-1]["reward"] == 0.5
    assert state["openenv_done"] is True

    await env.harness.runtime.cleanup_rollout(task, state)
    assert client.closed is True

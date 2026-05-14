from typing import Any

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


async def test_openenv_uses_public_async_generic_client(monkeypatch):
    FakeGenericEnvClient.instances.clear()
    monkeypatch.setattr(openenv_env, "GenericEnvClient", FakeGenericEnvClient)
    env = vf.OpenEnvEnv(
        num_train_examples=1,
        num_eval_examples=0,
        prompt_renderer=render_prompt,
    )

    async def create_server():
        return openenv_env.OpenEnvServer(
            sandbox_id="sandbox",
            exposure_id="exposure",
            base_url="http://localhost:8000",
            port=8000,
            contract="gym",
        )

    async def fetch_action_schema(base_url: str) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    monkeypatch.setattr(env, "_create_server", create_server)
    monkeypatch.setattr(env, "_fetch_action_schema", fetch_action_schema)

    state = vf.State({"info": {"seed": 7}, "trajectory": []})
    await env.setup_state(state)

    assert state["prompt"] == [UserMessage(content="seed-7")]
    assert len(FakeGenericEnvClient.instances) == 1
    client = FakeGenericEnvClient.instances[0]
    assert client.base_url == "http://localhost:8000"
    assert client.connected is True
    assert client.reset_seeds == [7]


async def test_openenv_uses_public_async_mcp_client(monkeypatch):
    FakeMCPToolClient.instances.clear()
    monkeypatch.setattr(openenv_env, "MCPToolClient", FakeMCPToolClient)
    monkeypatch.setattr(openenv_env, "CallToolAction", FakeCallToolAction)
    env = vf.OpenEnvEnv(
        num_train_examples=1,
        num_eval_examples=0,
        prompt_renderer=render_prompt,
    )

    async def create_server():
        return openenv_env.OpenEnvServer(
            sandbox_id="sandbox",
            exposure_id="exposure",
            base_url="http://localhost:8000",
            port=8000,
            contract="mcp",
        )

    async def fetch_action_schema(base_url: str) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"type": {"enum": ["list_tools", "call_tool"]}},
        }

    monkeypatch.setattr(env, "_create_server", create_server)
    monkeypatch.setattr(env, "_fetch_action_schema", fetch_action_schema)

    state = vf.State({"info": {"seed": 9}, "trajectory": []})
    await env.setup_state(state)
    result = await env._mcp_step_tool(
        state["openenv_mcp_client"], "echo", {"message": "hi"}
    )

    assert state["prompt"] == [UserMessage(content="mcp-9")]
    assert state["tool_defs"][0].name == "echo"
    assert result.reward == 1.0
    client = FakeMCPToolClient.instances[0]
    action = client.step_actions[0]
    assert action.tool_name == "echo"
    assert action.arguments == {"message": "hi"}

import io
import json
from unittest.mock import AsyncMock

import pytest
from mcp.server.fastmcp import FastMCP

import verifiers.v1 as vf
from tasksets.openenv_v1 import (
    OpenEnvConfig,
    OpenEnvState,
    OpenEnvTask,
    OpenEnvTaskset,
)
from tasksets.openenv_v1.servers.toolset import OpenEnvToolset
from tasksets.openenv_v1.servers.user import OpenEnvUser


class FakeSocket:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive_json(self):
        return self.responses.pop(0)

    async def close(self):
        self.closed = True


class FakeSession:
    sockets = []

    def __init__(self):
        self.socket = self.sockets.pop(0)
        self.closed = False

    async def ws_connect(self, url, **kwargs):
        assert url == "http://127.0.0.1:8001/ws"
        return self.socket

    async def close(self):
        self.closed = True


def write_manifest(path, contract: str = "gym") -> None:
    (path / ".build.json").write_text(
        json.dumps(
            {
                "image": "openenv:test",
                "port": 8001,
                "start_command": "run-openenv",
                "contract": contract,
            }
        )
    )


def task(contract: str, seed: int = 3) -> OpenEnvTask:
    return OpenEnvTask(
        idx=0,
        instruction=None if contract == "gym" else "Use a tool.",
        image="image",
        contract=contract,
        port=8001,
        start_command="run",
        seed=seed,
    )


def test_openenv_tasks_are_image_backed_seeded_episodes(tmp_path):
    write_manifest(tmp_path, "mcp")
    taskset = OpenEnvTaskset(
        OpenEnvConfig(
            project=str(tmp_path), num_tasks=2, seed=7, instruction="Use a tool."
        )
    )

    tasks = taskset.load_tasks()

    assert [task.seed for task in tasks] == [7, 8]
    assert tasks[0].image == "openenv:test"
    assert tasks[0].instruction == "Use a tool."
    assert tasks[0].workdir == "/app/env"
    assert taskset.tools(tasks[0])[0].config.colocated is True


@pytest.mark.asyncio
async def test_openenv_setup_only_starts_the_server(tmp_path):
    write_manifest(tmp_path)
    taskset = OpenEnvTaskset(OpenEnvConfig(project=str(tmp_path), num_tasks=1))
    runtime = AsyncMock()

    await taskset.setup(taskset.load_tasks()[0], runtime)

    runtime.run_background.assert_awaited_once_with(
        ["sh", "-lc", "run-openenv"],
        {"ENABLE_WEB_INTERFACE": "false"},
        "openenv.log",
    )
    runtime.run.assert_not_called()


@pytest.mark.asyncio
async def test_gym_user_drives_openenv_directly(monkeypatch):
    socket = FakeSocket(
        [
            {
                "type": "observation",
                "data": {
                    "observation": {"prompt": "seed-3"},
                    "reward": None,
                    "done": False,
                },
            },
            {
                "type": "observation",
                "data": {
                    "observation": {"messages": [{"content": "feedback"}]},
                    "reward": 1.0,
                    "done": True,
                },
            },
        ]
    )
    FakeSession.sockets = [socket]
    monkeypatch.setattr(
        "tasksets.openenv_v1.servers.user.urlopen",
        lambda *args, **kwargs: io.BytesIO(
            json.dumps(
                {
                    "action": {
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    }
                }
            ).encode()
        ),
    )
    monkeypatch.setattr(
        "tasksets.openenv_v1.servers.user.aiohttp.ClientSession", FakeSession
    )
    user = OpenEnvUser(vf.UserConfig(colocated=True))
    await user.setup_task(task("gym"))

    assert await user.respond("") == [{"role": "user", "content": "seed-3"}]
    assert await user.respond("[crane]") == [{"role": "user", "content": "feedback"}]
    assert socket.sent[-1]["data"] == {"message": "[crane]"}
    assert user.state == OpenEnvState(reward=1.0, done=True)


@pytest.mark.asyncio
async def test_mcp_toolset_bridges_openenv_tools(monkeypatch):
    discovery = FakeSocket(
        [
            {
                "type": "observation",
                "data": {
                    "observation": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo text.",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {"message": {"type": "string"}},
                                    "required": ["message"],
                                },
                            }
                        ]
                    }
                },
            }
        ]
    )
    calls = FakeSocket(
        [
            {
                "type": "observation",
                "data": {"observation": {}, "reward": 0.0, "done": False},
            },
            {
                "type": "observation",
                "data": {
                    "observation": {
                        "result": {"data": "hello"},
                        "error": None,
                    },
                    "reward": 0.5,
                    "done": True,
                },
            },
        ]
    )
    FakeSession.sockets = [discovery, calls]
    monkeypatch.setattr(
        "tasksets.openenv_v1.servers.toolset.aiohttp.ClientSession", FakeSession
    )
    toolset = OpenEnvToolset(vf.ToolsetConfig(colocated=True))
    await toolset.setup_task(task("mcp", seed=4))

    mcp = FastMCP("test")
    toolset._register(mcp)
    registered = await mcp.list_tools()
    assert registered[0].inputSchema["required"] == ["message"]
    assert await toolset.call_tool("echo", {"message": "hello"}) == "hello"
    assert calls.sent[-1]["data"]["tool_name"] == "echo"
    assert toolset.state == OpenEnvState(reward=0.5, done=True)


@pytest.mark.asyncio
async def test_only_gym_done_stops_the_rollout(tmp_path):
    write_manifest(tmp_path)
    taskset = OpenEnvTaskset(OpenEnvConfig(project=str(tmp_path), num_tasks=1))
    gym_trace = vf.Trace(task=taskset.load_tasks()[0], state=OpenEnvState(done=True))

    assert await taskset.openenv_done(gym_trace) is True
    assert await taskset.openenv_reward(gym_trace) == 0.0

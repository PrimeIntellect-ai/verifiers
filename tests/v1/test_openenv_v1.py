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
from tasksets.openenv_v1.taskset import OpenEnvUser


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


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def json(self):
        return self.payload


class FakeSession:
    sockets = []
    responses = []
    urls = []

    def __init__(self, **kwargs):
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def ws_connect(self, url, **kwargs):
        self.urls.append(url)
        return self.sockets.pop(0)

    def get(self, url):
        self.urls.append(url)
        return FakeResponse(self.responses.pop(0))

    async def close(self):
        self.closed = True


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeHTTPClient:
    responses = []
    requests = []

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, json):
        self.requests.append((url, json))
        return FakeHTTPResponse(self.responses.pop(0))


def task(seed: int = 3) -> OpenEnvTask:
    return OpenEnvTask(
        idx=0,
        prompt=None,
        image="image",
        seed=seed,
    )


def test_openenv_tasks_are_image_backed_seeded_episodes():
    taskset = OpenEnvTaskset(
        OpenEnvConfig(
            image="openenv:test",
            contract="mcp",
            num_tasks=2,
            seed=7,
            prompt="Use a tool.",
        )
    )

    tasks = taskset.load_tasks()

    assert [task.seed for task in tasks] == [7, 8]
    assert tasks[0].image == "openenv:test"
    assert tasks[0].prompt == "Use a tool."
    assert tasks[0].workdir == "/app/env"
    toolset = taskset.tools(tasks[0])[0]
    assert isinstance(toolset, vf.JSONRPCToolset)
    assert toolset.config.colocated is True
    assert toolset.config.endpoint == "http://openenv/mcp"
    assert toolset.config.uds == "/tmp/openenv.sock"


@pytest.mark.asyncio
async def test_openenv_setup_only_starts_the_server():
    taskset = OpenEnvTaskset(
        OpenEnvConfig(image="openenv:test", contract="gym", num_tasks=1)
    )
    runtime = AsyncMock()

    await taskset.setup(taskset.load_tasks()[0], runtime)

    runtime.run_background.assert_awaited_once_with(
        [
            "/app/.venv/bin/uvicorn",
            "server.app:app",
            "--uds",
            "/tmp/openenv.sock",
        ],
        {"ENABLE_WEB_INTERFACE": "false"},
        "openenv.log",
    )
    runtime.run.assert_not_called()


@pytest.mark.asyncio
async def test_gym_user_drives_openenv_directly(monkeypatch, tmp_path):
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
    FakeSession.responses = [
        {
            "action": {
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            }
        }
    ]
    FakeSession.urls = []
    monkeypatch.setattr(
        "tasksets.openenv_v1.taskset.aiohttp.ClientSession", FakeSession
    )
    monkeypatch.setattr(
        "tasksets.openenv_v1.taskset.aiohttp.UnixConnector", lambda **kwargs: None
    )
    socket_path = tmp_path / "openenv.sock"
    socket_path.touch()
    monkeypatch.setattr("tasksets.openenv_v1.taskset.OPENENV_SOCKET", str(socket_path))
    user = OpenEnvUser(vf.UserConfig(colocated=True))
    await user.setup_task(task())

    assert await user.respond("") == [{"role": "user", "content": "seed-3"}]
    assert await user.respond("[crane]") == [{"role": "user", "content": "feedback"}]
    assert socket.sent[-1]["data"] == {"message": "[crane]"}
    assert user.state == OpenEnvState(reward=1.0, done=True)
    assert FakeSession.urls == [
        "http://openenv/schema",
        "http://openenv/ws",
    ]


@pytest.mark.asyncio
async def test_mcp_toolset_bridges_openenv_tools(monkeypatch, tmp_path):
    FakeHTTPClient.responses = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo text.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                        },
                    }
                ]
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"data": "hello"},
        },
    ]
    FakeHTTPClient.requests = []
    monkeypatch.setattr("verifiers.v1.mcp.toolset.httpx.AsyncClient", FakeHTTPClient)
    socket_path = tmp_path / "openenv.sock"
    socket_path.touch()
    toolset = vf.JSONRPCToolset(
        vf.JSONRPCToolsetConfig(
            colocated=True,
            endpoint="http://openenv/mcp",
            uds=str(socket_path),
        )
    )
    await toolset.setup()

    mcp = FastMCP("test")
    toolset._register(mcp)
    registered = await mcp.list_tools()
    assert registered[0].inputSchema["required"] == ["message"]
    result = await mcp.call_tool("echo", {"message": "hello"})
    assert "hello" in result[0].text
    assert FakeHTTPClient.requests[-1][1]["params"] == {
        "name": "echo",
        "arguments": {"message": "hello"},
    }


@pytest.mark.asyncio
async def test_task_agnostic_jsonrpc_toolset_skips_task_loading():
    toolset = vf.JSONRPCToolset(vf.JSONRPCToolsetConfig(endpoint="http://openenv/mcp"))
    toolset._fetch_task = AsyncMock()

    await toolset._setup_task_from_channel("http://state", "secret")

    toolset._fetch_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_openenv_state_controls_stop_and_reward():
    taskset = OpenEnvTaskset(
        OpenEnvConfig(image="openenv:test", contract="gym", num_tasks=1)
    )
    trace = vf.Trace(
        task=taskset.load_tasks()[0], state=OpenEnvState(reward=0.5, done=True)
    )

    assert await taskset.openenv_done(trace) is True
    assert await taskset.openenv_reward(trace) == 0.5

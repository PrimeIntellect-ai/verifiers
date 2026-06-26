from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

import verifiers.v1 as vf
from verifiers.v1.env import EnvConfig, Environment
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime
from verifiers.v1.tasksets.openenv import OpenEnvConfig, OpenEnvTaskset
from verifiers.v1.tasksets.openenv.taskset import (
    OPENENV_SOCKET,
    OPENENV_URL,
)

ECHO_EXAMPLE = Path(__file__).parents[2] / "environments" / "openenv_echo_v1"


@pytest.mark.asyncio
async def test_uv_script_ignores_image_system_python() -> None:
    runtime = SubprocessRuntime(SubprocessConfig())
    runtime.write = AsyncMock()
    runtime.run = AsyncMock(
        side_effect=[
            ProgramResult(exit_code=0, stdout="", stderr=""),
            ProgramResult(
                exit_code=0, stdout="/tmp/openenv-test/bin/python\n", stderr=""
            ),
        ]
    )

    await runtime.prepare_uv_script("# openenv image system-python regression\n")

    prepare_command = runtime.run.await_args_list[1].args[0][2]
    assert "; unset UV_SYSTEM_PYTHON; uv sync" in prepare_command


@pytest.mark.parametrize("harness_id", [None, "rlm", "codex"])
def test_openenv_uses_an_external_mcp_harness(harness_id: str | None) -> None:
    harness: dict[str, object] = {"runtime": {"type": "docker"}}
    if harness_id:
        harness["id"] = harness_id
    config = EnvConfig.model_validate(
        {
            "taskset": {
                "id": "openenv",
                "image": "openenv:test",
                "prompt": "Use the available tool.",
                "resources": {"cpu": 2, "memory": 4, "disk": 10},
            },
            "harness": harness,
        }
    )
    env = Environment(config)
    task = env.taskset.load_tasks()[0]
    toolset = env.taskset.tools(task)[0]

    assert env.harness.config.id == (harness_id or "default")
    assert OpenEnvConfig.model_fields["image"].is_required()
    assert OpenEnvConfig.model_fields["prompt"].is_required()
    assert task.name == "openenv"
    assert task.image == "openenv:test"
    assert task.workdir == "/app/env"
    assert task.resources.cpu == 2
    assert task.resources.memory == 4
    assert task.resources.disk == 10
    assert isinstance(toolset, vf.JSONRPCToolset)
    assert toolset.config.colocated is True
    assert toolset.config.endpoint == f"{OPENENV_URL}/mcp"
    assert toolset.config.uds == OPENENV_SOCKET


def test_echo_image_is_owned_by_the_example(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ECHO_EXAMPLE))
    from openenv_echo_v1.taskset import (
        ECHO_IMAGE,
        OpenEnvEchoConfig,
        OpenEnvEchoTaskset,
    )

    taskset = OpenEnvEchoTaskset(OpenEnvEchoConfig(id="openenv-echo-v1"))
    task = taskset.load_tasks()[0]

    assert task.name == "openenv-echo-v1"
    assert task.image == ECHO_IMAGE
    assert task.prompt == (
        'Call the echo_message tool with the message "Hello, World!", then return '
        "the echoed text."
    )
    assert task.resources.cpu == 2
    assert task.resources.memory == 4
    assert task.resources.disk == 10


@pytest.mark.asyncio
async def test_openenv_setup_only_starts_its_server() -> None:
    taskset = OpenEnvTaskset(
        OpenEnvConfig(id="openenv", image="openenv:test", prompt="Use the tool.")
    )
    runtime = AsyncMock()

    await taskset.setup(taskset.load_tasks()[0], runtime)

    runtime.run_background.assert_awaited_once_with(
        [
            "/app/.venv/bin/uvicorn",
            "server.app:app",
            "--uds",
            OPENENV_SOCKET,
        ],
        {"ENABLE_WEB_INTERFACE": "false"},
        "/tmp/openenv.log",
    )


@pytest.mark.asyncio
async def test_codex_harness_receives_openenv_mcp_server() -> None:
    from verifiers.v1.harnesses.codex import CodexHarness, CodexHarnessConfig

    runtime = Mock()
    runtime.run_program = AsyncMock(
        return_value=ProgramResult(exit_code=0, stdout="", stderr="")
    )
    harness = CodexHarness(CodexHarnessConfig(id="codex"))

    await harness.launch(
        vf.RolloutContext(
            model="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
            client=Mock(),
            sampling=vf.SamplingConfig(),
        ),
        vf.Trace(task=vf.Task(idx=0, prompt="Use the Echo tool.")),
        runtime,
        endpoint="http://127.0.0.1:9000/v1",
        secret="secret",
        mcp_urls={"jsonrpc": "http://127.0.0.1:12345/mcp"},
    )

    argv, _ = runtime.run_program.await_args.args
    assert CodexHarness.SUPPORTS_MCP is True
    assert harness.config.version == "0.116.0"
    assert 'mcp_servers.jsonrpc.url="http://127.0.0.1:12345/mcp"' in argv


@pytest.mark.asyncio
async def test_jsonrpc_toolset_bridges_tools_to_mcp(monkeypatch) -> None:
    responses = iter(
        [
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
            {"jsonrpc": "2.0", "id": 1, "result": {"data": "hello"}},
        ]
    )
    requests = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json):
            requests.append((url, json))
            return httpx.Response(
                200,
                json=next(responses),
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("verifiers.v1.mcp.toolset.httpx.AsyncClient", Client)
    toolset = vf.JSONRPCToolset(vf.JSONRPCToolsetConfig(endpoint="http://openenv/mcp"))
    await toolset.setup()

    mcp = FastMCP("test")
    toolset._register(mcp)
    registered = await mcp.list_tools()
    result = await mcp.call_tool("echo", {"message": "hello"})

    assert registered[0].inputSchema["required"] == ["message"]
    assert "hello" in result[0].text
    assert requests[-1][1]["params"] == {
        "name": "echo",
        "arguments": {"message": "hello"},
    }

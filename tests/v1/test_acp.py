"""Model-free end-to-end coverage for the ACP process boundary."""

import asyncio
import json

import pytest

from verifiers.v1.acp import ACP, _packet, _PacketReader, _runner_program
from verifiers.v1.runtimes import DockerConfig, Runtime, provision_runtime

_PYTHON = "/usr/local/bin/python3"
_SNAPSHOT_SUPPORT = r"""
import json
import os
import subprocess
import sys

def snapshot():
    names = (
        "PATH",
        "VIRTUAL_ENV",
        "UV_INSTALL_DIR",
        "UV_RUN_RECURSION_DEPTH",
        "ACP_ENV_SENTINEL",
    )
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    )
    return {
        "present": {name: name in os.environ for name in names},
        "values": {name: os.environ.get(name) for name in names},
        "markers": sorted(
            name for name in os.environ if name.startswith("_VF_ACP_ORIGINAL_")
        ),
        "executable": sys.executable,
        "pip_returncode": pip.returncode,
    }
"""
_SNAPSHOT_SOURCE = _SNAPSHOT_SUPPORT + "\nprint(json.dumps(snapshot()))"
_AGENT_SOURCE = (
    _SNAPSHOT_SUPPORT
    + r"""


def send(value):
    print(json.dumps(value), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    if method == "initialize":
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "agentCapabilities": {},
        }
    elif method == "session/new":
        result = {"sessionId": "env-test"}
    elif method == "session/prompt":
        send({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "env-test",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": json.dumps(snapshot())},
                },
            },
        })
        result = {"stopReason": "end_turn"}
    else:
        result = {}
    send({"jsonrpc": "2.0", "id": request["id"], "result": result})
"""
)


async def _runtime_snapshot(runtime: Runtime, env: dict[str, str]) -> dict:
    result = await runtime.run([_PYTHON, "-c", _SNAPSHOT_SOURCE], env)
    assert result.exit_code == 0, result.stderr
    return json.loads(result.stdout)


def _config(command: list[str]) -> dict:
    return {
        "command": command,
        "messages": [{"role": "user", "content": "report env"}],
        "mcp_urls": {},
        "system_prompt": "",
        "session_path": None,
        "session_meta": {},
        "allow_empty_tool_reply": False,
    }


async def _stream_snapshot(
    runtime: Runtime, env: dict[str, str], command: list[str]
) -> dict:
    program = await _runner_program(runtime, env)
    process = await runtime.open_process([*program, "stream"], env)
    reader = _PacketReader(process.stdout)
    finished = False
    try:
        await process.write(
            _packet({"operation": "prompt", "config": _config(command)})
        )
        response = await reader.read()
        assert response["ok"], response
        await process.write(_packet({"operation": "shutdown"}))
        shutdown = await reader.read()
        assert shutdown == {"ok": True}
        assert await asyncio.wait_for(process.wait(), 10) == 0
        finished = True
    finally:
        if not finished:
            await process.kill()
            await process.wait()
    return json.loads(response["reply"])


@pytest.mark.docker
async def test_acp_agents_receive_the_pre_wrapper_environment() -> None:
    async with provision_runtime(DockerConfig()) as runtime:
        await runtime.write("env_agent.py", _AGENT_SOURCE.encode())
        command = [_PYTHON, "env_agent.py"]

        once_env = {
            "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "ACP_ENV_SENTINEL": "once",
        }
        once_expected = await _runtime_snapshot(runtime, once_env)
        once_result = await ACP().run(runtime, once_env, command, "report env")
        assert once_result.exit_code == 0, once_result.stderr
        once = json.loads(once_result.stdout)
        assert once == once_expected

        stream_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "VIRTUAL_ENV": "original venv with spaces",
            "UV_INSTALL_DIR": "",
            "UV_RUN_RECURSION_DEPTH": "9",
            "ACP_ENV_SENTINEL": "stream",
        }
        stream_expected = await _runtime_snapshot(runtime, stream_env)
        stream = await _stream_snapshot(runtime, stream_env, command)
        assert stream == stream_expected

        assert once["markers"] == stream["markers"] == []
        assert once["executable"] == stream["executable"] == _PYTHON
        assert once["pip_returncode"] == stream["pip_returncode"] == 0

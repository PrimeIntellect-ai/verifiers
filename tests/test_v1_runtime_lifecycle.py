from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import urllib.request
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import verifiers.v1 as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import Response, ResponseMessage, ToolCall
from verifiers.types import Tool
from verifiers.v1.runtime import Runtime
from verifiers.v1.utils import mcp_utils
from verifiers.v1.utils.mcp_proxy_utils import proxy_source
from verifiers.v1.utils.mcp_proxy_utils import MCP_PROXY_PATH
from verifiers.v1.utils.program_utils import command_env
from verifiers.v1.utils.sandbox_program_utils import (
    TOOL_DEFS_BY_PROTOCOL_PATH,
    TOOL_DEFS_PATH,
    sandbox_runner_program,
)


class FakeMCPHandle:
    def __init__(self, name: str):
        self.name = name
        self.tool_def = Tool(
            name=name,
            description="fake",
            parameters={"type": "object", "properties": {}},
        )

    async def __call__(self) -> str:
        return "ok"


class FakeClient:
    def __init__(self):
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeModelClient:
    def __init__(self, responses: list[Response]):
        self.responses = responses

    async def get_response(self, **kwargs: object) -> Response:
        _ = kwargs
        if not self.responses:
            raise AssertionError("No fake model responses left.")
        return self.responses.pop(0)


class FakeCreateSandboxRequest:
    def __init__(self, **kwargs: object):
        self.kwargs = kwargs


class FakeSandboxResult:
    def __init__(self, sandbox_id: str):
        self.id = sandbox_id


class FakeCommandResult:
    exit_code = 0
    stdout = "ok\n"
    stderr = ""


class FakeSandboxClient:
    created: list[str] = []
    deleted: list[str] = []
    commands: list[tuple[str, str]] = []

    @classmethod
    def reset(cls) -> None:
        cls.created = []
        cls.deleted = []
        cls.commands = []

    async def create(self, request: FakeCreateSandboxRequest) -> FakeSandboxResult:
        _ = request
        sandbox_id = f"sbx-{len(type(self).created) + 1}"
        type(self).created.append(sandbox_id)
        return FakeSandboxResult(sandbox_id)

    async def wait_for_creation(self, sandbox_id: str) -> None:
        _ = sandbox_id

    async def execute_command(
        self, *args: object, **kwargs: object
    ) -> FakeCommandResult:
        sandbox_id = str(kwargs.get("sandbox_id") or args[0])
        command = str(kwargs.get("command") or args[1])
        type(self).commands.append((sandbox_id, command))
        return FakeCommandResult()

    async def upload_bytes(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs

    async def upload_file(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs

    async def read_file(self, *args: object, **kwargs: object) -> str:
        _ = args, kwargs
        return ""

    async def delete(self, sandbox_id: str) -> None:
        type(self).deleted.append(sandbox_id)

    async def aclose(self) -> None:
        pass


async def echo_tool(query: str) -> str:
    return f"echo:{query}"


async def named_tool(name: str) -> str:
    return f"name:{name}"


async def failing_tool(section_id: str) -> str:
    _ = section_id
    raise ValueError("Invalid section_id format.")


def fake_response(
    content: str | None = None, tool_calls: list[ToolCall] | None = None
) -> Response:
    return Response(
        id="fake",
        created=0,
        model="fake",
        message=ResponseMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            is_truncated=False,
        ),
    )


async def program_sandbox_id(sandbox) -> str:
    return sandbox.id


def install_fake_sandboxes(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandboxClient.reset()
    module = SimpleNamespace(
        AsyncSandboxClient=FakeSandboxClient,
        CreateSandboxRequest=FakeCreateSandboxRequest,
    )
    monkeypatch.setitem(sys.modules, "prime_sandboxes", module)


async def endpoint_user(
    task: dict[str, object], state: dict[str, object]
) -> list[dict[str, str]]:
    _ = task
    state["user_seen"] = True
    return [{"role": "user", "content": "continue"}]


async def endpoint_program(task, state, client):
    _ = task, client
    root = state["endpoint_root_url"].rstrip("/")

    def get_json(url: str) -> dict[str, object]:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())

    def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode())

    tools = await asyncio.to_thread(get_json, f"{root}/vf/tools")
    openai_tools = await asyncio.to_thread(
        get_json, f"{root}/vf/tools?protocol=openai_chat_completions"
    )
    tool_payload: dict[str, object] = {"arguments": {"query": "hi"}}
    tool_result = await asyncio.to_thread(
        post_json,
        f"{root}/vf/tools/echo_tool",
        tool_payload,
    )
    user_payload: dict[str, object] = {
        "transcript": [{"role": "assistant", "content": "hello"}]
    }
    user_result = await asyncio.to_thread(
        post_json,
        f"{root}/vf/user",
        user_payload,
    )
    state["done"] = True
    stop_result = await asyncio.to_thread(post_json, f"{root}/vf/stop", {})
    return {
        "endpoint_tools": tools["tools"],
        "endpoint_openai_tools": openai_tools["tools"],
        "endpoint_tool_result": tool_result["result"],
        "endpoint_user_messages": user_result["messages"],
        "endpoint_stop": stop_result,
    }


async def mcp_proxy_program(task, state, client):
    _ = task, client
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(proxy_source())
        proxy_path = Path(f.name)
    env = dict(os.environ)
    env["VF_TOOL_BASE_URL"] = f"{state['endpoint_root_url'].rstrip('/')}/vf/tools"
    env["VF_TOOL_API_KEY"] = "intercepted"
    try:
        server = StdioServerParameters(
            command=sys.executable,
            args=[str(proxy_path)],
            env=env,
        )
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                result = await session.call_tool("echo_tool", {"query": "hi"})
        return {
            "mcp_tools": [tool.name for tool in listed.tools],
            "mcp_result": mcp_utils.mcp_result_value(result),
        }
    finally:
        proxy_path.unlink(missing_ok=True)


MCP_COMMAND_CLIENT = r"""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    command = json.loads(os.environ["VF_MCP_TOOL_COMMAND_JSON"])
    server = StdioServerParameters(
        command=command[0],
        args=command[1:],
        env=dict(os.environ),
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            result = await session.call_tool("echo_tool", {"query": "hi"})
    print(
        json.dumps(
            {
                "tools": [tool.name for tool in listed.tools],
                "result": result.content[0].text,
            }
        )
    )


asyncio.run(main())
"""


async def child_program(task, state):
    _ = task
    return {
        "child_runtime": dict(state["runtime"]),
        "child_trajectory_id": state["trajectory_id"],
    }


async def parent_program(task, state):
    child = vf.Harness(program=child_program)
    child_state = await state.run_harness(
        child,
        vf.Task({"prompt": [{"role": "user", "content": "child"}]}).freeze(),
    )
    return {"child_state": child_state}


async def mark_submitted(task, state):
    _ = task
    state["submitted"] = True
    return state


async def parent_calls_owned_child_program(task, state):
    child = vf.Harness(
        program=child_program, client=cast(Client, FakeClient()), model="child-model"
    )
    child_state = await state.run_harness(
        child,
        vf.Task({"prompt": [{"role": "user", "content": "child"}]}).freeze(),
    )
    return {"child_state": child_state}


async def submitted(task, state) -> bool:
    _ = task
    return bool(state.get("submitted"))


async def state_tools_program(task, state):
    _ = task
    tools = state.tools()
    state["tool_result"] = await tools["echo_tool"](query="state")
    return state


def test_model_client_default_keys_are_rollout_local() -> None:
    runtime = Runtime()
    client = FakeClient()
    state_a = vf.State.for_task(vf.Task({}).freeze())
    state_b = vf.State.for_task(vf.Task({}).freeze())

    runtime.bind_model_client(state_a, cast(Client, client))
    runtime.bind_model_client(state_b, cast(Client, client))

    assert state_a["runtime"]["client_key"] != state_b["runtime"]["client_key"]
    assert len(runtime.model_clients) == 2


@pytest.mark.asyncio
async def test_endpoint_exposes_tool_user_and_stop_surfaces() -> None:
    harness = vf.Harness(
        program=endpoint_program,
        toolsets=[vf.Toolset(tools=[echo_tool])],
        user=endpoint_user,
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)
    await harness.teardown()

    assert [tool["name"] for tool in state["endpoint_tools"]] == ["echo_tool"]
    openai_tool = state["endpoint_openai_tools"][0]
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "echo_tool"
    assert "query" in openai_tool["function"]["parameters"]["properties"]
    assert state["endpoint_tool_result"] == "echo:hi"
    assert state["endpoint_user_messages"] == [{"role": "user", "content": "continue"}]
    assert state["endpoint_stop"]["done"] is True
    assert state["endpoint_stop"]["stop_condition"] == "state_done"
    assert "runtime_id" not in state["runtime"]
    assert "endpoint_root_url" not in state


@pytest.mark.asyncio
async def test_state_helpers_load_runtime_tools_while_rollout_is_active() -> None:
    harness = vf.Harness(
        program=state_tools_program,
        toolsets=[vf.Toolset(tools=[echo_tool])],
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)

    assert state["tool_result"] == "echo:state"
    assert "runtime_id" not in state["runtime"]


@pytest.mark.asyncio
async def test_base_program_returns_tool_errors_to_model() -> None:
    client = FakeModelClient(
        [
            fake_response(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="failing_tool",
                        arguments='{"section_id": "bad"}',
                    )
                ]
            ),
            fake_response(content="Recovered."),
        ]
    )
    harness = vf.Harness(
        client=cast(Client, client),
        model="fake",
        toolsets=[vf.Toolset(tools=[failing_tool])],
        max_turns=2,
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)

    tool_message = state["completion"][1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "Invalid section_id format."
    assert state["completion"][-1]["content"] == "Recovered."
    assert state["error"] is None


@pytest.mark.asyncio
async def test_callable_tool_can_accept_name_argument() -> None:
    harness = vf.Harness(toolsets=[vf.Toolset(tools=[named_tool])])
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)
    harness.runtime.prepare_state(task, state)

    result = await harness.runtime.call_tool("named_tool", task, state, name="Ada")

    assert result == "name:Ada"


@pytest.mark.asyncio
async def test_callable_tool_rejects_reserved_hidden_args() -> None:
    harness = vf.Harness(toolsets=[vf.Toolset(tools=[echo_tool])])
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)
    harness.runtime.prepare_state(task, state)

    with pytest.raises(ValueError, match="runtime is reserved"):
        await harness.runtime.call_tool("echo_tool", task, state, runtime="bad")


@pytest.mark.asyncio
async def test_callable_tools_are_available_through_mcp_proxy() -> None:
    harness = vf.Harness(
        program=mcp_proxy_program,
        toolsets=[vf.Toolset(tools=[echo_tool])],
        tool_protocol="mcp",
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)
    await harness.teardown()

    assert state["runtime"]["tool_protocol"] == "mcp"
    assert state["mcp_tools"] == ["echo_tool"]
    assert state["mcp_result"] == "echo:hi"


@pytest.mark.asyncio
async def test_command_env_exposes_protocol_native_tool_payloads() -> None:
    harness = vf.Harness(toolsets=[vf.Toolset(tools=[echo_tool])])
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)
    state["endpoint_root_url"] = "http://127.0.0.1:1/rollout/test"
    state["endpoint_base_url"] = "http://127.0.0.1:1/rollout/test/v1"
    harness.runtime.prepare_state(task, state)

    env = await command_env({}, task, state, harness.runtime, include_base=False)

    openai_tools = json.loads(env["VF_TOOLS_JSON"])
    vf_tools = json.loads(env["VF_TOOL_DEFS_JSON"])
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "echo_tool"
    assert vf_tools[0]["name"] == "echo_tool"


def test_sandbox_base_program_uses_openai_tool_payloads() -> None:
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)

    program = sandbox_runner_program(
        program={},
        task=task,
        state=state,
        mode="base",
        entrypoint=None,
        max_turns=3,
        tool_defs=[
            Tool(
                name="echo_tool",
                description="",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )

    files = cast(dict[str, str], program["files"])
    tool_payloads = json.loads(files[TOOL_DEFS_PATH])
    assert tool_payloads == [
        {
            "type": "function",
            "function": {
                "name": "echo_tool",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tool_payloads_by_protocol = json.loads(files[TOOL_DEFS_BY_PROTOCOL_PATH])
    assert tool_payloads_by_protocol["openai_responses"][0]["name"] == "echo_tool"
    assert tool_payloads_by_protocol["anthropic_messages"][0]["name"] == "echo_tool"


def test_mcp_tool_protocol_injects_proxy_into_sandbox_program() -> None:
    harness = vf.Harness(
        program={"sandbox": True},
        sandbox={"image": "python:3.11-slim"},
        tool_protocol="mcp",
    )

    program = harness.prepare_sandbox_program({})
    sandbox = harness.prepare_sandbox_config({"image": "python:3.11-slim"})

    files = cast(dict[str, str], program["files"])
    env = cast(dict[str, str], program["env"])
    assert MCP_PROXY_PATH in files
    assert env["VF_TOOL_PROTOCOL"] == "mcp"
    assert json.loads(env["VF_MCP_TOOL_COMMAND_JSON"]) == ["python3", MCP_PROXY_PATH]
    packages = sandbox["packages"]
    assert isinstance(packages, list)
    assert "mcp>=1.14.1" in packages


@pytest.mark.asyncio
async def test_command_program_accepts_callable_tools_as_mcp() -> None:
    harness = vf.Harness(
        program={"command": [sys.executable, "-c", MCP_COMMAND_CLIENT]},
        toolsets=[vf.Toolset(tools=[echo_tool])],
        tool_protocol="mcp",
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)
    await harness.teardown()

    output = json.loads(state["command"]["stdout"])
    assert output == {"tools": ["echo_tool"], "result": "echo:hi"}


@pytest.mark.asyncio
async def test_nested_harness_inherits_model_controls_with_new_rollout_scope() -> None:
    harness = vf.Harness(program=parent_program)
    task = vf.Task({"prompt": [{"role": "user", "content": "parent"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["model"] = "model-a"
    state["runtime"]["sampling_args"] = {"temperature": 0.2}
    state["runtime"]["group_key"] = "group-a"
    harness.runtime.bind_model_client(state, cast(Client, FakeClient()))

    state = await harness.run(task, state)

    child_state = state["child_state"]
    assert child_state["trajectory_id"] != state["trajectory_id"]
    assert child_state["child_runtime"]["model"] == "model-a"
    assert child_state["child_runtime"]["sampling_args"] == {"temperature": 0.2}
    assert child_state["child_runtime"]["group_key"] == "group-a"
    assert child_state["child_runtime"]["client_key"] == state["runtime"]["client_key"]
    assert (
        state["child_rollouts"][0]["state"]["trajectory_id"]
        == child_state["trajectory_id"]
    )
    assert state["child_rollouts"][0]["state"]["metrics"] == child_state["metrics"]
    assert "runtime_id" not in state["child_rollouts"][0]["state"]["runtime"]
    assert "client_key" not in state["child_rollouts"][0]["state"]["runtime"]
    assert "client_key" not in state["child_rollouts"][0]["state"]["child_runtime"]


@pytest.mark.asyncio
async def test_state_finalize_strips_nested_runtime_handles() -> None:
    harness = vf.Harness(program=parent_program)
    task = vf.Task({"prompt": [{"role": "user", "content": "parent"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["model"] = "model-a"
    state["runtime"]["group_key"] = "group-a"
    harness.runtime.bind_model_client(state, cast(Client, FakeClient()))

    state = await harness.run(task, state)
    state.finalize()

    assert "runtime_id" not in state["runtime"]
    assert "client_key" not in state["runtime"]
    assert "runtime_id" not in state["child_state"]["runtime"]
    assert "client_key" not in state["child_state"]["runtime"]
    assert "client_key" not in state["child_state"]["child_runtime"]


@pytest.mark.asyncio
async def test_nested_harness_can_use_own_model_controls() -> None:
    harness = vf.Harness(program=parent_calls_owned_child_program)
    task = vf.Task({"prompt": [{"role": "user", "content": "parent"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["model"] = "parent-model"
    state["runtime"]["sampling_args"] = {"temperature": 0.2}
    harness.runtime.bind_model_client(state, cast(Client, FakeClient()))

    state = await harness.run(task, state)

    child_state = state["child_state"]
    assert child_state["child_runtime"]["model"] == "child-model"
    assert "client_key" not in child_state["child_runtime"]
    assert "client_key" not in state["runtime"]


@pytest.mark.asyncio
async def test_toolset_can_contribute_stop_condition() -> None:
    harness = vf.Harness(
        program=mark_submitted,
        toolsets=[vf.Toolset(stop=[submitted])],
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)

    assert state["submitted"] is True
    assert state["is_completed"] is True
    assert state["stop_condition"] == "submitted"


@pytest.mark.asyncio
async def test_runtime_owned_model_clients_close_after_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime()
    client = FakeClient()
    state = vf.State.for_task(vf.Task({}).freeze())

    monkeypatch.setattr("verifiers.v1.runtime.resolve_client", lambda config: client)

    runtime.bind_model_client(
        state,
        ClientConfig(
            client_type="openai_chat_completions",
            api_base_url="https://example.com/v1",
            api_key_var="KEY",
        ),
    )
    await runtime.release_model_client(state)

    assert client.closed is True
    assert runtime.model_clients == {}


@pytest.mark.asyncio
async def test_mcp_lifetime_follows_toolset_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def connect_mcp_tool(
        spec: vf.MCPTool, exit_stack: AsyncExitStack
    ) -> list[FakeMCPHandle]:
        _ = exit_stack
        return [FakeMCPHandle(spec.command)]

    monkeypatch.setattr(mcp_utils, "connect_mcp_tool", connect_mcp_tool)

    harness = vf.Harness(
        toolsets=[
            vf.Toolset(tools=[vf.MCPTool("global_tool")], scope="global"),
            vf.Toolset(tools=[vf.MCPTool("rollout_tool")], scope="rollout"),
            vf.Toolset(tools=[vf.MCPTool("group_tool")], scope="group"),
        ]
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state_a = vf.State.for_task(task)
    state_a["runtime"]["group_key"] = "group"
    state_b = vf.State.for_task(task)
    state_b["runtime"]["group_key"] = "group"

    await harness.runtime.ensure_mcp_tools(state_a)
    await harness.runtime.ensure_mcp_tools(state_b)

    keys = sorted(harness.runtime.mcp_exit_stacks)
    assert len([key for key in keys if key.startswith("global:")]) == 1
    assert len([key for key in keys if key.startswith("group:")]) == 1
    assert len([key for key in keys if key.startswith("rollout:")]) == 2
    assert sorted(harness.runtime.all_exposed_tools(state_a)) == [
        "global_tool",
        "group_tool",
        "rollout_tool",
    ]

    await harness.runtime.close_mcp_tools(state_a)

    keys = sorted(harness.runtime.mcp_exit_stacks)
    assert len([key for key in keys if key.startswith("global:")]) == 1
    assert len([key for key in keys if key.startswith("group:")]) == 1
    assert len([key for key in keys if key.startswith("rollout:")]) == 1

    await harness.runtime.close_mcp_tools(state_b)
    await harness.runtime.cleanup_group([task, task], [state_a, state_b])

    keys = sorted(harness.runtime.mcp_exit_stacks)
    assert len([key for key in keys if key.startswith("global:")]) == 1
    assert not [key for key in keys if key.startswith("group:")]
    assert not [key for key in keys if key.startswith("rollout:")]

    await harness.teardown()
    assert harness.runtime.mcp_exit_stacks == {}


@pytest.mark.asyncio
async def test_program_sandbox_group_scope_reuses_and_cleans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sandboxes(monkeypatch)

    harness = vf.Harness(
        program={"sandbox": True, "command": ["python", "-c", "print('ok')"]},
        sandbox={"image": "python:3.11-slim", "scope": "group"},
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state_a = vf.State.for_task(task)
    state_a["runtime"]["group_key"] = "group"
    state_b = vf.State.for_task(task)
    state_b["runtime"]["group_key"] = "group"

    state_a = await harness.run(task, state_a)
    state_b = await harness.run(task, state_b)

    assert FakeSandboxClient.created == ["sbx-1"]
    assert state_a["sandbox_id"] == "sbx-1"
    assert state_b["sandbox_id"] == "sbx-1"
    assert FakeSandboxClient.deleted == []

    await harness.cleanup_group([task, task], [state_a, state_b])

    assert FakeSandboxClient.deleted == ["sbx-1"]


@pytest.mark.asyncio
async def test_program_sandbox_global_scope_lives_until_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sandboxes(monkeypatch)

    harness = vf.Harness(
        program={"sandbox": True, "command": ["python", "-c", "print('ok')"]},
        sandbox={"image": "python:3.11-slim", "scope": "global"},
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)

    assert FakeSandboxClient.created == ["sbx-1"]
    assert state["sandbox_id"] == "sbx-1"
    assert FakeSandboxClient.deleted == []

    await harness.teardown()

    assert FakeSandboxClient.deleted == ["sbx-1"]


@pytest.mark.asyncio
async def test_toolset_can_bind_to_primary_program_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sandboxes(monkeypatch)

    harness = vf.Harness(
        program={"sandbox": True, "command": ["python", "-c", "print('ok')"]},
        sandbox={"image": "python:3.11-slim", "scope": "group"},
        toolsets=[vf.Toolset(tools=[program_sandbox_id], sandbox="program")],
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["group_key"] = "group"

    state = await harness.run(task, state)
    result = await harness.runtime.call_tool("program_sandbox_id", task, state)

    assert result == state["sandbox_id"]

    await harness.cleanup_group([task], [state])

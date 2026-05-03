from __future__ import annotations

from contextlib import AsyncExitStack
from typing import cast

import pytest

import verifiers.v1 as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import Tool
from verifiers.v1.runtime import Runtime
from verifiers.v1.utils import mcp_utils


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

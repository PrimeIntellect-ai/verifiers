from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import urllib.request
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import verifiers.v1 as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import Response, ResponseMessage, ToolCall
from verifiers.types import Tool
from verifiers.v1.runtime import Runtime
from verifiers.v1.utils import mcp_utils
from verifiers.v1.utils.mcp_proxy_utils import MCP_PROXY_CONFIG_PATH, MCP_PROXY_PATH
from verifiers.v1.utils.mcp_proxy_utils import proxy_command, proxy_source
from verifiers.v1.utils.program_utils import command_env
from verifiers.v1.utils.sandbox_program_utils import (
    RUNNER_CONFIG_PATH,
    TOOL_DEFS_BY_PROTOCOL_PATH,
    TOOL_DEFS_PATH,
    apply_internal_state_patch,
    runner_source,
    sandbox_runner_program,
)
from verifiers.v1.utils.sandbox_utils import collect_sandbox_artifacts


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


class CapturingModelClient(FakeModelClient):
    def __init__(self, responses: list[Response]):
        super().__init__(responses)
        self.requests: list[dict[str, object]] = []

    async def get_response(self, **kwargs: object) -> Response:
        prompt = kwargs.get("prompt")
        if isinstance(prompt, list):
            kwargs["prompt"] = list(prompt)
        self.requests.append(dict(kwargs))
        return await super().get_response(**kwargs)


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


async def borrowed_record_tool(value: str, state) -> str:
    state.setdefault("borrowed_tool_values", []).append(value)
    return f"recorded:{value}"


async def borrowed_stage_tool(value: str, state) -> str:
    state.setdefault("borrowed_stage_values", []).append(value)
    return f"stage:{value}"


async def named_tool(name: str) -> str:
    return f"name:{name}"


async def failing_tool(section_id: str) -> str:
    _ = section_id
    raise ValueError("Invalid section_id format.")


async def finish_tool(answer: str, state) -> str:
    state["answer"] = answer
    state.stop("submitted")
    return "submitted"


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


async def child_reads_program_sandbox(task, state) -> dict[str, object]:
    _ = task
    tools = state.tools()
    state["borrowed_sandbox_id"] = await tools["program_sandbox_id"]()
    return state


def install_fake_sandboxes(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandboxClient.reset()
    module = SimpleNamespace(
        AsyncSandboxClient=FakeSandboxClient,
        CreateSandboxRequest=FakeCreateSandboxRequest,
    )
    monkeypatch.setitem(sys.modules, "prime_sandboxes", module)


def install_fake_endpoint_tunnel(monkeypatch: pytest.MonkeyPatch) -> None:
    async def get_tunnel_url(self) -> str:
        _ = self
        return "http://127.0.0.1:1"

    monkeypatch.setattr(
        "verifiers.v1.utils.endpoint_utils.Endpoint.get_tunnel_url",
        get_tunnel_url,
    )


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
    config_path = proxy_path.with_suffix(".json")
    config_path.write_text(
        json.dumps(
            {
                "tool_base_url": f"{state['endpoint_root_url'].rstrip('/')}/vf/tools",
                "tool_api_key": "intercepted",
            }
        )
    )
    try:
        server = StdioServerParameters(
            command=sys.executable,
            args=[str(proxy_path), str(config_path)],
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
        config_path.unlink(missing_ok=True)


async def child_program(task, state):
    _ = task
    return {
        "child_runtime": dict(state["runtime"]),
        "child_trajectory_id": state["trajectory_id"],
    }


async def parent_program(task, state):
    runtime = state.runtime()
    child = vf.Harness(
        program=child_program,
        client=runtime.model_client(state),
        model=runtime.model(state),
        sampling_args=runtime.sampling_args(state),
    )
    child_task = vf.Task({"prompt": [{"role": "user", "content": "child"}]}).freeze()
    child_state = await child.run(child_task)
    return {"child_state": child_state}


async def mark_submitted(task, state):
    _ = task
    state["submitted"] = True
    return state


async def parent_calls_owned_child_program(task, state):
    child = vf.Harness(
        program=child_program, client=cast(Client, FakeClient()), model="child-model"
    )
    child_task = vf.Task({"prompt": [{"role": "user", "content": "child"}]}).freeze()
    child_state = await child.run(child_task)
    return {"child_state": child_state}


async def update_summary_with_resolved_handles(task, state):
    _ = task
    child = vf.Harness(system_prompt="Summarize the parent rollout in one word.")
    child_task = vf.Task(
        {"prompt": [{"role": "user", "content": str(state["completion"])}]}
    ).freeze()
    child_state = state.for_task(
        child_task,
        borrow="model",
        transcript="append",
    )
    child_state = await child.run(child_task, child_state)
    state["summary"] = child_state["completion"][0]["content"]
    state["child_trajectory_id"] = child_state["trajectory_id"]


async def update_child_uses_borrowed_tool(task, state):
    _ = task
    child = vf.Harness(max_turns=2)
    child_task = vf.Task({"prompt": [{"role": "user", "content": "inspect"}]}).freeze()
    child_state = state.for_task(
        child_task,
        borrow="model",
        tools="borrowed_record_tool",
        transcript="append",
    )
    child_state = await child.run(child_task, child_state)
    state["child_completion"] = child_state["completion"][-1]["content"]
    state["child_trajectory_id"] = child_state["trajectory_id"]


async def update_parallel_children_use_borrowed_tool(task, state):
    _ = task

    async def run_child(label: str) -> vf.State:
        child_task = vf.Task(
            {"prompt": [{"role": "user", "content": f"inspect {label}"}]}
        ).freeze()
        child_state = state.for_task(
            child_task,
            borrow="model",
            tools="borrowed_stage_tool",
            transcript="append",
        )
        return await vf.Harness(max_turns=2).run(child_task, child_state)

    children = await asyncio.gather(run_child("a"), run_child("b"))
    state["update_child_trajectory_ids"] = [
        child["trajectory_id"] for child in children
    ]


async def reward_child_uses_borrowed_tool(task, state) -> float:
    _ = task
    child_task = vf.Task(
        {"prompt": [{"role": "user", "content": "score sandbox state"}]}
    ).freeze()
    child_state = state.for_task(
        child_task,
        borrow="model",
        tools="borrowed_stage_tool",
    )
    child_state = await vf.Harness(max_turns=2).run(child_task, child_state)
    state["reward_child_completion"] = child_state["completion"][-1]["content"]
    state["reward_child_requests"] = child_state["num_model_requests"]
    return float("reward" in state.get("borrowed_stage_values", []))


async def submitted(task, state) -> bool:
    _ = task
    return bool(state.get("submitted"))


async def state_tools_program(task, state):
    _ = task
    tools = state.tools()
    state["tool_result"] = await tools["echo_tool"](query="state")
    state["tool_name"] = tools["echo_tool"].__name__
    return state


async def injected_tools_program(task, state, tools, tool_defs):
    _ = task
    state["tool_result"] = await tools["echo_tool"](query="injected")
    state["tool_name"] = tools["echo_tool"].__name__
    state["tool_docs"] = tools["echo_tool"].__doc__
    state["tool_defs"] = [tool.name for tool in tool_defs]
    return state


async def replay_answer_program(task, state):
    state["answer"] = task["answer"]
    return state


@vf.reward
async def replay_reward(task, state) -> float:
    return float(state.get("answer") == task.get("answer"))


def test_model_client_default_keys_are_rollout_local() -> None:
    runtime = Runtime()
    client = FakeClient()
    state_a = vf.State.for_task(vf.Task({}).freeze())
    state_b = vf.State.for_task(vf.Task({}).freeze())

    runtime.bind_model_client(state_a, cast(Client, client))
    runtime.bind_model_client(state_b, cast(Client, client))

    assert state_a["runtime"]["client_key"] != state_b["runtime"]["client_key"]
    assert len(runtime.model_clients) == 2


def test_v1_state_does_not_copy_task_answer_to_top_level() -> None:
    task = vf.Task({"answer": "gold"}).freeze()
    state = vf.State.for_task(task)

    assert "answer" not in state
    assert state["task"]["answer"] == "gold"


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
    assert state["tool_name"] == "echo_tool"
    assert "runtime_id" not in state["runtime"]


@pytest.mark.asyncio
async def test_entrypoint_program_receives_callable_tools() -> None:
    harness = vf.Harness(
        program=injected_tools_program,
        toolsets=[vf.Toolset(tools=[echo_tool])],
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)

    assert state["tool_result"] == "echo:injected"
    assert state["tool_name"] == "echo_tool"
    assert state["tool_defs"] == ["echo_tool"]


@pytest.mark.asyncio
async def test_offline_replay_program_scores_without_model_client() -> None:
    taskset = vf.Taskset(
        source=[
            {
                "prompt": [{"role": "user", "content": "Return the answer."}],
                "answer": "solved",
            }
        ],
        rewards=[replay_reward],
    )
    harness = vf.Harness(program=replay_answer_program)
    harness.attach_taskset(taskset)
    task = next(iter(taskset))

    state = await harness.run(task)
    await harness.teardown()

    assert state["answer"] == "solved"
    assert state["reward"] == 1.0
    assert state["stop_condition"] == "program_completed"
    assert state["trajectory"] == []
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
async def test_base_program_stops_after_tool_calls_state_stop() -> None:
    client = FakeModelClient(
        [
            fake_response(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="finish_tool",
                        arguments='{"answer": "done"}',
                    )
                ]
            )
        ]
    )
    harness = vf.Harness(
        client=cast(Client, client),
        model="fake",
        toolsets=[vf.Toolset(tools=[finish_tool])],
        max_turns=3,
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)

    assert state["answer"] == "done"
    assert state["is_completed"] is True
    assert state["stop_condition"] == "submitted"
    assert state["completion"][-1]["role"] == "tool"
    assert len(client.responses) == 0


@pytest.mark.asyncio
async def test_base_program_submits_system_prompt_before_prompt() -> None:
    client = CapturingModelClient([fake_response(content="ok")])
    harness = vf.Harness(client=cast(Client, client), model="fake", max_turns=1)
    task = vf.Task(
        {
            "system_prompt": "Use a short answer.",
            "prompt": [{"role": "user", "content": "hi"}],
        }
    ).freeze()

    state = await harness.run(task)

    prompt = cast(list[object], client.requests[0]["prompt"])
    assert [getattr(message, "role", None) for message in prompt] == [
        "system",
        "user",
    ]
    assert state["system_prompt"] == [
        {"role": "system", "content": "Use a short answer."}
    ]
    assert state["prompt"][0]["role"] == "system"
    assert state["completion"][-1]["content"] == "ok"


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
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    state = await harness.run(task)
    await harness.teardown()

    assert state["mcp_tools"] == ["echo_tool"]
    assert state["mcp_result"] == "echo:hi"


@pytest.mark.asyncio
async def test_command_env_exposes_model_endpoint_without_tool_payloads() -> None:
    harness = vf.Harness(toolsets=[vf.Toolset(tools=[echo_tool])])
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)
    state["endpoint_root_url"] = "http://127.0.0.1:1/rollout/test"
    state["endpoint_base_url"] = "http://127.0.0.1:1/rollout/test/v1"
    harness.runtime.prepare_state(task, state)

    env = await command_env({}, task, state, harness.runtime, include_base=False)

    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:1/rollout/test/v1"
    assert env["OPENAI_API_KEY"] == "intercepted"
    assert set(env) == {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    }


def test_sandbox_base_program_uses_openai_tool_payloads() -> None:
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)

    program = sandbox_runner_program(
        program={},
        task=task,
        state=state,
        mode="base",
        fn_ref=None,
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
    runner_config = json.loads(files[RUNNER_CONFIG_PATH])
    assert runner_config == {"max_turns": 3}
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


def test_sandbox_python_program_installs_runtime_client_deps() -> None:
    harness = vf.Harness(program={"sandbox": True}, sandbox={"packages": ["numpy"]})

    sandbox = harness.prepare_sandbox_config(
        {"packages": ["numpy"]},
        {"sandbox": True},
    )

    packages = sandbox["packages"]
    assert isinstance(packages, list)
    assert packages == ["numpy", "openai", "anthropic", "requests"]


@pytest.mark.asyncio
async def test_sandbox_base_program_max_turns_zero_is_unbounded(
    tmp_path: Path,
) -> None:
    namespace: dict[str, object] = {}
    source = runner_source().rsplit("asyncio.run(main())", 1)[0]
    exec(source, namespace)
    config_path = tmp_path / "runner_config.json"
    config_path.write_text(json.dumps({"max_turns": 0}))
    namespace["RUNNER_CONFIG_PATH"] = str(config_path)

    async def create_model_message(state, messages, client):
        _ = state, messages, client
        return {"role": "assistant", "content": "done"}

    async def call_user(state, transcript):
        _ = state, transcript
        return []

    async def check_stop(state):
        _ = state
        return False

    namespace["create_model_message"] = create_model_message
    namespace["call_user"] = call_user
    namespace["check_stop"] = check_stop

    state = {"prompt": [{"role": "user", "content": "hi"}], "runtime": {}}
    run_base = cast(Any, namespace["run_base"])
    result = await run_base({}, state, object())

    assert result["completion"] == [{"role": "assistant", "content": "done"}]
    assert result["stop_condition"] == "no_tools"


def test_sandbox_program_patch_cannot_set_lifecycle_fields() -> None:
    state = vf.State.for_task(vf.Task({"prompt": []}).freeze())

    with pytest.raises(RuntimeError, match="framework-managed"):
        apply_internal_state_patch(
            state,
            {"stop_condition": "user_stop"},
            mode="fn",
        )
    with pytest.raises(RuntimeError, match="framework-managed"):
        apply_internal_state_patch(
            state,
            {"is_completed": True},
            mode="base",
        )

    patch = {
        "stop_condition": "no_tools",
        "is_truncated": True,
        "error": {"message": "handled"},
    }
    apply_internal_state_patch(state, patch, mode="base")

    assert patch == {}
    assert state["stop_condition"] == "no_tools"
    assert state["is_truncated"] is True
    assert state["error"] == {"message": "handled"}


def test_program_tools_mcp_injects_proxy_into_sandbox_program() -> None:
    harness = vf.Harness(
        program={"sandbox": True, "command": ["true"], "tools": "mcp"},
        sandbox={"image": "python:3.11-slim"},
    )
    state = vf.State.for_task(vf.Task({}).freeze())
    state["endpoint_root_url"] = "http://127.0.0.1:1/rollout/test"

    program = harness.prepare_sandbox_program(
        {"sandbox": True, "command": ["true"], "tools": "mcp"}, state
    )
    sandbox = harness.prepare_sandbox_config(
        {"image": "python:3.11-slim"},
        {"sandbox": True, "command": ["true"], "tools": "mcp"},
    )

    files = cast(dict[str, str], program["files"])
    assert MCP_PROXY_PATH in files
    assert MCP_PROXY_CONFIG_PATH in files
    config = json.loads(files[MCP_PROXY_CONFIG_PATH])
    assert config == {
        "tool_base_url": "http://127.0.0.1:1/rollout/test/vf/tools",
        "tool_api_key": "intercepted",
    }
    assert proxy_command() == ["python3", MCP_PROXY_PATH, MCP_PROXY_CONFIG_PATH]
    packages = sandbox["packages"]
    assert isinstance(packages, list)
    assert "mcp>=1.14.1" in packages
    assert "requests" in packages


def test_program_tools_mcp_requires_sandbox_command() -> None:
    with pytest.raises(ValueError, match="requires program.sandbox"):
        vf.Harness(program={"command": ["true"], "tools": "mcp"})


def test_program_tools_callable_rejects_command_programs() -> None:
    with pytest.raises(ValueError, match="program.tools='callable'"):
        vf.Harness(program={"command": ["true"], "tools": "callable"})


async def write_real_sandbox_file(text: str, sandbox, state) -> str:
    command = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        f"Path('/tmp/host_tool.txt').write_text({text!r})\n"
        "print(Path('/tmp/host_tool.txt').read_text())\n"
        "PY\n"
    )
    result = await sandbox.execute(command, timeout=120, working_dir="/tmp")
    output = str(getattr(result, "stdout", "") or "").strip()
    state["real_sandbox_tool_output"] = output
    return output


REAL_MCP_PROXY_SCRIPT = r"""
import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server = StdioServerParameters(
        command="python3",
        args=["/tmp/vf_mcp_tools.py", "/tmp/vf_mcp_tools.json"],
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            result = await session.call_tool("echo_tool", {"query": "real-mcp"})
    payload = {
        "tools": [tool.name for tool in listed.tools],
        "result": result.content[0].text,
    }
    print(json.dumps(payload))


asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_real_sandbox_base_program_calls_host_callable_tool() -> None:
    client = FakeModelClient(
        [
            fake_response(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="write_real_sandbox_file",
                        arguments='{"text": "from-real-sandbox"}',
                    )
                ]
            ),
            fake_response(content="done"),
        ]
    )
    harness = vf.Harness(
        client=cast(Client, client),
        model="fake",
        program={"sandbox": True, "tools": "callable"},
        sandbox={
            "image": "python:3.11-slim",
            "scope": "group",
            "network_access": True,
            "timeout_minutes": 20,
            "command_timeout": 120,
        },
        toolsets=[
            vf.Toolset(
                tools=[write_real_sandbox_file],
                write=True,
                sandbox="program",
            )
        ],
        max_turns=2,
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "write file"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["group_key"] = "real-sandbox-callable-tools"

    state = await harness.run(task, state)

    try:
        assert state["sandbox_id"]
        assert state.get("real_sandbox_tool_output") == "from-real-sandbox", {
            "endpoint_root_url": state.get("endpoint_root_url"),
            "endpoint_base_url": state.get("endpoint_base_url"),
            "error": state.get("error"),
        }
        assert state["completion"][-1]["content"] == "done"
    finally:
        await harness.cleanup_group([task], [state])
        await harness.teardown()


@pytest.mark.asyncio
async def test_real_sandbox_command_program_uses_mcp_tool_proxy() -> None:
    harness = vf.Harness(
        program={
            "sandbox": True,
            "command": ["python", "/tmp/call_mcp.py"],
            "tools": "mcp",
            "files": {"/tmp/call_mcp.py": REAL_MCP_PROXY_SCRIPT},
        },
        sandbox={
            "image": "python:3.11-slim",
            "network_access": True,
            "timeout_minutes": 20,
            "command_timeout": 120,
        },
        toolsets=[vf.Toolset(tools=[echo_tool])],
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "call mcp"}]}).freeze()
    state = vf.State.for_task(task)

    state = await harness.run(task, state)

    try:
        stdout = state["command"]["stdout"].strip()
        payload = json.loads(stdout)
        assert payload == {"tools": ["echo_tool"], "result": "echo:real-mcp"}
    finally:
        await harness.cleanup_group([task], [state])
        await harness.teardown()


@pytest.mark.asyncio
async def test_nested_harness_uses_explicit_child_model_controls() -> None:
    harness = vf.Harness(program=parent_program)
    task = vf.Task({"prompt": [{"role": "user", "content": "parent"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["model"] = "model-a"
    state["runtime"]["sampling_args"] = {"temperature": 0.2}
    harness.runtime.bind_model_client(state, cast(Client, FakeClient()))

    state = await harness.run(task, state)

    child_state = state["child_state"]
    assert child_state["trajectory_id"] != state["trajectory_id"]
    assert child_state["child_runtime"]["model"] == "model-a"
    assert child_state["child_runtime"]["sampling_args"] == {"temperature": 0.2}
    assert "child_rollouts" not in state
    assert "client_key" not in child_state["runtime"]
    assert "client_key" not in child_state["child_runtime"]


@pytest.mark.asyncio
async def test_state_finalize_strips_nested_runtime_handles() -> None:
    harness = vf.Harness(program=parent_program)
    task = vf.Task({"prompt": [{"role": "user", "content": "parent"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["model"] = "model-a"
    state["runtime"]["group_key"] = "group-a"
    harness.runtime.bind_model_client(state, cast(Client, FakeClient()))
    state["runtime"]["resolved"] = {
        "model": {
            "runtime_id": harness.runtime.runtime_id,
            "client_key": state["runtime"]["client_key"],
        }
    }

    state = await harness.run(task, state)
    state.finalize()

    assert "runtime_id" not in state["runtime"]
    assert "client_key" not in state["runtime"]
    assert "resolved" not in state["runtime"]
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
async def test_update_child_harness_run_uses_resolved_runtime_handles() -> None:
    client = CapturingModelClient(
        [fake_response("parent answer"), fake_response("summary")]
    )
    harness = vf.Harness(updates=[update_summary_with_resolved_handles])
    task = vf.Task({"prompt": [{"role": "user", "content": "parent"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["model"] = "model-a"
    harness.runtime.bind_model_client(state, cast(Client, client))

    state = await harness.run(task, state)

    assert state["summary"] == "summary"
    assert len(client.requests) == 2
    assert len(state["trajectory"]) == 2
    assert state["trajectory"][0]["trajectory_id"] == state["trajectory_id"]
    assert state["trajectory"][1]["trajectory_id"] == state["child_trajectory_id"]
    assert state["num_model_requests"] == 2
    assert state["completion"][-1]["content"] == "summary"


@pytest.mark.asyncio
async def test_update_child_harness_can_borrow_live_tools() -> None:
    client = CapturingModelClient(
        [
            fake_response("parent answer"),
            fake_response(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="borrowed_record_tool",
                        arguments='{"value": "from-child"}',
                    )
                ]
            ),
            fake_response("child judged"),
        ]
    )
    harness = vf.Harness(
        updates=[update_child_uses_borrowed_tool],
        toolsets=[vf.Toolset(tools=[borrowed_record_tool], write=True)],
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "parent"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["model"] = "model-a"
    harness.runtime.bind_model_client(state, cast(Client, client))

    state = await harness.run(task, state)

    assert state["borrowed_tool_values"] == ["from-child"]
    assert state["child_completion"] == "child judged"
    assert len(client.requests) == 3
    assert len(state["trajectory"]) == 3
    assert state["trajectory"][0]["trajectory_id"] == state["trajectory_id"]
    assert state["trajectory"][1]["trajectory_id"] == state["child_trajectory_id"]
    assert state["trajectory"][2]["trajectory_id"] == state["child_trajectory_id"]
    assert state["num_model_requests"] == 3
    assert state["completion"][-1]["content"] == "child judged"


@pytest.mark.asyncio
async def test_update_and_reward_children_can_share_borrowed_live_tools() -> None:
    client = CapturingModelClient(
        [
            fake_response("parent answer"),
            fake_response(
                tool_calls=[
                    ToolCall(
                        id="call_update_a",
                        name="borrowed_stage_tool",
                        arguments='{"value": "update-a"}',
                    )
                ]
            ),
            fake_response("update a done"),
            fake_response(
                tool_calls=[
                    ToolCall(
                        id="call_update_b",
                        name="borrowed_stage_tool",
                        arguments='{"value": "update-b"}',
                    )
                ]
            ),
            fake_response("update b done"),
            fake_response(
                tool_calls=[
                    ToolCall(
                        id="call_reward",
                        name="borrowed_stage_tool",
                        arguments='{"value": "reward"}',
                    )
                ]
            ),
            fake_response('{"score": 1.0}'),
        ]
    )
    harness = vf.Harness(
        updates=[update_parallel_children_use_borrowed_tool],
        rewards=[reward_child_uses_borrowed_tool],
        toolsets=[vf.Toolset(tools=[borrowed_stage_tool], write=True)],
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "parent"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["model"] = "model-a"
    harness.runtime.bind_model_client(state, cast(Client, client))

    state = await harness.run(task, state)

    assert state["borrowed_stage_values"] == ["update-a", "update-b", "reward"]
    assert state["reward"] == 1.0
    assert state["reward_child_completion"] == '{"score": 1.0}'
    assert len(client.requests) == 7
    assert len(state["trajectory"]) == 5
    assert state["trajectory"][0]["trajectory_id"] == state["trajectory_id"]
    update_trajectory_ids = set(state["update_child_trajectory_ids"])
    assert len(update_trajectory_ids) == 2
    assert {
        str(record["trajectory_id"]) for record in state["trajectory"][1:]
    } == update_trajectory_ids
    assert all(
        record["completion"] != [{"role": "assistant", "content": '{"score": 1.0}'}]
        for record in state["trajectory"]
    )
    assert "runtime" not in state or "resolved" not in state.get("runtime", {})
    assert state["num_model_requests"] == 5
    assert state["reward_child_requests"] == 2


@pytest.mark.asyncio
async def test_toolset_can_contribute_stop_condition() -> None:
    harness = vf.Harness(
        program=mark_submitted,
        toolsets=[vf.Toolset(stops=[submitted])],
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
    install_fake_endpoint_tunnel(monkeypatch)

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
    assert "resolved" not in state_a.get("runtime", {})
    assert "resolved" not in state_b.get("runtime", {})
    assert "lease_key" not in state_a.get("runtime", {}).get("sandbox", {})
    assert "lease_key" not in state_b.get("runtime", {}).get("sandbox", {})


@pytest.mark.asyncio
async def test_program_sandbox_global_scope_lives_until_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sandboxes(monkeypatch)
    install_fake_endpoint_tunnel(monkeypatch)

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
    assert "resolved" not in state.get("runtime", {})


@pytest.mark.asyncio
async def test_optional_sandbox_program_artifact_records_none() -> None:
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)
    client = SimpleNamespace(
        execute_command=lambda **kwargs: SimpleNamespace(
            exit_code=2, stdout="", stderr=""
        )
    )

    await collect_sandbox_artifacts(
        client,
        "sbx",
        {
            "artifacts": {
                "missing_log": {
                    "path": "/tmp/missing.log",
                    "format": "text",
                    "optional": True,
                }
            }
        },
        state,
    )

    assert state["artifacts"]["missing_log"] is None


@pytest.mark.asyncio
async def test_sandbox_program_artifact_optional_must_be_boolean() -> None:
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)
    client = SimpleNamespace(
        execute_command=lambda **kwargs: SimpleNamespace(
            exit_code=2, stdout="", stderr=""
        )
    )

    with pytest.raises(TypeError, match="optional must be a boolean"):
        await collect_sandbox_artifacts(
            client,
            "sbx",
            {
                "artifacts": {
                    "missing_log": {
                        "path": "/tmp/missing.log",
                        "optional": "true",
                    }
                }
            },
            state,
        )


@pytest.mark.asyncio
async def test_toolset_can_bind_to_primary_program_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sandboxes(monkeypatch)
    install_fake_endpoint_tunnel(monkeypatch)

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


@pytest.mark.asyncio
async def test_toolset_sandbox_prefer_program_falls_back_to_owned_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sandboxes(monkeypatch)

    harness = vf.Harness(
        toolsets=[
            vf.Toolset(
                tools=[program_sandbox_id],
                sandbox={
                    "prefer": "program",
                    "image": "python:3.11-slim",
                    "scope": "rollout",
                },
            )
        ]
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = await harness.setup_state(task, vf.State.for_task(task))

    result = await state.tools()["program_sandbox_id"]()

    assert result == "sbx-1"
    assert FakeSandboxClient.created == ["sbx-1"]

    await harness.runtime.cleanup_rollout(task, state)

    assert FakeSandboxClient.deleted == ["sbx-1"]


@pytest.mark.asyncio
async def test_toolset_sandbox_prefer_program_uses_active_program_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sandboxes(monkeypatch)
    install_fake_endpoint_tunnel(monkeypatch)

    harness = vf.Harness(
        program={"sandbox": True, "command": ["python", "-c", "print('ok')"]},
        sandbox={"image": "python:3.11-slim", "scope": "group"},
        toolsets=[
            vf.Toolset(
                tools=[program_sandbox_id],
                sandbox={
                    "prefer": "program",
                    "image": "python:3.11-slim",
                    "scope": "rollout",
                },
            )
        ],
    )
    task = vf.Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)
    state["runtime"]["group_key"] = "group"

    state = await harness.run(task, state)
    result = await harness.runtime.call_tool("program_sandbox_id", task, state)

    assert result == state["sandbox_id"]
    assert FakeSandboxClient.created == ["sbx-1"]

    await harness.cleanup_group([task], [state])

    assert FakeSandboxClient.deleted == ["sbx-1"]


@pytest.mark.asyncio
async def test_child_state_can_borrow_primary_program_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sandboxes(monkeypatch)
    install_fake_endpoint_tunnel(monkeypatch)

    parent = vf.Harness(
        program={"sandbox": True, "command": ["python", "-c", "print('ok')"]},
        sandbox={"image": "python:3.11-slim", "scope": "group"},
    )
    child = vf.Harness(
        program=child_reads_program_sandbox,
        toolsets=[vf.Toolset(tools=[program_sandbox_id], sandbox="program")],
    )
    parent_task = vf.Task({"prompt": [{"role": "user", "content": "parent"}]}).freeze()
    parent_state = vf.State.for_task(parent_task)
    parent_state["runtime"]["group_key"] = "group"

    parent_state = await parent.run(parent_task, parent_state)
    child_task = vf.Task({"prompt": [{"role": "user", "content": "child"}]}).freeze()
    child_state = parent_state.for_task(child_task, borrow="sandbox")
    child_state = await child.run(child_task, child_state)

    assert parent_state["sandbox_id"] == "sbx-1"
    assert child_state["borrowed_sandbox_id"] == "sbx-1"
    assert FakeSandboxClient.deleted == []

    await parent.cleanup_group([parent_task], [parent_state])

    assert FakeSandboxClient.deleted == ["sbx-1"]

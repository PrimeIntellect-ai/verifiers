"""Tests for the composable architecture: Task, Agent, ComposableEnv, SweTaskAdapter, UserSimEnv."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verifiers.envs.experimental.agent import LLMAgent, ReActAgent, SingleTurnAgent, _filter_signature
from verifiers.envs.experimental.swe_task_adapter import SweTaskAdapter
from verifiers.envs.experimental.task import Task
from verifiers.types import (
    AssistantMessage,
    Response,
    ResponseMessage,
    State,
    Tool,
    ToolCall,
    TrajectoryStep,
)


# ── Fixtures ────────────────────────────────────────────────────────────


class MockTask:
    """Minimal Task implementation for testing."""

    needs_sandbox = False

    def get_dataset(self):
        return [{"question": "fix the bug", "info": {"id": 1}, "answer": ""}]

    def get_prompt(self, info):
        return [{"role": "user", "content": f"Fix bug #{info.get('id', 0)}"}]

    def get_image(self, info):
        return "python:3.11-slim"

    def get_workdir(self, info):
        return "/testbed"

    def get_env_vars(self):
        return {"FOO": "bar"}

    async def setup(self, sandbox_client, sandbox_id, state):
        state["task_setup_done"] = True

    async def evaluate(self, sandbox_client, sandbox_id, state):
        return 1.0

    def get_extra_tools(self):
        return []

    async def apply_gold_patch(self, sandbox_client, sandbox_id, state):
        pass


class MockSweTask:
    """Minimal SweTask protocol implementation for testing."""

    def get_dataset(self):
        return [{"question": "fix it", "info": {"id": 1}, "answer": ""}]

    def get_instruction(self, info):
        return "Please fix the bug in utils.py"

    def get_docker_image(self, info):
        return "registry/image:v1"

    def get_agent_workdir(self, info):
        return "/workspace"

    def get_env_vars(self):
        return {"PATH": "/usr/bin"}

    async def setup_sandbox(self, sandbox_client, sandbox_id, state):
        state["swe_setup_done"] = True

    async def run_tests(self, sandbox_client, sandbox_id, state, run_bg, timeout):
        return "All tests passed"

    def calculate_reward(self, test_output, info):
        return 1.0 if "passed" in test_output else 0.0

    async def apply_gold_patch(self, sandbox_client, sandbox_id, state):
        pass


def _make_mock_response(content="Done", tool_calls=None):
    """Create a mock Response object."""
    return Response(
        id="resp-1",
        created=0,
        model="test-model",
        usage=None,
        message=ResponseMessage(
            content=content,
            reasoning_content=None,
            thinking_blocks=None,
            tool_calls=tool_calls,
            finish_reason="stop",
            is_truncated=False,
            tokens=None,
        ),
    )


def _make_mock_client(responses=None):
    """Create a mock Client that returns preset responses."""
    client = AsyncMock()
    if responses is None:
        responses = [_make_mock_response()]
    client.get_response = AsyncMock(side_effect=responses)
    return client


def _make_state(client=None, model="test-model"):
    """Create a minimal State dict."""
    state = State(input={"prompt": [], "example_id": 0, "task": "test"})
    state["client"] = client or _make_mock_client()
    state["model"] = model
    state["sampling_args"] = {}
    state["trajectory"] = []
    state["timing"] = {"start_time": 0}
    return state


# ── Task Protocol ───────────────────────────────────────────────────────


def test_mock_task_implements_protocol():
    task = MockTask()
    assert isinstance(task, Task)


def test_task_get_prompt():
    task = MockTask()
    prompt = task.get_prompt({"id": 42})
    assert len(prompt) == 1
    assert "42" in prompt[0]["content"]


# ── filter_signature ────────────────────────────────────────────────────


def test_filter_signature_removes_args():
    def my_tool(command: str, state: dict, timeout: int = 90) -> str:
        return "ok"

    filtered = _filter_signature(my_tool, ["state", "timeout"])
    import inspect

    sig = inspect.signature(filtered)
    param_names = list(sig.parameters.keys())
    assert "command" in param_names
    assert "state" not in param_names
    assert "timeout" not in param_names


def test_filter_signature_noop_on_empty():
    def my_tool(x: int) -> int:
        return x

    result = _filter_signature(my_tool, [])
    assert result is my_tool


# ── LLMAgent ────────────────────────────────────────────────────────────


def test_react_agent_is_llm_agent_alias():
    assert ReActAgent is LLMAgent


def test_llm_agent_add_tool():
    agent = LLMAgent()

    def my_tool(question: str) -> str:
        """A test tool."""
        return "answer"

    agent.add_tool(my_tool)
    assert agent.tool_defs is not None
    assert len(agent.tool_defs) == 1
    assert agent.tool_defs[0].name == "my_tool"


def test_llm_agent_add_tool_with_args_to_skip():
    agent = LLMAgent()

    def execute_bash(command: str, state: dict, timeout: int = 90) -> str:
        """Execute a bash command."""
        return "output"

    agent.add_tool(execute_bash, args_to_skip=["state", "timeout"])
    assert len(agent.tool_defs) == 1
    td = agent.tool_defs[0]
    assert td.name == "execute_bash"
    # The schema should only have "command", not "state" or "timeout"
    props = td.parameters.get("properties", {})
    assert "command" in props
    assert "state" not in props
    assert "timeout" not in props


def test_llm_agent_remove_tool():
    agent = LLMAgent()

    def tool_a() -> str:
        return "a"

    agent.add_tool(tool_a)
    assert len(agent._tools) == 1
    agent.remove_tool("tool_a")
    assert len(agent._tools) == 0
    assert agent.tool_defs is None or len(agent.tool_defs) == 0


def test_llm_agent_inject_tool_args():
    agent = LLMAgent()

    def my_tool(command: str, state: dict) -> str:
        return f"state={state}"

    agent.add_tool(my_tool, args_to_skip=["state"])
    agent.inject_tool_args(state={"sandbox_id": "sb-123"})
    assert agent._injected_args["state"] == {"sandbox_id": "sb-123"}


@pytest.mark.asyncio
async def test_llm_agent_run_no_tools():
    """Agent with no tools should make one LLM call and return."""
    client = _make_mock_client([_make_mock_response("The answer is 42")])
    state = _make_state(client=client)
    agent = LLMAgent()

    steps = await agent.run([{"role": "user", "content": "What is 6*7?"}], state)
    assert len(steps) == 1
    assert steps[0]["completion"][0].content == "The answer is 42"


@pytest.mark.asyncio
async def test_llm_agent_run_with_tool_calls():
    """Agent should dispatch tool calls and continue the loop."""
    # First response: model calls a tool
    tool_call = ToolCall(id="tc-1", name="greet", arguments='{"name": "Alice"}')
    resp1 = _make_mock_response(content=None, tool_calls=[tool_call])
    # Second response: model returns final answer
    resp2 = _make_mock_response(content="Hello Alice!")
    client = _make_mock_client([resp1, resp2])
    state = _make_state(client=client)

    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hi {name}!"

    agent = LLMAgent(tools=[greet])
    steps = await agent.run([{"role": "user", "content": "Say hi to Alice"}], state)
    assert len(steps) == 2  # tool call turn + final answer turn


# ── SingleTurnAgent ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_turn_agent():
    client = _make_mock_client([_make_mock_response("I'm a user reporting a bug")])
    state = _make_state(client=client)
    agent = SingleTurnAgent(system_prompt="You are a user.")
    steps = await agent.run([{"role": "user", "content": "What's wrong?"}], state)
    assert len(steps) == 1
    assert steps[0]["extras"]["agent_id"] == "single_turn"


# ── SweTaskAdapter ──────────────────────────────────────────────────────


def test_swe_task_adapter_get_prompt():
    swe_task = MockSweTask()
    adapter = SweTaskAdapter(swe_task)
    prompt = adapter.get_prompt({"id": 1})
    assert len(prompt) == 1
    assert prompt[0]["role"] == "user"
    assert "fix the bug" in prompt[0]["content"]


def test_swe_task_adapter_get_prompt_with_system():
    swe_task = MockSweTask()
    adapter = SweTaskAdapter(swe_task, system_prompt="You are a developer.")
    prompt = adapter.get_prompt({"id": 1})
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"


def test_swe_task_adapter_get_image():
    swe_task = MockSweTask()
    adapter = SweTaskAdapter(swe_task)
    assert adapter.get_image({"id": 1}) == "registry/image:v1"


def test_swe_task_adapter_get_workdir():
    swe_task = MockSweTask()
    adapter = SweTaskAdapter(swe_task)
    assert adapter.get_workdir({"id": 1}) == "/workspace"


@pytest.mark.asyncio
async def test_swe_task_adapter_setup():
    swe_task = MockSweTask()
    adapter = SweTaskAdapter(swe_task)
    state = _make_state()
    await adapter.setup(MagicMock(), "sb-1", state)
    assert state.get("swe_setup_done") is True


@pytest.mark.asyncio
async def test_swe_task_adapter_evaluate():
    swe_task = MockSweTask()
    adapter = SweTaskAdapter(swe_task, test_timeout=60)
    state = _make_state()
    state["info"] = {"id": 1}
    state["_run_background_job"] = AsyncMock()

    reward = await adapter.evaluate(MagicMock(), "sb-1", state)
    assert reward == 1.0
    assert state["test_output"] == "All tests passed"


@pytest.mark.asyncio
async def test_swe_task_adapter_evaluate_missing_run_bg():
    swe_task = MockSweTask()
    adapter = SweTaskAdapter(swe_task)
    state = _make_state()
    # Don't set _run_background_job
    with pytest.raises(RuntimeError, match="_run_background_job"):
        await adapter.evaluate(MagicMock(), "sb-1", state)

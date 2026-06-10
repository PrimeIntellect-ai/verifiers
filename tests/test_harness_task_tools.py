"""The `SUPPORTS_TASK_TOOLS` gate (see `verifiers/v1/env.py`).

A harness without an MCP client (e.g. `rlm`) cannot expose a task's tool servers to the
model, so building an `Environment` that pairs such a harness with a taskset that declares
tools must fail fast with an informative error rather than silently dropping the tools.
"""

import sys
import types

import pytest

from harnesses.default import DefaultHarness
from harnesses.rlm import RLMHarness
from verifiers.v1.env import Environment, EnvConfig
from verifiers.v1.harness import Harness
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig


def test_harness_flags():
    assert Harness.SUPPORTS_TASK_TOOLS is True  # default: harnesses expose task tools
    assert DefaultHarness.SUPPORTS_TASK_TOOLS is True
    assert (
        RLMHarness.SUPPORTS_TASK_TOOLS is False
    )  # rlm drives its own tools, no MCP client


def _taskset_module(name: str, *, with_tools: bool) -> types.ModuleType:
    mod = types.ModuleType(name)

    class _Config(TasksetConfig):
        pass

    class _Taskset(Taskset):
        def load_tasks(self):
            return [Task(idx=0, instruction="x")]

    if with_tools:
        _Taskset.tools = lambda self, task: [
            object()
        ]  # non-empty -> declares tool servers

    mod.load_taskset = lambda config: _Taskset(config)
    return mod


def _harness_module(name: str, *, supports: bool) -> types.ModuleType:
    mod = types.ModuleType(name)

    class _Harness(Harness):
        SUPPORTS_TASK_TOOLS = supports

        async def launch(
            self, *args, **kwargs
        ):  # never invoked: the gate fires at __init__
            raise NotImplementedError

    mod.load_harness = lambda config: _Harness(config)
    return mod


@pytest.fixture
def plugins(monkeypatch):
    """Register tool / no-tool tasksets and supporting / non-supporting harnesses as
    importable plugin modules so the loaders resolve them by id."""
    for name, mod in {
        "tools_ts": _taskset_module("tools_ts", with_tools=True),
        "notools_ts": _taskset_module("notools_ts", with_tools=False),
        "nomcp_h": _harness_module("nomcp_h", supports=False),
        "mcp_h": _harness_module("mcp_h", supports=True),
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _build(taskset_id: str, harness_id: str) -> Environment:
    return Environment(
        EnvConfig(taskset={"id": taskset_id}, harness={"id": harness_id})
    )


def test_tool_taskset_on_unsupported_harness_raises(plugins):
    with pytest.raises(ValueError, match="does not support task tools"):
        _build("tools-ts", "nomcp-h")


def test_tool_taskset_on_supporting_harness_ok(plugins):
    _build("tools-ts", "mcp-h")


def test_no_tool_taskset_on_unsupported_harness_ok(plugins):
    _build("notools-ts", "nomcp-h")

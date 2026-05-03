from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from typing import Any

import pytest

import verifiers as vf
from verifiers.v1 import (
    Env,
    Harness,
    HarnessConfig,
    State,
    Task,
    Taskset,
    TasksetConfig,
    Toolset,
)


REF_MODULE = "v1_config_extension_refs"


def source_loader() -> list[dict[str, object]]:
    return [
        {
            "example_id": 0,
            "prompt": [{"role": "user", "content": "Say ok."}],
            "answer": "ok",
        }
    ]


def eval_source_loader() -> list[dict[str, object]]:
    return [
        {
            "prompt": [{"role": "user", "content": "Say eval ok."}],
            "answer": "eval ok",
        }
    ]


@vf.metric
async def config_metric(task: Mapping[str, object], state: dict[str, object]) -> float:
    return float(task.get("answer") == "ok" and state.get("answer") == "ok")


@vf.reward(weight=0.25)
async def config_reward(task: Mapping[str, object], state: dict[str, object]) -> float:
    return float(task.get("answer") == state.get("answer"))


@vf.cleanup(priority=5)
async def config_cleanup(task: Mapping[str, object], state: dict[str, object]) -> None:
    state["cleaned"] = True


@vf.cleanup(stage="group", priority=5)
async def config_group_cleanup(
    tasks: list[Mapping[str, object]], states: list[dict[str, object]]
) -> None:
    _ = tasks
    for state in states:
        state["group_cleaned"] = True


async def config_tool(query: str, prefix: str) -> str:
    return f"{prefix}:{query}"


async def direct_tool() -> str:
    return "direct"


async def hidden_tool() -> str:
    return "hidden"


async def config_user(
    task: Mapping[str, object], state: dict[str, object]
) -> list[dict[str, str]]:
    _ = task
    if state.get("user_called"):
        return []
    state["user_called"] = True
    return [{"role": "user", "content": "continue"}]


def token_factory(task: Mapping[str, object], state: dict[str, object]) -> str:
    _ = task, state
    return "secret-token"


async def config_user_with_bindings(
    task: Mapping[str, object],
    state: dict[str, object],
    token: str,
    transcript: list[object],
) -> list[dict[str, str]]:
    _ = task
    state["token_seen"] = token
    state["transcript_len"] = len(transcript)
    return [{"role": "user", "content": token}]


async def sandbox_user(
    task: Mapping[str, object], state: dict[str, object], sandbox: object
) -> list[dict[str, str]]:
    _ = task
    state["sandbox_seen"] = sandbox
    return [{"role": "user", "content": "sandbox ok"}]


async def config_program(
    task: Mapping[str, object], state: dict[str, object]
) -> dict[str, object]:
    state["answer"] = task["answer"]
    return {"program": "ran"}


ref_module = types.ModuleType(REF_MODULE)
setattr(ref_module, "source_loader", source_loader)
setattr(ref_module, "eval_source_loader", eval_source_loader)
setattr(ref_module, "config_metric", config_metric)
setattr(ref_module, "config_reward", config_reward)
setattr(ref_module, "config_cleanup", config_cleanup)
setattr(ref_module, "config_group_cleanup", config_group_cleanup)
setattr(ref_module, "config_tool", config_tool)
setattr(ref_module, "direct_tool", direct_tool)
setattr(ref_module, "hidden_tool", hidden_tool)
setattr(ref_module, "config_user", config_user)
setattr(ref_module, "token_factory", token_factory)
setattr(ref_module, "config_user_with_bindings", config_user_with_bindings)
setattr(ref_module, "sandbox_user", sandbox_user)
setattr(ref_module, "config_program", config_program)
sys.modules[REF_MODULE] = ref_module


def ref(name: str) -> str:
    return f"{REF_MODULE}:{name}"


def test_taskset_config_extends_constructor_surface() -> None:
    taskset = Taskset(
        config={
            "source": ref("source_loader"),
            "eval_source": ref("eval_source_loader"),
            "taskset_id": "configured",
            "metrics": [ref("config_metric")],
            "rewards": [ref("config_reward")],
            "cleanup": [ref("config_cleanup")],
            "toolsets": [
                {
                    "tools": [ref("config_tool")],
                    "bindings": {"config_tool.prefix": "task.answer"},
                }
            ],
            "user": ref("config_user"),
        }
    )

    rows = taskset.rows()
    eval_rows = taskset.eval_rows()
    task = taskset.task(rows[0])

    assert task["taskset_id"] == "configured"
    assert task["task_id"] == "0"
    assert eval_rows[0]["answer"] == "eval ok"
    assert taskset.metrics == [config_metric]
    assert taskset.rewards == [config_reward]
    assert taskset.cleanup == [config_cleanup]
    assert taskset.user is not None
    assert len(taskset.toolsets) == 1
    assert taskset.toolsets[0].tools == (config_tool,)
    assert taskset.toolsets[0].bindings == {"config_tool.prefix": "task.answer"}


def test_taskset_get_eval_dataset_uses_eval_source() -> None:
    taskset = Taskset(source=source_loader, eval_source=eval_source_loader)

    assert taskset.get_dataset()[0]["answer"] == "ok"
    assert taskset.get_eval_dataset()[0]["answer"] == "eval ok"


def test_env_passes_taskset_eval_dataset_to_environment() -> None:
    env = Env(
        taskset=Taskset(source=source_loader, eval_source=eval_source_loader),
        harness=Harness(program=config_program),
    )

    assert env.get_dataset()[0]["answer"] == "ok"
    assert env.get_eval_dataset()[0]["answer"] == "eval ok"


def test_harness_config_extends_constructor_surface() -> None:
    direct_toolset = Toolset(tools=[direct_tool])
    harness = Harness(
        toolsets=[direct_toolset],
        metrics=[config_metric],
        config={
            "program": ref("config_program"),
            "metrics": [],
            "rewards": [ref("config_reward")],
            "cleanup": [ref("config_cleanup")],
            "toolsets": [
                {
                    "tools": [ref("config_tool")],
                    "hide": ["config_tool"],
                }
            ],
            "user": ref("config_user"),
            "max_turns": 3,
        },
    )

    assert harness.program is config_program
    assert harness.config.max_turns == 3
    assert harness.metrics == [config_metric]
    assert harness.rewards == [config_reward]
    assert harness.cleanup == [config_cleanup]
    assert harness.user is not None
    assert len(harness.toolsets) == 2
    assert harness.toolsets[0] is direct_toolset
    assert harness.toolsets[1].hide == ("config_tool",)


def test_harness_max_turns_arg_overrides_config() -> None:
    harness = Harness(max_turns=9, config={"max_turns": 3})

    assert harness.config.max_turns == 9


def test_constructor_mapping_args_override_config_mapping_values() -> None:
    harness = Harness(
        sandbox={"image": "constructor", "nested": {"a": 1}},
        config={"sandbox": {"image": "config", "scope": "group", "nested": {"b": 2}}},
    )

    assert harness.sandbox == {
        "image": "constructor",
        "scope": "group",
        "nested": {"a": 1, "b": 2},
    }


@pytest.mark.asyncio
async def test_user_config_supports_scope_bindings_and_objects() -> None:
    harness = Harness(
        config={
            "user": {
                "fn": ref("config_user_with_bindings"),
                "scope": "group",
                "bindings": {"token": "objects.token"},
                "objects": {"token": ref("token_factory")},
            }
        }
    )
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = State.for_task(task)

    messages = await harness.runtime.user_messages(
        task, state, transcript=[{"role": "assistant", "content": "hello"}]
    )

    assert harness.user is not None
    assert harness.user.scope == "group"
    assert state["token_seen"] == "secret-token"
    assert state["transcript_len"] == 1
    assert messages == [{"role": "user", "content": "secret-token"}]


@pytest.mark.asyncio
async def test_user_config_can_request_scoped_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = object()
    harness = Harness(
        config={
            "user": {
                "fn": ref("sandbox_user"),
                "sandbox": {"image": "python:3.11-slim", "scope": "group"},
            }
        }
    )
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = State.for_task(task)

    async def resolve_user_sandbox(*args: Any, **kwargs: Any) -> object:
        _ = args, kwargs
        return sandbox

    monkeypatch.setattr(harness.runtime, "resolve_user_sandbox", resolve_user_sandbox)

    messages = await harness.runtime.user_messages(task, state)

    assert harness.user is not None
    assert harness.user.sandbox == {"image": "python:3.11-slim", "scope": "group"}
    assert state["sandbox_seen"] is sandbox
    assert messages == [{"role": "user", "content": "sandbox ok"}]


@pytest.mark.asyncio
async def test_configured_program_scores_and_cleans_rollout() -> None:
    taskset = Taskset(source=source_loader)
    harness = Harness(
        config={
            "program": ref("config_program"),
            "rewards": [ref("config_reward")],
            "cleanup": [ref("config_cleanup")],
        }
    )
    task = next(iter(taskset))
    state = await harness.run(task)

    assert state["program"] == "ran"
    assert state["answer"] == "ok"
    assert state["reward"] == 0.25
    assert state["cleaned"] is True
    assert state["is_completed"] is True


@pytest.mark.asyncio
async def test_harness_run_releases_group_scope_when_no_group_boundary() -> None:
    harness = Harness(program=config_program, cleanup=[config_group_cleanup])
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()

    state = await harness.run(task)

    assert state["group_cleaned"] is True


@pytest.mark.asyncio
async def test_harness_run_defers_group_cleanup_when_group_boundary_exists() -> None:
    harness = Harness(program=config_program, cleanup=[config_group_cleanup])
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()
    state = State.for_task(task)
    state["runtime"]["group_key"] = "group"

    state = await harness.run(task, state)

    assert "group_cleaned" not in state
    await harness.cleanup_group([task], [state])
    assert state["group_cleaned"] is True


def test_subclasses_can_define_new_config_surface() -> None:
    class CustomHarnessConfig(HarnessConfig):
        custom_flag: bool = False

    class CustomHarness(Harness):
        config_type = CustomHarnessConfig

    harness = CustomHarness(config={"custom_flag": True})

    assert harness.config.custom_flag is True
    assert "custom_flag" in CustomHarness.config_schema()


def test_config_schema_is_visible_from_primary_types() -> None:
    assert "toolsets" in Taskset.config_schema()
    assert "toolsets" in Harness.config_schema()
    assert "source" in TasksetConfig.schema_text()
    assert "eval_source" in TasksetConfig.schema_text()
    assert "program" in HarnessConfig.schema_text()


def test_configs_load_from_toml_sections(tmp_path) -> None:
    config_path = tmp_path / "env.toml"
    config_path.write_text(
        "\n".join(
            [
                "[env.taskset]",
                f'source = "{ref("source_loader")}"',
                f'rewards = ["{ref("config_reward")}"]',
                "",
                "[env.harness]",
                f'program = "{ref("config_program")}"',
                "max_turns = 7",
            ]
        )
    )

    taskset_config = TasksetConfig.from_toml(config_path, "env.taskset")
    harness_config = HarnessConfig.from_toml(config_path, ("env", "harness"))

    taskset = Taskset(config=taskset_config)
    harness = Harness(config=harness_config)

    assert taskset.source is source_loader
    assert taskset.rewards == [config_reward]
    assert harness.program is config_program
    assert harness.config.max_turns == 7


def test_task_runtime_tools_filter_exposed_tools() -> None:
    harness = Harness(toolsets=[Toolset(tools=[direct_tool, hidden_tool])])
    task = Task(
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "runtime": {"tools": ["direct_tool"]},
        }
    ).freeze()
    state = State.for_task(task)

    harness.runtime.prepare_state(task, state)

    assert state["tools"] == ["direct_tool"]
    assert [tool.name for tool in harness.runtime.tool_defs(state) or []] == [
        "direct_tool"
    ]
    assert list(harness.runtime.tool_calls(task, state)) == ["direct_tool"]


def test_toolset_config_is_load_bearing() -> None:
    toolset = Toolset(
        tools=[direct_tool],
        bindings={"hidden_tool.prefix": "task.answer"},
        config={
            "tools": [ref("hidden_tool")],
            "objects": {"source": ref("source_loader")},
            "write": True,
            "scope": "group",
            "cleanup": [ref("config_cleanup")],
        },
    )

    assert toolset.tools == (direct_tool, hidden_tool)
    assert toolset.bindings == {"hidden_tool.prefix": "task.answer"}
    assert toolset.objects == {"source": source_loader}
    assert toolset.write is True
    assert toolset.scope == "group"
    assert toolset.cleanup == (config_cleanup,)


def test_toolset_write_arg_overrides_config() -> None:
    toolset = Toolset(write=False, config={"write": True})

    assert toolset.write is False


def test_toolset_config_accepts_mcp_tool_specs() -> None:
    toolset = Toolset(
        config={
            "tools": [
                {
                    "command": "uvx",
                    "args": ["mcp-server-fetch"],
                    "env": {"API_KEY": "test"},
                    "cwd": "/tmp",
                }
            ],
        }
    )

    assert isinstance(toolset.tools[0], vf.MCPTool)
    assert toolset.tools[0].command == "uvx"
    assert toolset.tools[0].args == ("mcp-server-fetch",)
    assert toolset.tools[0].env == {"API_KEY": "test"}
    assert toolset.tools[0].cwd == "/tmp"


def test_add_toolset_accepts_same_shapes_as_constructor() -> None:
    taskset = Taskset(source=source_loader)
    harness = Harness()

    taskset.add_toolset({"tools": [direct_tool]})
    harness.add_toolset(config_tool)

    assert taskset.toolsets[0].tools == (direct_tool,)
    assert harness.toolsets[0].tools == (config_tool,)

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Mapping
from typing import Any, cast

import pytest

import verifiers as vf
from verifiers.v1 import (
    Env,
    EnvConfig,
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


@vf.metric(stage="group")
async def group_config_metric(
    tasks: list[Mapping[str, object]], states: list[dict[str, object]]
) -> list[float]:
    _ = tasks
    return [float(index) for index, _ in enumerate(states)]


@vf.reward(stage="group")
async def group_config_reward(
    tasks: list[Mapping[str, object]], states: list[dict[str, object]]
) -> list[float]:
    return [
        float(task.get("answer") == state.get("answer"))
        for task, state in zip(tasks, states)
    ]


@vf.advantage
async def config_advantage(
    tasks: list[Mapping[str, object]], states: list[dict[str, object]]
) -> list[float]:
    _ = tasks
    return [float(index) for index, _ in enumerate(states)]


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


@vf.setup(priority=5)
async def config_setup(task: Mapping[str, object], state: dict[str, object]) -> None:
    _ = task
    state["configured_setup"] = True
    cast(list[str], state.setdefault("setup_order", [])).append("config_setup")


@vf.update(priority=5)
async def config_update(task: Mapping[str, object], state: dict[str, object]) -> None:
    _ = task
    state["updated"] = True


@vf.reward
async def updated_reward(task: Mapping[str, object], state: dict[str, object]) -> float:
    _ = task
    return float(state.get("updated") is True)


@vf.update(stage="group")
async def config_group_update(
    tasks: list[Mapping[str, object]], states: list[dict[str, object]]
) -> None:
    _ = tasks
    for state in states:
        state["group_updated"] = True


@vf.reward(stage="group")
async def group_updated_reward(
    tasks: list[Mapping[str, object]], states: list[dict[str, object]]
) -> list[float]:
    _ = tasks
    return [float(state.get("group_updated") is True) for state in states]


async def config_tool(query: str, prefix: str) -> str:
    return f"{prefix}:{query}"


async def direct_tool() -> str:
    return "direct"


async def hidden_tool() -> str:
    return "hidden"


async def object_tool(value: str, box: dict[str, object]) -> str:
    values = cast(list[str], box.setdefault("values", []))
    values.append(value)
    return value


def load_object_box() -> dict[str, object]:
    return {"values": []}


async def update_from_binding(
    task: Mapping[str, object], state: dict[str, object], expected: str
) -> None:
    _ = task
    state["expected"] = expected


async def colliding_tool(value: str, token: str) -> str:
    return f"{token}:{value}"


async def colliding_update(
    task: Mapping[str, object], state: dict[str, object], token: str
) -> None:
    _ = task
    state["colliding_update_token"] = token


colliding_update.__name__ = "colliding_tool"


class DynamicSchemaTool:
    def __init__(self, tool_def: vf.Tool):
        self.name = tool_def.name
        self.tool_def = tool_def

    async def __call__(self, state: dict[str, object], **kwargs: object) -> str:
        calls = cast(list[object], state.setdefault("dynamic_tool_calls", []))
        calls.append({self.name: kwargs})
        return "recorded"


def dynamic_toolset(task: Mapping[str, object]) -> Toolset:
    tool = task["dynamic_tool"]
    if not isinstance(tool, Mapping):
        raise TypeError("dynamic_tool must be a mapping.")
    tool = cast(Mapping[str, Any], tool)
    return Toolset(
        tools=[
            DynamicSchemaTool(
                vf.Tool(
                    name=str(tool["name"]),
                    description=str(tool["description"]),
                    parameters=dict(cast(Mapping[str, Any], tool["parameters"])),
                )
            )
        ]
    )


class TruthyMarkerToolset(Toolset):
    async def fake_handler(self, expected: str) -> None:
        _ = expected


TruthyMarkerToolset.fake_handler.update = "truthy"


async def config_user(
    task: Mapping[str, object], state: dict[str, object]
) -> list[dict[str, str]]:
    _ = task
    if state.get("user_called"):
        return []
    state["user_called"] = True
    return [{"role": "user", "content": "continue"}]


def token_factory() -> str:
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


async def direct_user_with_transcript(
    task: Mapping[str, object],
    state: dict[str, object],
    transcript: list[object],
) -> list[dict[str, str]]:
    _ = task
    state["direct_transcript_len"] = len(transcript)
    return [{"role": "user", "content": "continue"}]


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


async def setup_aware_program(
    task: Mapping[str, object], state: dict[str, object]
) -> dict[str, object]:
    _ = task
    if state.get("configured_setup") is not True:
        raise AssertionError("setup did not run before program")
    return {"program_saw_setup": True}


def config_toolset(prefix: str = "cfg") -> Toolset:
    return Toolset(
        tools=[config_tool],
        bindings={"config_tool.prefix": prefix},
    )


ref_module = types.ModuleType(REF_MODULE)
setattr(ref_module, "source_loader", source_loader)
setattr(ref_module, "eval_source_loader", eval_source_loader)
setattr(ref_module, "config_metric", config_metric)
setattr(ref_module, "config_reward", config_reward)
setattr(ref_module, "config_advantage", config_advantage)
setattr(ref_module, "config_cleanup", config_cleanup)
setattr(ref_module, "config_group_cleanup", config_group_cleanup)
setattr(ref_module, "config_setup", config_setup)
setattr(ref_module, "config_update", config_update)
setattr(ref_module, "updated_reward", updated_reward)
setattr(ref_module, "config_group_update", config_group_update)
setattr(ref_module, "group_updated_reward", group_updated_reward)
setattr(ref_module, "config_tool", config_tool)
setattr(ref_module, "config_toolset", config_toolset)
setattr(ref_module, "dynamic_toolset", dynamic_toolset)
setattr(ref_module, "direct_tool", direct_tool)
setattr(ref_module, "hidden_tool", hidden_tool)
setattr(ref_module, "config_user", config_user)
setattr(ref_module, "token_factory", token_factory)
setattr(ref_module, "config_user_with_bindings", config_user_with_bindings)
setattr(ref_module, "sandbox_user", sandbox_user)
setattr(ref_module, "config_program", config_program)
setattr(ref_module, "setup_aware_program", setup_aware_program)
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
            "advantages": [ref("config_advantage")],
            "setups": [ref("config_setup")],
            "cleanups": [ref("config_cleanup")],
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
    assert taskset.advantages == [config_advantage]
    assert taskset.setups == [config_setup]
    assert taskset.cleanups == [config_cleanup]
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


def test_env_defaults_to_base_harness() -> None:
    taskset = Taskset(source=source_loader)
    env = Env(taskset=taskset)

    assert isinstance(env.harness, Harness)
    assert env.harness.taskset is taskset
    assert env.get_dataset()[0]["answer"] == "ok"


def test_env_capabilities_follow_v1_group_runtime_signals() -> None:
    rollout_env = Env(
        taskset=Taskset(source=source_loader, rewards=[config_reward]),
        harness=Harness(program=config_program),
    )
    group_metric_env = Env(
        taskset=Taskset(source=source_loader, metrics=[group_config_metric]),
        harness=Harness(program=config_program),
    )
    group_reward_env = Env(
        taskset=Taskset(source=source_loader, rewards=[group_config_reward]),
        harness=Harness(program=config_program),
    )
    advantage_env = Env(
        taskset=Taskset(source=source_loader, advantages=[config_advantage]),
        harness=Harness(program=config_program),
    )

    assert not rollout_env.requires_group_rollouts
    assert not rollout_env.provides_advantages
    assert group_metric_env.requires_group_rollouts
    assert not group_metric_env.provides_advantages
    assert group_reward_env.requires_group_rollouts
    assert not group_reward_env.provides_advantages
    assert advantage_env.requires_group_rollouts
    assert advantage_env.provides_advantages


def test_env_capabilities_follow_group_lifecycle_handlers() -> None:
    group_update_env = Env(
        taskset=Taskset(source=source_loader, updates=[config_group_update]),
        harness=Harness(program=config_program),
    )
    group_cleanup_env = Env(
        taskset=Taskset(source=source_loader, cleanups=[config_group_cleanup]),
        harness=Harness(program=config_program),
    )

    assert group_update_env.requires_group_rollouts
    assert not group_update_env.provides_advantages
    assert group_cleanup_env.requires_group_rollouts
    assert not group_cleanup_env.provides_advantages


def test_group_lifecycle_handlers_reject_extra_args() -> None:
    @vf.update(stage="group")
    async def bad_group_update(tasks, states, extra) -> None:
        _ = tasks, states, extra

    with pytest.raises(ValueError, match="exactly tasks and states"):
        Env(
            taskset=Taskset(source=source_loader, updates=[bad_group_update]),
            harness=Harness(program=config_program),
        )


def test_env_capabilities_follow_custom_taskset_init_group() -> None:
    class GroupSetupTaskset(Taskset):
        async def init_group(
            self, task: Task, num_rollouts: int
        ) -> tuple[list[Task], list[State]]:
            return await super().init_group(task, num_rollouts)

    env = Env(
        taskset=GroupSetupTaskset(source=source_loader),
        harness=Harness(program=config_program),
    )

    assert env.requires_group_rollouts
    assert not env.provides_advantages


def test_harness_config_extends_constructor_surface() -> None:
    direct_toolset = Toolset(tools=[direct_tool])
    harness = Harness(
        toolsets=[direct_toolset],
        metrics=[config_metric],
        config={
            "program": ref("config_program"),
            "metrics": [],
            "rewards": [ref("config_reward")],
            "advantages": [ref("config_advantage")],
            "setups": [ref("config_setup")],
            "cleanups": [ref("config_cleanup")],
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
    assert harness.advantages == [config_advantage]
    assert harness.setups == [config_setup]
    assert harness.cleanups == [config_cleanup]
    assert harness.user is not None
    assert len(harness.toolsets) == 2
    assert harness.toolsets[0] is direct_toolset
    assert harness.toolsets[1].hide == ("config_tool",)


def test_harness_owns_default_render_completion_update() -> None:
    harness = Harness(program=config_program)

    assert any(
        getattr(handler, "__self__", None) is harness
        and getattr(handler, "__name__", "") == "render_completion"
        for handler in harness.runtime.rollout_update
    )


@pytest.mark.asyncio
async def test_update_config_runs_before_rollout_scoring() -> None:
    harness = Harness(
        program=config_program,
        config={
            "updates": [{"fn": ref("config_update"), "priority": 5}],
            "rewards": [{"fn": ref("updated_reward"), "weight": 0.75}],
        },
    )
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()

    state = await harness.run(task)

    assert state["updated"] is True
    assert state["reward"] == 0.75
    assert getattr(harness.updates[0], "__name__") == "config_update"


@pytest.mark.asyncio
async def test_setup_config_runs_before_program() -> None:
    harness = Harness(
        config={
            "program": ref("setup_aware_program"),
            "setups": [{"fn": ref("config_setup"), "priority": 20}],
        },
    )
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()

    state = await harness.run(task)

    assert state["program_saw_setup"] is True
    assert state["setup_order"] == ["config_setup"]
    assert getattr(harness.setups[0], "__name__") == "config_setup"


@pytest.mark.asyncio
async def test_taskset_setup_runs_before_program() -> None:
    taskset = Taskset(source=source_loader, setups=[config_setup])
    harness = Harness(program=setup_aware_program)
    Env(taskset=taskset, harness=harness)
    task = next(iter(taskset))

    state = await harness.run(task)

    assert state["program_saw_setup"] is True
    assert state["setup_order"] == ["config_setup"]


@pytest.mark.asyncio
async def test_group_update_config_runs_before_group_scoring() -> None:
    harness = Harness(
        config={
            "updates": [{"fn": ref("config_group_update"), "stage": "group"}],
            "rewards": [{"fn": ref("group_updated_reward"), "stage": "group"}],
        },
    )
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = State.for_task(task)

    await harness.score_group([task], [state])

    assert state["group_updated"] is True
    assert state["reward"] == 1.0


def test_lifecycle_fields_are_framework_managed() -> None:
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = State.for_task(task)

    for key, value in {
        "is_completed": True,
        "stop_condition": "done",
        "is_truncated": True,
        "error": {"message": "boom"},
    }.items():
        with pytest.raises(RuntimeError, match="framework-managed"):
            State({key: value})
        with pytest.raises(RuntimeError, match="framework-managed"):
            state[key] = value
        with pytest.raises(RuntimeError, match="framework-managed"):
            state.update({key: value})
        with pytest.raises(RuntimeError, match="framework-managed"):
            state.setdefault(key, value)
        with pytest.raises(RuntimeError, match="framework-managed"):
            state.pop(key)
    with pytest.raises(RuntimeError, match="framework-managed"):
        state.popitem()
    with pytest.raises(RuntimeError, match="framework-managed"):
        state.clear()

    state._set_completed(True)
    state._set_stop_condition("done")
    state._set_truncated(True)
    state._set_error({"message": "boom"})

    assert state["is_completed"] is True
    assert state["stop_condition"] == "done"
    assert state["is_truncated"] is True
    assert state["error"] == {"message": "boom"}


def test_toolsets_config_accepts_addressable_map_and_fn_tables() -> None:
    taskset = Taskset(
        source=source_loader,
        config={
            "toolsets": {
                "direct": {"tools": [ref("direct_tool")]},
                "configured": {
                    "fn": ref("config_toolset"),
                    "prefix": "from_config",
                },
            }
        },
    )

    assert set(taskset.named_toolsets) == {"direct", "configured"}
    assert taskset.toolsets[0].tools == (direct_tool,)
    assert taskset.toolsets[1].bindings == {"config_tool.prefix": "from_config"}


@pytest.mark.asyncio
async def test_task_toolsets_show_hide_selects_named_defaults() -> None:
    harness = Harness(
        toolsets={
            "direct": Toolset(tools=[direct_tool]),
            "hidden": Toolset(tools=[hidden_tool]),
        }
    )
    task = Task(
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "toolsets": {"show": ["direct"]},
        }
    ).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    assert state["tools"] == ["direct_tool"]
    assert list(harness.runtime.tool_calls(task, state)) == ["direct_tool"]


@pytest.mark.asyncio
async def test_task_toolsets_can_add_rollout_local_toolsets() -> None:
    harness = Harness()
    task = Task(
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "answer": "ok",
            "toolsets": {
                "local": {
                    "tools": [ref("config_tool")],
                    "bindings": {"config_tool.prefix": "task.answer"},
                }
            },
        }
    ).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    assert state["tools"] == ["config_tool"]
    assert await state.get_tools()["config_tool"](query="q") == "ok:q"


@pytest.mark.asyncio
async def test_task_toolsets_can_add_dynamic_schema_backed_tools() -> None:
    harness = Harness()
    task = Task(
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "dynamic_tool": {
                "name": "lookup_city",
                "description": "Look up one city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            "toolsets": {"dynamic": {"fn": ref("dynamic_toolset")}},
        }
    ).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    tool_defs = harness.runtime.tool_defs(state)
    assert tool_defs is not None
    assert state["tools"] == ["lookup_city"]
    assert tool_defs[0].name == "lookup_city"
    assert tool_defs[0].parameters["properties"] == {"city": {"type": "string"}}
    assert await state.get_tools()["lookup_city"](city="Paris") == "recorded"
    assert state["dynamic_tool_calls"] == [{"lookup_city": {"city": "Paris"}}]


@pytest.mark.asyncio
async def test_tool_bindings_inject_owner_private_objects() -> None:
    harness = Harness(
        toolsets=[
            Toolset(
                tools=[object_tool],
                objects={"box": load_object_box},
                bindings={"object_tool.box": "objects.box"},
                write=True,
            )
        ]
    )
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    assert await state.get_tools()["object_tool"](value="alpha") == "alpha"


@pytest.mark.asyncio
async def test_rollout_handlers_receive_bound_hidden_args() -> None:
    harness = Harness(
        toolsets=[
            Toolset(
                updates=[update_from_binding],
                bindings={"update_from_binding.expected": "task.answer"},
            )
        ]
    )
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    await harness.runtime.update_rollout(task, state)

    assert state["expected"] == "ok"


@pytest.mark.asyncio
async def test_toolset_bindings_ignore_truthy_handler_markers() -> None:
    harness = Harness(
        toolsets=[
            TruthyMarkerToolset(
                bindings={"fake_handler.expected": "task.answer"},
            )
        ]
    )
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()

    with pytest.raises(ValueError, match="fake_handler.expected"):
        await harness.setup_state(task, State.for_task(task))


@pytest.mark.asyncio
async def test_object_bindings_are_private_to_callable_tools() -> None:
    harness = Harness(
        toolsets=[
            Toolset(
                updates=[update_from_binding],
                objects={"box": load_object_box},
                bindings={"update_from_binding.expected": "objects.box"},
            )
        ]
    )
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    with pytest.raises(ValueError, match="objects"):
        await harness.setup_state(task, State.for_task(task))


@pytest.mark.asyncio
async def test_bindings_must_match_declared_callable_args() -> None:
    harness = Harness(
        toolsets=[
            Toolset(
                tools=[object_tool],
                objects={"box": load_object_box},
                bindings={"object_tool.missing": "objects.box"},
            )
        ]
    )
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()

    with pytest.raises(TypeError, match="missing"):
        await harness.setup_state(task, State.for_task(task))


@pytest.mark.asyncio
async def test_tool_bindings_do_not_leak_to_same_named_handlers() -> None:
    harness = Harness(
        updates=[colliding_update],
        toolsets=[
            Toolset(
                tools=[colliding_tool],
                objects={"token": load_object_box},
                bindings={"colliding_tool.token": "objects.token"},
            )
        ],
    )
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    assert await state.get_tools()["colliding_tool"](value="x") == "{'values': []}:x"
    with pytest.raises(TypeError, match="token"):
        await harness.runtime.update_rollout(task, state)


def test_harness_max_turns_arg_overrides_config() -> None:
    harness = Harness(max_turns=9, config={"max_turns": 3})

    assert harness.config.max_turns == 9


def test_task_prompt_rejects_system_messages() -> None:
    with pytest.raises(ValueError, match="Use system_prompt instead"):
        Task({"prompt": [{"role": "system", "content": "sys"}]}).freeze()


def test_task_system_prompt_is_normalized() -> None:
    task = Task(
        {
            "system_prompt": "sys",
            "prompt": [{"role": "user", "content": "hi"}],
        }
    ).freeze()

    assert task["system_prompt"] == [{"role": "system", "content": "sys"}]
    assert task["prompt"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_harness_resolves_taskset_system_prompt() -> None:
    taskset = Taskset(source=source_loader, system_prompt="taskset sys")
    harness = Harness(program=config_program)
    Env(taskset=taskset, harness=harness)
    task = next(iter(taskset))
    state = await harness.setup_state(task, State.for_task(task))

    assert state["system_prompt"] == [{"role": "system", "content": "taskset sys"}]
    assert state["prompt"] == [{"role": "user", "content": "Say ok."}]


@pytest.mark.asyncio
async def test_harness_rejects_multiple_system_prompt_sources_by_default() -> None:
    taskset = Taskset(source=source_loader, system_prompt="taskset sys")
    harness = Harness(program=config_program, system_prompt="harness sys")
    Env(taskset=taskset, harness=harness)
    task = next(iter(taskset))

    with pytest.raises(ValueError, match="Multiple system_prompt sources"):
        await harness.setup_state(task, State.for_task(task))


@pytest.mark.asyncio
async def test_task_max_turns_overrides_harness_default() -> None:
    harness = Harness(max_turns=9)
    task = Task(
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "max_turns": 3,
        }
    ).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    assert state.get_max_turns(harness.config.max_turns) == 3


@pytest.mark.asyncio
async def test_explicit_state_runtime_max_turns_overrides_task_controls() -> None:
    harness = Harness(max_turns=9)
    task = Task(
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "max_turns": 3,
        }
    ).freeze()
    state = State.for_task(task)
    state["runtime"] = {"max_turns": 2}
    state = await harness.setup_state(task, state)

    assert state.get_max_turns(harness.config.max_turns) == 2


def test_task_runtime_is_not_public_task_schema() -> None:
    with pytest.raises(TypeError, match="task.runtime"):
        Task({"runtime": {"unknown": True}}).freeze()


def test_task_runtime_rejects_legacy_max_turns() -> None:
    with pytest.raises(TypeError, match="task.runtime"):
        Task({"runtime": {"max_turns": "3"}}).freeze()


def test_task_rejects_non_integer_max_turns() -> None:
    with pytest.raises(TypeError):
        Task({"max_turns": "3"}).freeze()


def test_task_sandbox_must_be_mapping() -> None:
    with pytest.raises(TypeError, match="task.sandbox"):
        Task({"prompt": [], "sandbox": "rollout"}).freeze()


def test_option_only_program_requires_sandbox_placement() -> None:
    with pytest.raises(ValueError, match="require sandbox placement"):
        Harness(program={"sandbox": False})

    Harness(program={"sandbox": True}, sandbox={"image": "python:3.11-slim"})


def test_constructor_mapping_args_override_config_mapping_values() -> None:
    harness = Harness(
        sandbox={"image": "constructor", "memory_gb": 8},
        config={"sandbox": {"image": "config", "scope": "group"}},
    )

    assert harness.sandbox is not None
    assert harness.sandbox["image"] == "constructor"
    assert harness.sandbox["memory_gb"] == 8
    assert harness.sandbox["scope"] == "group"


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
async def test_direct_user_callable_receives_default_transcript_binding() -> None:
    harness = Harness(user=direct_user_with_transcript)
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = State.for_task(task)

    messages = await harness.runtime.user_messages(
        task, state, transcript=[{"role": "assistant", "content": "hello"}]
    )

    assert state["direct_transcript_len"] == 1
    assert messages == [{"role": "user", "content": "continue"}]


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
    assert harness.user.sandbox is not None
    assert harness.user.sandbox["image"] == "python:3.11-slim"
    assert harness.user.sandbox["scope"] == "group"
    assert state["sandbox_seen"] is sandbox
    assert messages == [{"role": "user", "content": "sandbox ok"}]


@pytest.mark.asyncio
async def test_configured_program_scores_and_cleans_rollout() -> None:
    taskset = Taskset(source=source_loader)
    harness = Harness(
        config={
            "program": ref("config_program"),
            "rewards": [ref("config_reward")],
            "cleanups": [ref("config_cleanup")],
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
    harness = Harness(program=config_program, cleanups=[config_group_cleanup])
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()

    state = await harness.run(task)

    assert state["group_cleaned"] is True


@pytest.mark.asyncio
async def test_harness_run_defers_group_cleanup_when_group_boundary_exists() -> None:
    harness = Harness(program=config_program, cleanups=[config_group_cleanup])
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

    assert getattr(harness.config, "custom_flag") is True
    assert "custom_flag" in CustomHarness.config_schema()


def test_config_schema_is_visible_from_primary_types() -> None:
    assert "toolsets" in Taskset.config_schema()
    assert "toolsets" in Harness.config_schema()
    assert "source" in TasksetConfig.schema_text()
    assert "eval_source" in TasksetConfig.schema_text()
    assert "program" in HarnessConfig.schema_text()
    assert "image" in vf.SandboxConfig.schema_text()
    assert "bindings" in vf.ToolsetConfig.schema_text()


def test_env_config_normalizes_mapping_config_to_attributes() -> None:
    config = EnvConfig(
        {
            "taskset": {"taskset_id": "dict"},
            "harness": {"model": "configured-model"},
        }
    )

    assert config.taskset == {"taskset_id": "dict"}
    assert config.harness == {"model": "configured-model"}


def test_env_config_rejects_unknown_top_level_sections() -> None:
    with pytest.raises(ValueError):
        EnvConfig({"taskset": {}, "math": {"taskset": {}}})


def test_env_config_requires_child_sections_to_be_configs() -> None:
    with pytest.raises(ValueError):
        EnvConfig({"taskset": 1})


def test_env_config_merges_child_config_defaults_with_nested_sections() -> None:
    class LocalTasksetConfig(TasksetConfig):
        split: str = "train"

    child_config = LocalTasksetConfig({"split": "nested"}, split=None)
    config = EnvConfig(
        {
            "taskset": {"split": "nested"},
            "harness": {"max_turns": 3},
        },
        taskset=LocalTasksetConfig(split="default"),
        harness=HarnessConfig(max_turns=10),
    )
    default_config = EnvConfig(
        {"taskset": {}},
        taskset=LocalTasksetConfig(split="kwarg"),
    )

    assert child_config.split == "nested"
    assert isinstance(config.taskset, LocalTasksetConfig)
    assert config.taskset.split == "nested"
    assert isinstance(config.harness, HarnessConfig)
    assert config.harness.max_turns == 3
    assert isinstance(default_config.taskset, LocalTasksetConfig)
    assert default_config.taskset.split == "kwarg"


def test_load_environment_coerces_typed_env_config_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "typed_env_config"
    module = types.ModuleType(module_name)
    seen: dict[str, object] = {}

    def load_environment(split: str = "train", config: EnvConfig | None = None) -> Env:
        seen["split"] = split
        seen["config"] = config
        config = config or EnvConfig()
        return Env(
            taskset=Taskset(source=source_loader, config=config.taskset),
            harness=Harness(config=config.harness),
        )

    module.load_environment = load_environment
    monkeypatch.setitem(sys.modules, module_name, module)

    env = vf.load_environment(
        "typed-env-config",
        split="test",
        config={
            "taskset": {"taskset_id": "typed"},
            "harness": {"model": "typed-model"},
        },
    )

    assert seen["split"] == "test"
    assert isinstance(seen["config"], EnvConfig)
    assert env.taskset.config.taskset_id == "typed"
    assert env.harness.config.model == "typed-model"
    assert env.env_args == {
        "split": "test",
        "config": {
            "taskset": {"taskset_id": "typed"},
            "harness": {"model": "typed-model"},
        },
    }


def test_load_environment_leaves_untyped_config_arg_as_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "untyped_env_config"
    module = types.ModuleType(module_name)
    seen: dict[str, object] = {}

    def load_environment(split: str = "train", config=None) -> Env:
        seen["split"] = split
        seen["config"] = config
        return Env(taskset=Taskset(source=source_loader))

    module.load_environment = load_environment
    monkeypatch.setitem(sys.modules, module_name, module)

    vf.load_environment(
        "untyped-env-config",
        split="test",
        config={"taskset": {"taskset_id": "raw"}},
    )

    assert seen["split"] == "test"
    assert seen["config"] == {"taskset": {"taskset_id": "raw"}}


def test_config_objects_project_to_base_config_fields() -> None:
    class LocalHarnessConfig(HarnessConfig):
        toolset: object | None = None

    config = LocalHarnessConfig({"model": "parent", "toolset": {"show": ["search"]}})
    harness_config = HarnessConfig(config)

    assert config.toolset == {"show": ["search"]}
    assert harness_config.model == "parent"
    assert not hasattr(harness_config, "toolset")


def test_unset_base_config_defaults_do_not_override_child_defaults() -> None:
    class LocalHarnessConfig(HarnessConfig):
        max_turns: int = 4

    default_config = LocalHarnessConfig(HarnessConfig())
    explicit_config = LocalHarnessConfig(HarnessConfig(max_turns=10))

    assert default_config.max_turns == 4
    assert explicit_config.max_turns == 10


def test_config_field_name_is_allowed_in_typed_configs() -> None:
    class LocalTasksetConfig(TasksetConfig):
        config: dict[str, object] | None = None

    loaded_config = LocalTasksetConfig.from_config({"config": {"mode": "loaded"}})
    direct_config = LocalTasksetConfig(config={"mode": "direct"})
    merged_config = LocalTasksetConfig(
        {"taskset_id": "local"},
        config={"mode": "merged"},
    )

    assert loaded_config.config == {"mode": "loaded"}
    assert direct_config.config == {"mode": "direct"}
    assert merged_config.taskset_id == "local"
    assert merged_config.config == {"mode": "merged"}


@pytest.mark.parametrize(
    "module_name",
    [
        "environments.dspy_flights.dspy_flights",
        "environments.hello_group_reward_v1.hello_group_reward_v1",
        "environments.hello_parallel_sandbox_v1.hello_parallel_sandbox_v1",
        "environments.hello_rlm_v1.hello_rlm_v1",
        "environments.hello_self_judge_v1.hello_self_judge_v1",
        "environments.hello_subagent_v1.hello_subagent_v1",
        "environments.nested_harness_v1.nested_harness_v1",
    ],
)
def test_reference_v1_loaders_preserve_mapping_config_sections(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(module_name)
    env_id = module_name.rsplit(".", 1)[-1]
    monkeypatch.setitem(sys.modules, env_id, module)

    env = vf.load_environment(
        env_id,
        config={
            "taskset": {"taskset_id": "from-env-args"},
            "harness": {"model": "configured-model"},
        },
    )

    assert env.taskset.config.taskset_id == "from-env-args"
    assert env.harness.config.model == "configured-model"


def test_reference_v1_harness_loaders_preserve_child_defaults() -> None:
    group_reward = importlib.import_module(
        "environments.hello_group_reward_v1.hello_group_reward_v1"
    )
    parallel_sandbox = importlib.import_module(
        "environments.hello_parallel_sandbox_v1.hello_parallel_sandbox_v1"
    )
    self_judge = importlib.import_module(
        "environments.hello_self_judge_v1.hello_self_judge_v1"
    )

    assert group_reward.load_harness().config.max_turns == 1
    assert parallel_sandbox.load_harness().config.max_turns == 4
    assert self_judge.load_harness().config.max_turns == 8


def test_bfcl_v1_loader_preserves_mapping_config_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("environments.bfcl_v3.bfcl_v3")
    seen: dict[str, object] = {}

    def fake_taskset(config: object = None, **kwargs: object) -> Taskset:
        _ = kwargs
        seen["taskset_config"] = config
        return Taskset(source=source_loader, config=config)

    def fake_harness(config: object = None, **kwargs: object) -> Harness:
        _ = kwargs
        seen["harness_config"] = config
        return Harness(config=config)

    monkeypatch.setattr(module, "load_taskset", fake_taskset)
    monkeypatch.setattr(module, "load_harness", fake_harness)

    env = module.load_v1_environment(
        config=EnvConfig(
            taskset={"taskset_id": "bfcl-env-args"},
            harness={"model": "bfcl-model"},
        )
    )

    assert env.taskset.config.taskset_id == "bfcl-env-args"
    assert env.harness.config.model == "bfcl-model"
    assert isinstance(seen["taskset_config"], module.BFCLTasksetConfig)
    assert isinstance(seen["harness_config"], module.BFCLHarnessConfig)


def test_tau2_loader_forwards_mapping_harness_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_tau2_import_stubs(monkeypatch)
    module = importlib.import_module("environments.tau2_bench_v1.tau2_bench_v1")
    seen: dict[str, object] = {}

    def fake_taskset(config: object = None) -> Taskset:
        seen["taskset_config"] = config
        return Taskset(source=source_loader)

    monkeypatch.setattr(module, "load_taskset", fake_taskset)

    env = module.load_v1_environment(
        config=EnvConfig(
            taskset={"max_turns": 7},
            harness={"model": "configured-model", "max_turns": 3},
        )
    )

    assert type(env.harness) is Harness
    assert env.harness.config.model == "configured-model"
    assert env.harness.config.max_turns == 3
    assert isinstance(seen["taskset_config"], module.Tau2TasksetConfig)
    assert seen["taskset_config"].max_turns == 7


def install_tau2_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = [
        "tau2",
        "tau2.agent",
        "tau2.agent.llm_agent",
        "tau2.config",
        "tau2.data_model",
        "tau2.data_model.message",
        "tau2.data_model.simulation",
        "tau2.data_model.tasks",
        "tau2.environment",
        "tau2.environment.environment",
        "tau2.evaluator",
        "tau2.evaluator.evaluator",
        "tau2.orchestrator",
        "tau2.orchestrator.orchestrator",
        "tau2.registry",
        "tau2.run",
        "tau2.user",
        "tau2.user.user_simulator",
        "tau2.utils",
        "tau2.utils.utils",
    ]
    for module_name in modules:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    llm_agent = sys.modules["tau2.agent.llm_agent"]
    llm_agent.AGENT_INSTRUCTION = "agent"
    llm_agent.SYSTEM_PROMPT = "{agent_instruction} {domain_policy}"
    llm_agent.LLMAgent = type("LLMAgent", (), {})
    llm_agent.is_valid_agent_history_message = lambda message: True

    config = sys.modules["tau2.config"]
    config.DEFAULT_LLM_ARGS_AGENT = {}
    config.DEFAULT_LLM_ARGS_USER = {}
    config.DEFAULT_MAX_ERRORS = 10
    config.DEFAULT_MAX_STEPS = 20

    message = sys.modules["tau2.data_model.message"]
    for name in (
        "AssistantMessage",
        "Message",
        "MultiToolMessage",
        "ToolCall",
        "ToolMessage",
        "UserMessage",
    ):
        setattr(message, name, type(name, (), {}))

    simulation = sys.modules["tau2.data_model.simulation"]
    simulation.SimulationRun = type("SimulationRun", (), {})
    simulation.TerminationReason = type(
        "TerminationReason",
        (),
        {
            "AGENT_ERROR": "agent_error",
            "AGENT_STOP": "agent_stop",
            "MAX_STEPS": "max_steps",
            "TOO_MANY_ERRORS": "too_many_errors",
            "USER_STOP": "user_stop",
        },
    )

    tasks = sys.modules["tau2.data_model.tasks"]
    tasks.Task = type("Task", (), {})

    environment = sys.modules["tau2.environment.environment"]
    environment.Environment = type("Environment", (), {})

    evaluator = sys.modules["tau2.evaluator.evaluator"]
    evaluator.EvaluationType = type("EvaluationType", (), {"ALL": "all"})
    evaluator.evaluate_simulation = lambda **kwargs: None

    orchestrator = sys.modules["tau2.orchestrator.orchestrator"]
    orchestrator.DEFAULT_FIRST_AGENT_MESSAGE = object()
    orchestrator.Role = type(
        "Role", (), {"AGENT": "agent", "ENV": "env", "USER": "user"}
    )

    registry_module = sys.modules["tau2.registry"]
    registry_module.registry = type(
        "Registry", (), {"get_env_constructor": lambda *args: None}
    )()

    run = sys.modules["tau2.run"]
    run.load_tasks = lambda *args, **kwargs: []

    user_simulator = sys.modules["tau2.user.user_simulator"]
    user_simulator.UserSimulator = type("UserSimulator", (), {})
    user_simulator.is_valid_user_history_message = lambda message: True

    utils = sys.modules["tau2.utils.utils"]
    utils.DATA_DIR = "unused"
    utils.format_time = lambda value: str(value)
    utils.get_now = lambda: "now"


def test_self_judge_loader_projects_shortcuts_to_child_configs() -> None:
    module = importlib.import_module(
        "environments.hello_self_judge_v1.hello_self_judge_v1"
    )

    taskset = module.load_taskset(num_examples=2)
    harness = module.load_harness(max_turns=3)
    shortcut_env = module.load_environment(num_examples=2, max_turns=3)
    override_env = module.load_environment(
        num_examples=2,
        max_turns=3,
        config=EnvConfig(
            taskset={"num_examples": 1},
            harness={"max_turns": 5},
        ),
    )

    assert len(taskset.rows()) == 2
    assert harness.config.max_turns == 3
    assert len(shortcut_env.taskset.rows()) == 2
    assert shortcut_env.harness.config.max_turns == 3
    assert len(override_env.taskset.rows()) == 1
    assert override_env.harness.config.max_turns == 5


def test_subagent_loader_keeps_child_harness_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("environments.hello_subagent_v1.hello_subagent_v1")
    monkeypatch.setitem(sys.modules, "hello_subagent_v1", module)

    env = vf.load_environment(
        "hello-subagent-v1", config={"harness": {"model": "parent"}}
    )

    assert env.harness.config.model == "parent"
    child_harness = env.harness.toolsets[0].bindings["ask_subagent.harness"]
    assert child_harness.config.model is None


def test_nested_configs_validate_and_feed_runtime_objects() -> None:
    sandbox = vf.SandboxConfig(
        image="python:3.12-slim",
        packages="numpy",
        setup_commands="echo ready",
        scope="group",
    )
    harness = Harness(program={"sandbox": True}, sandbox=sandbox)

    assert harness.sandbox is not None
    assert harness.sandbox["image"] == "python:3.12-slim"
    assert harness.sandbox["packages"] == ["numpy"]
    assert harness.sandbox["setup_commands"] == ["echo ready"]
    assert harness.sandbox["scope"] == "group"

    toolset = Toolset(
        config=vf.ToolsetConfig(
            tools=[ref("hidden_tool")],
            show="hidden_tool",
            sandbox={"prefer": "program"},
            write=True,
        )
    )

    assert toolset.tools == (hidden_tool,)
    assert toolset.show == ("hidden_tool",)
    assert toolset.write is True
    assert isinstance(toolset.sandbox, Mapping)
    assert toolset.sandbox["prefer"] == "program"


def test_nested_configs_reject_unknown_fields() -> None:
    with pytest.raises(ValueError):
        vf.SandboxConfig.model_validate({"image": "python:3.11", "unknown": True})

    with pytest.raises(ValueError):
        vf.ToolsetConfig.model_validate({"tools": [], "show": ["a"], "hide": ["b"]})


def test_configs_load_from_toml_sections(tmp_path) -> None:
    config_path = tmp_path / "env.toml"
    config_path.write_text(
        "\n".join(
            [
                "[env.taskset]",
                f'source = "{ref("source_loader")}"',
                "",
                "[[env.taskset.rewards]]",
                f'fn = "{ref("config_reward")}"',
                "weight = 0.5",
                "",
                "[env.taskset.toolsets.configured]",
                f'fn = "{ref("config_toolset")}"',
                'prefix = "toml"',
                "",
                "[env.harness]",
                "max_turns = 7",
                "",
                "[env.harness.program]",
                f'fn = "{ref("config_program")}"',
            ]
        )
    )

    taskset_config = TasksetConfig.from_toml(config_path, "env.taskset")
    harness_config = HarnessConfig.from_toml(config_path, ("env", "harness"))

    taskset = Taskset(config=taskset_config)
    harness = Harness(config=harness_config)

    assert taskset.source is source_loader
    assert getattr(taskset.rewards[0], "__name__") == "config_reward"
    assert getattr(taskset.rewards[0], "reward_weight") == 0.5
    assert taskset.named_toolsets["configured"].bindings == {
        "config_tool.prefix": "toml"
    }
    assert harness.program == {"fn": ref("config_program")}
    assert callable(harness._program)
    assert harness.config.max_turns == 7


@pytest.mark.asyncio
async def test_task_tools_filter_exposed_tools() -> None:
    harness = Harness(toolsets=[Toolset(tools=[direct_tool, hidden_tool])])
    task = Task(
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "tools": {"show": ["direct_tool"]},
        }
    ).freeze()
    state = await harness.setup_state(task, State.for_task(task))

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
            "cleanups": [ref("config_cleanup")],
        },
    )

    assert toolset.tools == (direct_tool, hidden_tool)
    assert toolset.bindings == {"hidden_tool.prefix": "task.answer"}
    assert toolset.objects == {"source": source_loader}
    assert toolset.write is True
    assert toolset.scope == "group"
    assert toolset.cleanups == (config_cleanup,)


def test_toolset_write_arg_overrides_config() -> None:
    toolset = Toolset(write=False, config={"write": True})

    assert toolset.write is False


def test_toolset_sandbox_prefer_requires_program() -> None:
    with pytest.raises(ValueError, match="sandbox.prefer must be 'program'"):
        Toolset(sandbox={"prefer": "other"})


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

    taskset.add_toolset({"direct": {"tools": [direct_tool]}})
    harness.add_toolset({"configured": config_toolset})

    assert taskset.named_toolsets["direct"].tools == (direct_tool,)
    assert harness.named_toolsets["configured"].tools == (config_tool,)


def test_taskset_extension_refreshes_attached_harness_runtime() -> None:
    taskset = Taskset(source=source_loader)
    harness = Harness()
    harness.attach_taskset(taskset)

    taskset.add_toolset({"direct": {"tools": [direct_tool]}})

    assert "direct" in harness.runtime.named_toolsets

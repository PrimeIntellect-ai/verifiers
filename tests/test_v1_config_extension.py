from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from typing import Any, cast

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
    assert harness.cleanups == [config_cleanup]
    assert harness.user is not None
    assert len(harness.toolsets) == 2
    assert harness.toolsets[0] is direct_toolset
    assert harness.toolsets[1].hide == ("config_tool",)


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
    assert await state.tools()["config_tool"](query="q") == "ok:q"


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
    assert await state.tools()["lookup_city"](city="Paris") == "recorded"
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

    assert await state.tools()["object_tool"](value="alpha") == "alpha"


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

    assert await state.tools()["colliding_tool"](value="x") == "{'values': []}:x"
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

    assert harness.max_turns(state) == 3


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

    assert harness.max_turns(state) == 2


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
    assert harness._program is config_program
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

import importlib
import sys
import types
from collections.abc import Mapping
from typing import Any, cast

import pytest
from pydantic import BaseModel

import verifiers as vf
from verifiers.v1 import (
    Config,
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
from verifiers.v1.toolset import normalize_toolset
from verifiers.v1.utils.config_utils import config_data as v1_config_data
from verifiers.v1.utils.taskset_utils import rows_from_source, source_config_args


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


def positional_only_source(
    config: str = "positional-default", /, split: str = "train"
) -> list[dict[str, object]]:
    return [{"config": config, "split": split}]


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


@vf.update(stage="group")
async def group_update_from_binding(
    tasks: list[Mapping[str, object]], states: list[dict[str, object]], expected: str
) -> None:
    _ = tasks
    for state in states:
        state["group_expected"] = expected


@vf.reward
async def reward_from_binding(
    task: Mapping[str, object], state: dict[str, object], expected: str
) -> float:
    _ = state
    return float(task.get("answer") == expected)


@vf.reward(stage="group")
async def group_reward_from_binding(
    tasks: list[Mapping[str, object]], states: list[dict[str, object]], expected: str
) -> list[float]:
    _ = tasks
    return [float(state.get("answer") == expected) for state in states]


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


async def setup_aware_program(
    task: Mapping[str, object], state: dict[str, object]
) -> dict[str, object]:
    _ = task
    if state.get("configured_setup") is not True:
        raise AssertionError("setup did not run before program")
    return {"program_saw_setup": True}


def config_toolset(prefix: str = "cfg") -> Toolset:
    def prefix_value() -> str:
        return prefix

    return Toolset(
        tools=[config_tool],
        bindings={"config_tool.prefix": prefix_value},
    )


def load_another_harness_config() -> HarnessConfig:
    return HarnessConfig(max_turns=6, rewards=[config_reward])


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
setattr(ref_module, "load_another_harness_config", load_another_harness_config)
sys.modules[REF_MODULE] = ref_module


def test_source_config_args_preserve_config_object_aliases() -> None:
    raw_config = {
        "config": "shadowed",
        "taskset_config": "also-shadowed",
        "split": "eval",
    }

    args = source_config_args(raw_config)

    assert args["config"] is raw_config
    assert args["taskset_config"] is raw_config
    assert args["split"] == "eval"


def test_config_data_recursively_dumps_nested_configs_without_none() -> None:
    class NestedConfig(Config):
        sandbox: vf.SandboxConfig | None = None

    class OuterConfig(Config):
        nested: NestedConfig = NestedConfig()
        label: str | None = "default"

    config = OuterConfig(
        nested=NestedConfig(
            sandbox=vf.SandboxConfig(image="python:3.12-slim", workdir=None)
        ),
        label=None,
    )

    assert v1_config_data(config) == {
        "nested": {"sandbox": {"image": "python:3.12-slim"}}
    }


def test_source_loader_filters_positional_only_config_names() -> None:
    rows = rows_from_source(
        positional_only_source,
        {"config": "shadowed", "split": "eval"},
    )

    assert rows == [{"config": "positional-default", "split": "eval"}]


def test_toolset_mapping_treats_show_hide_strings_as_tool_names() -> None:
    shown = normalize_toolset(
        {
            "tools": [direct_tool, hidden_tool],
            "show": "direct_tool",
        }
    )
    hidden = normalize_toolset(
        {
            "tools": [direct_tool, hidden_tool],
            "hide": "hidden_tool",
        }
    )

    assert shown.show == ("direct_tool",)
    assert hidden.hide == ("hidden_tool",)


def ref(name: str) -> str:
    return f"{REF_MODULE}:{name}"


def dynamic_ref(value: object) -> str:
    name = getattr(value, "__name__", type(value).__name__)
    ref_name = f"{name}_{id(value)}"
    setattr(ref_module, ref_name, value)
    return ref(ref_name)


def config_value(value: object) -> object:
    if callable(value):
        return dynamic_ref(value)
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, Mapping):
        return {str(key): config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [config_value(item) for item in value]
    if isinstance(value, tuple):
        return [config_value(item) for item in value]
    return value


def config_data(config: object | None) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        return config.model_dump(exclude_none=True)
    if isinstance(config, Mapping):
        return {str(key): item for key, item in config.items()}
    raise TypeError("test config must be a mapping or config object")


def has_runtime_toolset(value: object) -> bool:
    if isinstance(value, Toolset):
        return True
    if isinstance(value, Mapping):
        return any(has_runtime_toolset(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(has_runtime_toolset(item) for item in value)
    return False


def make_taskset(config: object | None = None, **values: object) -> Taskset:
    data = {**config_data(config), **values}
    runtime_toolsets = data.pop("toolsets", None)
    config_type = type(config) if isinstance(config, TasksetConfig) else TasksetConfig
    if runtime_toolsets is not None and not has_runtime_toolset(runtime_toolsets):
        data["toolsets"] = runtime_toolsets
        runtime_toolsets = None
    taskset = Taskset(config=config_type.model_validate(config_value(data)))
    if runtime_toolsets is not None:
        taskset.add_toolset(runtime_toolsets)
    return taskset


def make_harness(config: object | None = None, **values: object) -> Harness:
    data = {**config_data(config), **values}
    runtime_client = data.pop("client", None)
    runtime_toolsets = data.pop("toolsets", None)
    config_type = type(config) if isinstance(config, HarnessConfig) else HarnessConfig
    if runtime_toolsets is not None and not has_runtime_toolset(runtime_toolsets):
        data["toolsets"] = runtime_toolsets
        runtime_toolsets = None
    harness = Harness(config=config_type.model_validate(config_value(data)))
    if runtime_client is not None:
        harness.client = runtime_client
    if runtime_toolsets is not None:
        harness.add_toolset(runtime_toolsets)
    return harness


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
    taskset = make_taskset(source=source_loader, eval_source=eval_source_loader)

    assert taskset.get_dataset()[0]["answer"] == "ok"
    assert taskset.get_eval_dataset()[0]["answer"] == "eval ok"


def test_env_passes_taskset_eval_dataset_to_environment() -> None:
    env = Env(
        taskset=make_taskset(source=source_loader, eval_source=eval_source_loader),
        harness=make_harness(program=config_program),
    )

    assert env.get_dataset()[0]["answer"] == "ok"
    assert env.get_eval_dataset()[0]["answer"] == "eval ok"


def test_env_defaults_to_base_harness() -> None:
    taskset = make_taskset(source=source_loader)
    env = Env(taskset=taskset)

    assert isinstance(env.harness, Harness)
    assert env.harness.taskset is taskset
    assert env.get_dataset()[0]["answer"] == "ok"


def test_env_capabilities_follow_v1_group_runtime_signals() -> None:
    rollout_env = Env(
        taskset=make_taskset(source=source_loader, rewards=[config_reward]),
        harness=make_harness(program=config_program),
    )
    group_metric_env = Env(
        taskset=make_taskset(source=source_loader, metrics=[group_config_metric]),
        harness=make_harness(program=config_program),
    )
    group_reward_env = Env(
        taskset=make_taskset(source=source_loader, rewards=[group_config_reward]),
        harness=make_harness(program=config_program),
    )
    advantage_env = Env(
        taskset=make_taskset(source=source_loader, advantages=[config_advantage]),
        harness=make_harness(program=config_program),
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
        taskset=make_taskset(source=source_loader, updates=[config_group_update]),
        harness=make_harness(program=config_program),
    )
    group_cleanup_env = Env(
        taskset=make_taskset(source=source_loader, cleanups=[config_group_cleanup]),
        harness=make_harness(program=config_program),
    )

    assert group_update_env.requires_group_rollouts
    assert not group_update_env.provides_advantages
    assert group_cleanup_env.requires_group_rollouts
    assert not group_cleanup_env.provides_advantages


@pytest.mark.asyncio
async def test_group_lifecycle_handlers_require_bound_extra_args() -> None:
    @vf.update(stage="group")
    async def bad_group_update(tasks, states, extra) -> None:
        _ = tasks, states, extra

    env = Env(
        taskset=make_taskset(source=source_loader, updates=[bad_group_update]),
        harness=make_harness(program=config_program),
    )
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = State.for_task(task)

    with pytest.raises(TypeError, match="extra"):
        await env.harness.runtime.update_group([task], [state])


def test_env_capabilities_follow_custom_taskset_init_group() -> None:
    class GroupSetupTaskset(Taskset):
        async def init_group(
            self, task: Task, num_rollouts: int
        ) -> tuple[list[Task], list[State]]:
            return await super().init_group(task, num_rollouts)

    env = Env(
        taskset=GroupSetupTaskset(
            config=TasksetConfig(source=dynamic_ref(source_loader))
        ),
        harness=make_harness(program=config_program),
    )

    assert env.requires_group_rollouts
    assert not env.provides_advantages


def test_harness_config_extends_constructor_surface() -> None:
    direct_toolset = Toolset(tools=[direct_tool])
    harness = make_harness(
        config={
            "program": ref("config_program"),
            "metrics": [ref("config_metric")],
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
    harness.add_toolset(direct_toolset)

    assert harness.program is config_program
    assert harness.config.max_turns == 3
    assert [metric.__name__ for metric in harness.metrics] == [
        "num_turns",
        "config_metric",
    ]
    assert harness.rewards == [config_reward]
    assert harness.advantages == [config_advantage]
    assert harness.setups == [config_setup]
    assert harness.cleanups == [config_cleanup]
    assert harness.user is not None
    assert len(harness.toolsets) == 2
    assert harness.toolsets[0].hide == ("config_tool",)
    assert harness.toolsets[1] is direct_toolset


def test_harness_owns_default_render_completion_update() -> None:
    harness = make_harness(program=config_program)

    assert any(
        getattr(handler, "__self__", None) is harness
        and getattr(handler, "__name__", "") == "render_completion"
        for handler in harness.runtime.rollout_update
    )


def test_harness_owns_default_num_turns_metric() -> None:
    harness = make_harness(program=config_program)

    assert any(
        signal["name"] == "num_turns" for signal in harness.runtime.rollout_signals
    )


@pytest.mark.asyncio
async def test_update_config_runs_before_rollout_scoring() -> None:
    harness = make_harness(
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
    harness = make_harness(
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
    taskset = make_taskset(source=source_loader, setups=[config_setup])
    harness = make_harness(program=setup_aware_program)
    Env(taskset=taskset, harness=harness)
    task = next(iter(taskset))

    state = await harness.run(task)

    assert state["program_saw_setup"] is True
    assert state["setup_order"] == ["config_setup"]


@pytest.mark.asyncio
async def test_group_update_config_runs_before_group_scoring() -> None:
    harness = make_harness(
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
    assert vf.State is State

    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = vf.State.for_task(task)
    assert state.uses_v1_contract is True

    for key, value in {
        "is_completed": True,
        "stop_condition": "done",
        "is_truncated": True,
        "error": {"message": "boom"},
    }.items():
        assert State({key: value})[key] == value
        with pytest.raises(RuntimeError, match="framework-managed"):
            state[key] = value
        with pytest.raises(RuntimeError, match="framework-managed"):
            state.update({key: value})
        with pytest.raises(RuntimeError, match="framework-managed"):
            state.setdefault(key, value)
        with pytest.raises(RuntimeError, match="framework-managed"):
            state.pop(key)
    state["user_field"] = "ok"
    assert state.popitem() == ("user_field", "ok")

    protected_only = State()._enable_v1_contract()
    protected_only._set_completed(False)
    protected_only._set_stop_condition(None, overwrite=True)
    protected_only._set_truncated(False, overwrite=True)
    protected_only._set_error(None)
    with pytest.raises(RuntimeError, match="framework-managed"):
        protected_only.popitem()
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
    taskset = make_taskset(
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
    prefix = taskset.toolsets[1].bindings["config_tool.prefix"]
    assert callable(prefix)
    assert prefix() == "from_config"


@pytest.mark.asyncio
async def test_task_toolsets_show_hide_selects_named_defaults() -> None:
    harness = make_harness(
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
    harness = make_harness()
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
    harness = make_harness()
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
    harness = make_harness(
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


def test_binding_strings_must_be_framework_paths() -> None:
    with pytest.raises(ValueError, match="Binding string sources"):
        Toolset(tools=[config_tool], bindings={"config_tool.prefix": "literal"})


def test_binding_sources_reject_direct_objects() -> None:
    with pytest.raises(TypeError, match="framework path or callable"):
        Toolset(tools=[config_tool], bindings={"config_tool.prefix": object()})


def test_toolset_binding_keys_must_target_callable_args() -> None:
    with pytest.raises(ValueError, match="callable.arg"):
        Toolset(tools=[config_tool], bindings={"prefix": "task.answer"})


@pytest.mark.asyncio
async def test_rollout_handlers_receive_bound_hidden_args() -> None:
    harness = make_harness(
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
async def test_harness_handlers_receive_bound_hidden_args() -> None:
    harness = make_harness(
        updates=[update_from_binding],
        bindings={"update_from_binding.expected": "task.answer"},
    )
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    await harness.runtime.update_rollout(task, state)

    assert state["expected"] == "ok"


@pytest.mark.asyncio
async def test_taskset_handlers_receive_bound_hidden_args() -> None:
    taskset = make_taskset(
        updates=[update_from_binding],
        bindings={"update_from_binding.expected": "task.answer"},
    )
    harness = make_harness()
    harness.attach_taskset(taskset)
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    await harness.runtime.update_rollout(task, state)

    assert state["expected"] == "ok"


@pytest.mark.asyncio
async def test_group_handlers_receive_bound_hidden_args() -> None:
    harness = make_harness(
        updates=[group_update_from_binding],
        bindings={"group_update_from_binding.expected": "tasks.0.answer"},
    )
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()
    state = State.for_task(task)

    await harness.runtime.update_group([task], [state])

    assert state["group_expected"] == "ok"


@pytest.mark.asyncio
async def test_signals_receive_bound_hidden_args() -> None:
    harness = make_harness(
        rewards=[reward_from_binding],
        bindings={"reward_from_binding.expected": "task.answer"},
    )
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()
    state = await harness.setup_state(task, State.for_task(task))

    await harness.runtime.score_rollout(task, state)

    assert state["reward"] == 1.0
    assert state["metrics"]["reward_from_binding"] == 1.0


@pytest.mark.asyncio
async def test_group_signals_receive_bound_hidden_args() -> None:
    harness = make_harness(
        rewards=[group_reward_from_binding],
        bindings={"group_reward_from_binding.expected": "states.0.answer"},
    )
    task = Task({"prompt": [{"role": "user", "content": "hi"}]}).freeze()
    state = State.for_task(task)
    state["answer"] = "ok"

    await harness.runtime.score_group([task], [state])

    assert state["reward"] == 1.0
    assert state["metrics"]["group_reward_from_binding"] == 1.0


@pytest.mark.asyncio
async def test_object_bindings_are_private_to_callable_tools() -> None:
    harness = make_harness(
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
    harness = make_harness(
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
    harness = make_harness(
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


def test_constructor_scalar_kwargs_are_removed() -> None:
    with pytest.raises(TypeError):
        Harness(max_turns=9)
    with pytest.raises(TypeError):
        Taskset(source=ref("source_loader"))
    with pytest.raises(TypeError):
        vf.OpenCode(max_turns=9)


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
    taskset = make_taskset(source=source_loader, system_prompt="taskset sys")
    harness = make_harness(program=config_program)
    Env(taskset=taskset, harness=harness)
    task = next(iter(taskset))
    state = await harness.setup_state(task, State.for_task(task))

    assert state["system_prompt"] == [{"role": "system", "content": "taskset sys"}]
    assert state["prompt"] == [{"role": "user", "content": "Say ok."}]


@pytest.mark.asyncio
async def test_harness_rejects_multiple_system_prompt_sources_by_default() -> None:
    taskset = make_taskset(source=source_loader, system_prompt="taskset sys")
    harness = make_harness(program=config_program, system_prompt="harness sys")
    Env(taskset=taskset, harness=harness)
    task = next(iter(taskset))

    with pytest.raises(ValueError, match="Multiple system_prompt sources"):
        await harness.setup_state(task, State.for_task(task))


@pytest.mark.asyncio
async def test_task_max_turns_overrides_harness_default() -> None:
    harness = make_harness(max_turns=9)
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
    harness = make_harness(max_turns=9)
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
        make_harness(program={"sandbox": False})

    make_harness(program={"sandbox": True}, sandbox={"image": "python:3.11-slim"})


def test_harness_config_sandbox_values_live_only_in_config() -> None:
    harness = make_harness(
        config={"sandbox": {"image": "configured", "memory_gb": 8, "scope": "group"}},
    )

    assert harness.sandbox is not None
    assert harness.sandbox["image"] == "configured"
    assert harness.sandbox["memory_gb"] == 8
    assert harness.sandbox["scope"] == "group"


@pytest.mark.asyncio
async def test_user_config_supports_scope_bindings_and_objects() -> None:
    harness = make_harness(
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
    harness = make_harness(user=direct_user_with_transcript)
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
    harness = make_harness(
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
    taskset = make_taskset(source=source_loader)
    harness = make_harness(
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
    harness = make_harness(program=config_program, cleanups=[config_group_cleanup])
    task = Task(
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "ok"}
    ).freeze()

    state = await harness.run(task)

    assert state["group_cleaned"] is True


@pytest.mark.asyncio
async def test_harness_run_defers_group_cleanup_when_group_boundary_exists() -> None:
    harness = make_harness(program=config_program, cleanups=[config_group_cleanup])
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


def test_config_annotation_only_nested_config_defaults_recursively() -> None:
    class LeafConfig(Config):
        value: int = 1

    class ChildConfig(Config):
        leaf: LeafConfig = LeafConfig()

    class ParentConfig(Config):
        child: ChildConfig = ChildConfig()

    first = ParentConfig()
    second = ParentConfig()
    configured = ParentConfig.model_validate({"child": {"leaf": {"value": 3}}})

    assert isinstance(first.child, ChildConfig)
    assert isinstance(first.child.leaf, LeafConfig)
    assert first.child.leaf.value == 1
    assert first.child is not second.child
    assert first.child.leaf is not second.child.leaf
    assert configured.child.leaf.value == 3
    assert "child: ChildConfig = ChildConfig" in ParentConfig.schema_text()


def test_env_config_normalizes_mapping_config_to_attributes() -> None:
    config = EnvConfig.model_validate(
        {
            "taskset": {"taskset_id": "dict"},
            "harness": {"model": "configured-model"},
        }
    )

    assert isinstance(config.taskset, TasksetConfig)
    assert isinstance(config.harness, HarnessConfig)
    assert config.taskset.taskset_id == "dict"
    assert config.harness.model == "configured-model"


def test_env_config_defaults_taskset_and_harness_to_base_configs() -> None:
    config = EnvConfig()

    assert isinstance(config.taskset, TasksetConfig)
    assert isinstance(config.harness, HarnessConfig)


def test_env_config_rejects_unknown_top_level_sections() -> None:
    with pytest.raises(ValueError):
        EnvConfig.model_validate({"taskset": {}, "math": {"taskset": {}}})


def test_env_config_requires_child_sections_to_be_configs() -> None:
    with pytest.raises(ValueError):
        EnvConfig.model_validate({"taskset": 1})
    with pytest.raises(ValueError, match="EnvConfig.taskset cannot be None"):
        EnvConfig.model_validate({"taskset": None})
    with pytest.raises(ValueError, match="EnvConfig.harness cannot be None"):
        EnvConfig(harness=None)


def test_env_config_child_config_objects_must_match_domain() -> None:
    class LocalTasksetConfig(TasksetConfig):
        split: str = "train"

    class LocalHarnessConfig(HarnessConfig):
        mode: str = "default"

    config = EnvConfig(
        taskset=LocalTasksetConfig(split="test"),
        harness=LocalHarnessConfig(mode="custom"),
    )

    assert isinstance(config.taskset, LocalTasksetConfig)
    assert isinstance(config.harness, LocalHarnessConfig)

    class LocalConfig(Config):
        split: str = "train"

    with pytest.raises(ValueError):
        EnvConfig(taskset=LocalConfig())
    with pytest.raises(ValueError):
        EnvConfig(harness=LocalConfig())


def test_env_config_validates_nested_sections_into_annotated_child_types() -> None:
    class LocalTasksetConfig(TasksetConfig):
        split: str = "train"

    class LocalEnvConfig(EnvConfig):
        taskset: LocalTasksetConfig = LocalTasksetConfig()
        harness: HarnessConfig = HarnessConfig(max_turns=10)

    config = LocalEnvConfig.model_validate(
        {"taskset": {"split": "nested"}, "harness": {"max_turns": 3}}
    )
    default_config = LocalEnvConfig()

    assert isinstance(config.taskset, LocalTasksetConfig)
    assert config.taskset.split == "nested"
    assert isinstance(config.harness, HarnessConfig)
    assert config.harness.max_turns == 3
    assert isinstance(default_config.taskset, LocalTasksetConfig)
    assert default_config.taskset.split == "train"
    assert default_config.harness.max_turns == 10


def test_config_model_validate_keeps_serializable_nested_values() -> None:
    config = HarnessConfig.model_validate(
        {
            "sampling_args": {
                "temperature": 0.7,
                "extra_body": {
                    "top_p": None,
                    "min_p": 0.05,
                },
                "stop": [None, "DONE"],
            }
        }
    )

    assert config.sampling_args == {
        "temperature": 0.7,
        "extra_body": {
            "top_p": None,
            "min_p": 0.05,
        },
        "stop": [None, "DONE"],
    }


def test_config_rejects_live_python_objects() -> None:
    with pytest.raises(ValueError):
        HarnessConfig(
            program=config_program,
        )
    with pytest.raises(ValueError):
        TasksetConfig(
            source=source_loader,
        )


def test_config_json_round_trip_preserves_values() -> None:
    config = HarnessConfig(
        sampling_args={
            "extra_body": {
                "min_p": 0.05,
            },
            "stop": [None, "DONE"],
        },
        program=vf.ProgramConfig(fn=ref("config_program")),
    )

    dump = config.model_dump(mode="json", exclude_none=True)

    assert HarnessConfig.model_validate(dump) == config
    assert dump["program"]["fn"] == ref("config_program")
    assert dump["program"]["files"] == {}
    assert dump["program"]["setup"] == []


def test_env_config_subclasses_cannot_define_root_fields() -> None:
    with pytest.raises(TypeError, match="unsupported root env config fields"):

        class LocalEnvConfig(EnvConfig):
            split: str = "train"


def test_env_config_subclasses_must_use_domain_child_configs() -> None:
    class LocalConfig(Config):
        split: str = "train"

    with pytest.raises(TypeError, match="taskset must be typed"):

        class LocalEnvConfig(EnvConfig):
            taskset: LocalConfig


def test_env_config_rejects_legacy_config_ref_merging() -> None:
    with pytest.raises(ValueError):
        EnvConfig.model_validate(
            {
                "harness": {
                    "config": ref("load_another_harness_config"),
                    "rewards": [{"fn": ref("updated_reward"), "weight": 0}],
                }
            }
        )


def test_harness_config_normalizes_program_mapping() -> None:
    config = HarnessConfig(
        program={
            "command": ["echo", "ok"],
            "sandbox": {"packages": "numpy"},
            "channels": {"mcp": True},
        }
    )

    assert isinstance(config.program, vf.ProgramConfig)
    assert config.program.command == ["echo", "ok"]
    assert isinstance(config.program.sandbox, vf.SandboxConfig)
    assert config.program.sandbox.packages == ["numpy"]
    assert config.program.channels == {"mcp": True}


def test_harness_config_rejects_unknown_program_tool_interface() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        HarnessConfig(program={"command": ["echo"], "channels": {"ptc": True}})


def test_load_environment_coerces_typed_env_config_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "typed_env_config"
    module = types.ModuleType(module_name)
    seen: dict[str, object] = {}

    def load_environment(split: str = "train", *, config: EnvConfig) -> Env:
        seen["split"] = split
        seen["config"] = config
        return Env(
            taskset=make_taskset(source=source_loader, config=config.taskset),
            harness=make_harness(config=config.harness),
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


def test_load_environment_coerces_env_config_subclass_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "typed_env_config_subclass"
    module = types.ModuleType(module_name)
    seen: dict[str, object] = {}

    class LocalTasksetConfig(TasksetConfig):
        split: str = "train"

    class LocalHarnessConfig(HarnessConfig):
        mode: str = "default"

    class LocalEnvConfig(EnvConfig):
        taskset: LocalTasksetConfig
        harness: LocalHarnessConfig

    class LocalTaskset(Taskset):
        config_type = LocalTasksetConfig

    class LocalHarness(Harness):
        config_type = LocalHarnessConfig

    def load_environment(config: LocalEnvConfig) -> Env:
        seen["config"] = config
        taskset_config = LocalTasksetConfig.model_validate(
            {**config.taskset.model_dump(), "source": ref("source_loader")}
        )
        return Env(
            taskset=LocalTaskset(config=taskset_config),
            harness=LocalHarness(config=config.harness),
        )

    module.load_environment = load_environment
    monkeypatch.setitem(sys.modules, module_name, module)

    env = vf.load_environment(
        "typed-env-config-subclass",
        config={
            "taskset": {"taskset_id": "typed", "split": "test"},
            "harness": {"mode": "custom"},
        },
    )
    config = seen["config"]

    assert isinstance(config, LocalEnvConfig)
    assert isinstance(config.taskset, LocalTasksetConfig)
    assert isinstance(config.harness, LocalHarnessConfig)
    assert env.taskset.config.taskset_id == "typed"
    assert env.taskset.config.split == "test"
    assert env.harness.config.mode == "custom"


def test_load_environment_supplies_default_typed_env_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "default_typed_env_config"
    module = types.ModuleType(module_name)
    seen: dict[str, object] = {}

    def load_environment(config: EnvConfig) -> Env:
        seen["config"] = config
        return Env(
            taskset=make_taskset(source=source_loader, config=config.taskset),
            harness=make_harness(config=config.harness),
        )

    module.load_environment = load_environment
    monkeypatch.setitem(sys.modules, module_name, module)

    env = vf.load_environment("default-typed-env-config")

    assert isinstance(seen["config"], EnvConfig)
    assert env.env_args == {}


def test_load_environment_leaves_untyped_config_arg_as_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "untyped_env_config"
    module = types.ModuleType(module_name)
    seen: dict[str, object] = {}

    def load_environment(split: str = "train", config=None) -> Env:
        seen["split"] = split
        seen["config"] = config
        return Env(taskset=make_taskset(source=source_loader))

    module.load_environment = load_environment
    monkeypatch.setitem(sys.modules, module_name, module)

    vf.load_environment(
        "untyped-env-config",
        split="test",
        config={"taskset": {"taskset_id": "raw"}},
    )

    assert seen["split"] == "test"
    assert seen["config"] == {"taskset": {"taskset_id": "raw"}}


def test_config_objects_are_strict_when_projected_to_base_config_fields() -> None:
    class LocalHarnessConfig(HarnessConfig):
        toolset: dict[str, object] | None = None

    config = LocalHarnessConfig(
        model="parent",
        toolset={"show": ["search"]},
    )

    assert config.toolset == {"show": ["search"]}
    with pytest.raises(ValueError):
        HarnessConfig.model_validate(config.model_dump())


def test_unset_base_config_defaults_do_not_override_child_defaults() -> None:
    class LocalHarnessConfig(HarnessConfig):
        max_turns: int = 4

    default_config = LocalHarnessConfig.model_validate({})
    explicit_config = LocalHarnessConfig.model_validate({"max_turns": 10})

    assert default_config.max_turns == 4
    assert explicit_config.max_turns == 10


def test_config_field_name_is_ordinary_serializable_data() -> None:
    class LocalTasksetConfig(TasksetConfig):
        config: dict[str, object] | None = None

    config = LocalTasksetConfig.model_validate({"config": {"mode": "loaded"}})

    assert config.config == {"mode": "loaded"}


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

    assert (
        group_reward.load_harness(
            config=group_reward.GroupRewardHarnessConfig()
        ).config.max_turns
        == 1
    )
    assert (
        parallel_sandbox.load_harness(
            config=parallel_sandbox.ParallelSandboxHarnessConfig()
        ).config.max_turns
        == 4
    )
    assert (
        self_judge.load_harness(
            config=self_judge.SelfJudgeHarnessConfig()
        ).config.max_turns
        == 8
    )


def test_bfcl_loader_preserves_mapping_config_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("environments.bfcl_v3.bfcl_v3")
    seen: dict[str, object] = {}

    def fake_taskset(config: object = None, **kwargs: object) -> Taskset:
        _ = kwargs
        seen["taskset_config"] = config
        return make_taskset(source=source_loader, config=config)

    def fake_harness(config: object = None, **kwargs: object) -> Harness:
        _ = kwargs
        seen["harness_config"] = config
        return make_harness(config=config)

    monkeypatch.setattr(module, "load_taskset", fake_taskset)
    monkeypatch.setattr(module, "load_harness", fake_harness)

    env = module.load_environment(
        config=module.BFCLEnvConfig(
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
        return make_taskset(source=source_loader)

    monkeypatch.setattr(module, "load_taskset", fake_taskset)

    env = module.load_environment(
        config=module.Tau2EnvConfig(
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

    taskset = module.load_taskset(config=module.SelfJudgeTasksetConfig(num_examples=2))
    harness = module.load_harness(config=module.SelfJudgeHarnessConfig(max_turns=3))
    shortcut_env = module.load_environment(
        config=module.SelfJudgeEnvConfig(
            taskset={"num_examples": 2},
            harness={"max_turns": 3},
        ),
    )
    override_env = module.load_environment(
        config=module.SelfJudgeEnvConfig(
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
    toolset = env.harness.toolsets[0]
    assert toolset.bindings["ask_subagent.harness"] == "objects.harness"
    child_harness = toolset.objects["harness"]()
    assert child_harness.config.model is None


def test_nested_configs_validate_and_feed_runtime_objects() -> None:
    sandbox = vf.SandboxConfig(
        image="python:3.12-slim",
        packages="numpy",
        setup_commands="echo ready",
        scope="group",
    )
    harness = make_harness(program={"sandbox": True}, sandbox=sandbox)

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

    taskset = make_taskset(config=taskset_config)
    harness = make_harness(config=harness_config)

    assert taskset.source is source_loader
    assert getattr(taskset.rewards[0], "__name__") == "config_reward"
    assert getattr(taskset.rewards[0], "reward_weight") == 0.5
    prefix = taskset.named_toolsets["configured"].bindings["config_tool.prefix"]
    assert callable(prefix)
    assert prefix() == "toml"
    assert harness.program == {"fn": ref("config_program")}
    assert callable(harness._program)
    assert harness.config.max_turns == 7


@pytest.mark.asyncio
async def test_task_tools_filter_exposed_tools() -> None:
    harness = make_harness(toolsets=[Toolset(tools=[direct_tool, hidden_tool])])
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
    taskset = make_taskset(source=source_loader)
    harness = make_harness()

    taskset.add_toolset({"direct": {"tools": [direct_tool]}})
    harness.add_toolset({"configured": config_toolset})

    assert taskset.named_toolsets["direct"].tools == (direct_tool,)
    assert harness.named_toolsets["configured"].tools == (config_tool,)


def test_taskset_extension_refreshes_attached_harness_runtime() -> None:
    taskset = make_taskset(source=source_loader)
    harness = make_harness()
    harness.attach_taskset(taskset)

    taskset.add_toolset({"direct": {"tools": [direct_tool]}})

    assert "direct" in harness.runtime.named_toolsets

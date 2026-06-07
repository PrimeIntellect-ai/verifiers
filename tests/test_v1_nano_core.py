from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import importlib
import json
import sys
from types import ModuleType

from aiohttp import ClientSession, web
import pytest

import verifiers.v1 as vf
from verifiers.errors import InfraError, ToolError
from verifiers.types import ClientConfig, EvalConfig, Response
from verifiers.utils import eval_utils
from verifiers.v1.loaders import (
    load_environment_from_components,
    load_harness_from_module,
    load_taskset_from_module,
)
from verifiers.v1.mcp import BoundUpdate, MCPToolRegistry, split_result
from verifiers.v1.protocols import parse_anthropic_user_messages
from verifiers.v1.toolset import ToolBinding


def attach_mock_model(
    env: vf.Env, mock_client, model: str = "test-model"
) -> vf.ModelConfig:
    config = vf.ModelConfig(client=ClientConfig(), model=model)
    env.harness.load_model_client = lambda _: vf.ModelClient(
        config=config, client=mock_client
    )

    async def close_model_client(_: vf.ModelClient) -> None:
        return None

    env.harness.close_model_client = close_model_client
    return config


class NanoTask(vf.Task):
    answer: str


class NanoTaskset(vf.Taskset):
    task_type = NanoTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [
            NanoTask(
                row_id=0,
                prompt=[{"role": "user", "content": "say ok"}],
                answer="ok",
                max_turns=1,
            )
        ]

    @vf.setup
    async def setup_extras(self, state: vf.State) -> None:
        state.extras["seen_setup"] = True

    @vf.reward
    async def exact(self, task: NanoTask, state: vf.State) -> float:
        completion = state.completion
        text = str(completion[-1].content if completion else "")
        return float(text.strip() == task.answer)


class GroupNanoTask(NanoTask):
    candidate: int


class GroupTaskset(NanoTaskset):
    async def init_group(
        self, task: NanoTask, num_rollouts: int
    ) -> tuple[list[GroupNanoTask], list[vf.State]]:
        tasks = [
            GroupNanoTask.model_validate(
                {
                    **task.model_dump(
                        mode="json", exclude_none=True, exclude_defaults=True
                    ),
                    "candidate": index,
                }
            )
            for index in range(num_rollouts)
        ]
        return tasks, [vf.State(task_id=task.task_id) for task in tasks]

    @vf.reward(stage="group")
    async def relative(
        self, tasks: list[GroupNanoTask], states: list[vf.State]
    ) -> list[float]:
        _ = states
        return [float(task.candidate == 0) for task in tasks]


@vf.advantage
def custom_env_advantage(tasks: list[vf.Task], states: list[vf.State]) -> None:
    _ = tasks
    for index, state in enumerate(states):
        value = float(index + 10)
        for turn in state.transcript:
            if turn.tokens is not None:
                turn.tokens.prompt_advantages = [0.0 for _ in turn.tokens.prompt_ids]
                turn.tokens.completion_advantages = [
                    value for _ in turn.tokens.completion_ids
                ]


class EmptyPromptUserConfig(vf.UserConfig):
    pass


class EmptyPromptUser(vf.User[EmptyPromptUserConfig]):
    @vf.user
    def respond(self) -> dict:
        return {"messages": [{"role": "user", "content": "server prompt"}]}


class EmptyPromptTasksetConfig(vf.TasksetConfig):
    user: vf.UserConfig | None = EmptyPromptUserConfig()


class EmptyPromptTaskset(vf.Taskset[EmptyPromptTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [{"example_id": 0, "prompt": [], "max_turns": 1}]


class PatchOnlyUserConfig(vf.UserConfig):
    pass


class PatchOnlyUser(vf.User[PatchOnlyUserConfig]):
    @vf.user(
        sets={
            "done": "state.extras.done",
            "stop_condition": "state.stop_condition",
        }
    )
    def respond(self) -> dict:
        return {"done": True, "stop_condition": "user_bootstrap_done"}


class PatchOnlyUserTasksetConfig(vf.TasksetConfig):
    user: vf.UserConfig | None = PatchOnlyUserConfig()


class PatchOnlyUserTaskset(vf.Taskset[PatchOnlyUserTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [{"example_id": 0, "prompt": [], "max_turns": 1}]


class ToolUserConfig(vf.UserConfig):
    pass


class ToolUser(vf.User[ToolUserConfig]):
    @vf.user
    def respond(self) -> dict:
        return {"messages": [{"role": "user", "content": "server prompt"}]}

    @vf.tool
    def ping(self) -> str:
        return "pong"


class TaskSetupToolsetConfig(vf.ToolsetConfig):
    scope: vf.Scope = "env"


class TaskSetupToolset(vf.Toolset[TaskSetupToolsetConfig]):
    count = 0

    @vf.tool(
        args={"task": "task"},
        sets={
            "grader_case": "state.metadata.grader_case",
            "setup_count": "state.extras.setup_count",
            "setup": "state.artifacts.setup",
        },
    )
    def materialize(self, task: dict) -> dict:
        type(self).count += 1
        count = type(self).count
        return {
            "content": "",
            "grader_case": f"{task['task_id']}:{count}",
            "setup_count": count,
            "setup": {"count": count},
        }


class ServerSetupTaskset(vf.Taskset):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [
            {"example_id": 0, "prompt": "say ok", "max_turns": 1},
            {"example_id": 1, "prompt": "say ok", "max_turns": 1},
        ]

    def load_toolsets(self, config: vf.TasksetConfig) -> vf.ToolsetConfigs:
        _ = config
        return {"setup": TaskSetupToolsetConfig(hide=["materialize"])}

    @vf.setup
    async def materialize_task(
        self,
        task: vf.Task,
        state: vf.State,
        harness: vf.Harness,
        runtime: vf.Runtime,
        toolsets: MCPToolRegistry,
    ) -> None:
        _ = task
        result = await toolsets.call_hidden("materialize", {})
        harness.apply_tool_result(state, result)
        case = state.metadata["grader_case"]
        assert isinstance(case, str)
        await runtime.write("grader_case.txt", case.encode())


class DemoToolsetConfig(vf.ToolsetConfig):
    pass


class DemoToolset(vf.Toolset[DemoToolsetConfig]):
    @vf.tool
    def echo(self, text: str) -> str:
        return text.upper()

    @vf.tool
    def suffix(self, text: str) -> str:
        return text + "!"


class BoundToolsetConfig(vf.ToolsetConfig):
    pass


class BoundToolset(vf.Toolset[BoundToolsetConfig]):
    @vf.tool(
        args={"task_name": "task.name"},
        extends={"events": "state.extras.events"},
        sets={
            "profile": "state.extras.profile",
        },
    )
    def record(self, name: str, task_name: str) -> dict:
        return {
            "content": f"{task_name}:{name}",
            "events": [{"task": task_name, "name": name}],
            "profile": {"task": task_name},
        }


class DynamicToolsetConfig(vf.ToolsetConfig):
    pass


class DynamicToolset(vf.Toolset[DynamicToolsetConfig]):
    @vf.tool(
        hidden=True,
        args={"task_name": "task.name"},
        sets={"ready": "state.extras.dynamic_ready"},
    )
    def setup(self, task_name: str) -> dict:
        return {
            "ready": True,
            "tools": [
                {
                    "name": "dynamic_echo",
                    "description": f"Echo through {task_name}",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            ],
        }

    @vf.tool(hidden=True, sets={"called": "state.extras.dynamic_called"})
    def call_tool(self, name: str, input: vf.JsonData) -> dict:
        return {"content": f"{name}:{input['text']}", "called": True}


class DynamicUserConfig(vf.UserConfig):
    pass


class DynamicUser(vf.User[DynamicUserConfig]):
    @vf.tool(
        hidden=True,
        sets={"ready": "state.extras.user_dynamic_ready"},
    )
    def setup(self) -> dict:
        return {
            "ready": True,
            "tools": [
                {
                    "name": "user_echo",
                    "description": "Echo through user server",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            ],
        }

    @vf.tool(hidden=True, sets={"called": "state.extras.user_dynamic_called"})
    def call_tool(self, name: str, input: vf.JsonData) -> dict:
        return {"content": f"{name}:{input['text']}", "called": True}


class EnvScopedToolsetConfig(vf.ToolsetConfig):
    scope: vf.Scope = "env"


class EnvScopedUserConfig(vf.UserConfig):
    scope: vf.Scope = "env"


class EnvServerTasksetConfig(vf.TasksetConfig):
    toolsets: vf.ToolsetConfigs = {"demo": EnvScopedToolsetConfig()}
    user: vf.UserConfig | None = EnvScopedUserConfig()


class EnvServerTaskset(vf.Taskset[EnvServerTasksetConfig]):
    pass


class ConfiguredToolsetTasksetConfig(vf.TasksetConfig):
    toolsets: vf.ToolsetConfigs = {"demo": DemoToolsetConfig()}


class ConfiguredToolsetTaskset(vf.Taskset[ConfiguredToolsetTasksetConfig]):
    pass


def test_v1_state_is_pydantic_and_extras_owned() -> None:
    task = NanoTask(prompt="hello", answer="world")
    state = vf.State(task_id=task.task_id)

    state.extras["x"] = 1
    state.reward += 0.5
    state.assert_serializable()

    assert state.extras == {"x": 1}
    assert state.reward == 0.5
    with pytest.raises(TypeError):
        state["x"] = 2
    assert state.to_output(task)["transcript"] == []


def test_v1_state_to_output_preserves_empty_turn_prompt() -> None:
    task = vf.Task(prompt="fallback")
    state = vf.State(transcript=[vf.Turn(prompt=[], completion=[])])

    assert state.prompt == []
    assert state.to_output(task)["prompt"] == []


def test_v1_state_to_output_state_columns_preserve_task_prompt_fallback() -> None:
    task = vf.Task(prompt="fallback")
    state = vf.State()

    assert state.to_output(task, state_columns=["prompt"])["prompt"] == [
        {"role": "user", "content": "fallback"}
    ]


def test_v1_task_id_is_deterministic_from_task_contents() -> None:
    first = NanoTask(prompt="hello", answer="world")
    second = NanoTask(prompt="hello", answer="world")
    changed = NanoTask(prompt="hello", answer="there")
    explicit = NanoTask(task_id="chosen", prompt="hello", answer="world")

    assert first.task_id == second.task_id
    assert first.task_id != changed.task_id
    assert explicit.task_id == "chosen"


def test_v1_state_messages_uses_latest_prompt_once() -> None:
    first_prompt = [vf.UserMessage(content="first")]
    first_completion = [vf.AssistantMessage(content="one")]
    second_prompt = [*first_prompt, *first_completion, vf.UserMessage(content="second")]
    second_completion = [vf.AssistantMessage(content="two")]
    state = vf.State(
        transcript=[
            vf.Turn(prompt=first_prompt, completion=first_completion),
            vf.Turn(prompt=second_prompt, completion=second_completion),
        ]
    )

    assert [message.content for message in state.messages] == [
        "first",
        "one",
        "second",
        "two",
    ]


def test_task_user_defaults_to_auto_and_omits_from_json() -> None:
    task = vf.Task(prompt="hello")

    assert task.user is None
    assert "user" not in task.model_dump(mode="json", exclude_none=True)
    assert (
        vf.Task(user=False).model_dump(mode="json", exclude_none=True)["user"] is False
    )


def test_v1_loader_does_not_recurse_to_same_package_config_id() -> None:
    module = ModuleType("same_env.taskset")

    with pytest.raises(AttributeError, match="does not expose load_taskset"):
        load_taskset_from_module(module, config={"id": "same-env"})

    with pytest.raises(AttributeError, match="does not expose load_harness"):
        load_harness_from_module(module, config={"id": "same-env"})


def test_v1_harness_allows_metrics_but_rejects_rewards() -> None:
    class MetricHarness(vf.Harness):
        @vf.metric
        async def command_calls(self) -> float:
            return 1.0

    assert MetricHarness().signals[0]["name"] == "command_calls"

    class RewardHarness(vf.Harness):
        @vf.reward
        async def execution_reward(self) -> float:
            return 1.0

    with pytest.raises(ValueError, match="Harness signals must be metrics"):
        RewardHarness()


class TaskExtras(vf.Extras):
    task_flag: bool = True


class HarnessExtras(vf.Extras):
    harness_count: int = 2


class ExtrasTasksetConfig(vf.TasksetConfig):
    extras: TaskExtras = TaskExtras()


class ExtrasHarnessConfig(vf.HarnessConfig):
    extras: HarnessExtras = HarnessExtras()
    max_turns: int = 1


class ExtrasTaskset(vf.Taskset[ExtrasTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [{"example_id": 0, "prompt": "say ok"}]


def test_v1_extras_config_schemas_realize_and_reject_conflicts() -> None:
    env = vf.Env(taskset=ExtrasTaskset(), harness=vf.Harness(ExtrasHarnessConfig()))
    state = vf.State()
    env.harness.initialize_extras(state)

    assert state.extras == {"task_flag": True, "harness_count": 2}
    env.harness.validate_extras(state)

    class TaskConflict(vf.Extras):
        shared: int = 1

    class HarnessConflict(vf.Extras):
        shared: str = "x"

    class ConflictTasksetConfig(vf.TasksetConfig):
        extras: TaskConflict = TaskConflict()

    class ConflictHarnessConfig(vf.HarnessConfig):
        extras: HarnessConflict = HarnessConflict()

    class ConflictTaskset(vf.Taskset[ConflictTasksetConfig]):
        pass

    with pytest.raises(ValueError, match="defined by both"):
        vf.Env(
            taskset=ConflictTaskset(),
            harness=vf.Harness(ConflictHarnessConfig()),
        )


def test_toolset_config_sources_and_enabled_flags_resolve_directly() -> None:
    taskset = ConfiguredToolsetTaskset(
        config={"toolsets": {"demo": {"enabled": False}}}
    )
    assert taskset.toolsets == {}

    taskset = ConfiguredToolsetTaskset(
        config={
            "toolsets": {
                "custom": {
                    "source": f"{__name__}:DemoToolsetConfig",
                }
            }
        }
    )
    assert list(taskset.toolsets) == ["demo", "custom"]

    with pytest.raises(ValueError, match="cannot be disabled"):
        ConfiguredToolsetTaskset(config={"toolsets": {"custom": {"enabled": False}}})

    with pytest.raises(ValueError, match="set source"):
        ConfiguredToolsetTaskset(config={"toolsets": {"custom": {}}})

    with pytest.raises(TypeError, match="source must match"):
        ConfiguredToolsetTaskset(
            config={
                "toolsets": {
                    "demo": {
                        "source": f"{__name__}:BoundToolsetConfig",
                    }
                }
            }
        )


@pytest.mark.asyncio
async def test_v1_model_client_uses_serialized_transcript_record(mock_client) -> None:
    config = vf.ModelConfig(client=ClientConfig(), model="test-model")
    state = vf.State(task_id="task-1")
    state.transcript.append(
        vf.Turn(
            prompt=[vf.UserMessage(content="first")],
            completion=[vf.AssistantMessage(content="done")],
            tokens=vf.TurnTokens(
                prompt_ids=[1],
                prompt_mask=[0],
                completion_ids=[2],
                completion_mask=[1],
                completion_logprobs=[-0.1],
            ),
        )
    )

    assert [message.content for message in state.messages] == ["first", "done"]

    await vf.ModelClient(config=config, client=mock_client).get_response(
        prompt=[vf.UserMessage(content="next")],
        state=state,
    )

    client_state = mock_client.last_call_kwargs["state"]
    assert "trajectory" not in client_state
    assert client_state["task_id"] == "task-1"
    assert client_state["transcript"][0]["prompt"][0]["content"] == "first"
    assert client_state["transcript"][0]["tokens"]["completion_ids"] == [2]


def test_v1_model_client_renderer_handle_is_live_only(mock_client) -> None:
    config = vf.ModelConfig(client=ClientConfig(), model="test-model")
    model = vf.ModelClient(config=config, client=mock_client)
    assert model.get_renderer() is None

    renderer = object()
    model = vf.ModelClient(config=config, client=mock_client, renderer=renderer)
    assert model.get_renderer() is renderer


@pytest.mark.asyncio
async def test_v1_standalone_env_rollout_scores_from_transcript(mock_client) -> None:
    mock_client.set_default_response("ok")
    env = vf.Env(taskset=NanoTaskset(), harness=vf.Harness())
    model = attach_mock_model(env, mock_client)
    row = env.get_dataset()[0]
    task = env.taskset.to_task(row)

    state = await env.run_rollout(row, model=model)
    output = state.to_output(task)

    assert output["reward"] == 1.0
    assert output["metrics"]["exact"] == 1.0
    assert output["metrics"]["num_turns"] == 1.0
    assert output["transcript"][0]["completion"][0]["content"] == "ok"


@pytest.mark.asyncio
async def test_harness_run_accepts_string_task_and_model(mock_client) -> None:
    mock_client.set_default_response("ok")
    harness = vf.Harness()
    configs: list[vf.ModelConfig] = []

    def load_model_client(config: vf.ModelConfig) -> vf.ModelClient:
        configs.append(config)
        return vf.ModelClient(config=config, client=mock_client)

    async def close_model_client(_: vf.ModelClient) -> None:
        return None

    harness.load_model_client = load_model_client
    harness.close_model_client = close_model_client

    state = await harness.run(task="hello world", model="openai/gpt-5")

    assert state.task_id is not None
    assert state.prompt[0].content == "hello world"
    assert state.completion[0].content == "ok"
    assert configs[0].model == "openai/gpt-5"
    assert configs[0].client.api_key_var == "PRIME_API_KEY"
    assert configs[0].client.api_base_url == "https://api.pinference.ai/api/v1"
    assert mock_client.last_call_kwargs["model"] == "openai/gpt-5"


@pytest.mark.asyncio
async def test_harness_run_requires_user_when_task_user_is_true(mock_client) -> None:
    mock_client.set_default_response("ok")
    env = vf.Env(taskset=NanoTaskset(), harness=vf.Harness())
    model = attach_mock_model(env, mock_client)

    state = await env.run_rollout(
        {"prompt": "say ok", "answer": "ok", "user": True, "max_turns": 1},
        model=model,
    )

    assert state.stop_condition == "has_error"
    assert state.error is not None
    assert "requires a user server" in state.error["message"]


@pytest.mark.asyncio
async def test_harness_run_rejects_nested_scoring_context(mock_client) -> None:
    task = vf.Task(prompt="judge")
    state = vf.State(task_id=task.task_id)
    model = vf.ModelConfig(client=ClientConfig(), model="test-model")
    context = vf.Context(
        task=task,
        state=state,
        model_client=vf.ModelClient(config=model, client=mock_client),
        scoring=True,
    )

    with pytest.raises(RuntimeError, match="Nested scored harness runs"):
        await vf.Harness().run(task="nested judge", context=context, score=True)


@pytest.mark.asyncio
async def test_harness_run_with_context_preserves_parent_state_task_id(
    mock_client,
) -> None:
    mock_client.set_default_response("child")
    harness = vf.Harness()
    model = vf.ModelConfig(model="test-model")

    def load_model_client(config: vf.ModelConfig) -> vf.ModelClient:
        return vf.ModelClient(config=config, client=mock_client)

    async def close_model_client(_: vf.ModelClient) -> None:
        return None

    harness.load_model_client = load_model_client
    harness.close_model_client = close_model_client
    parent_task = vf.Task(task_id="parent-task", prompt="parent")
    state = vf.State(task_id="parent-task")

    async with harness.open_context(
        task=parent_task,
        state=state,
        model=model,
    ) as context:
        await harness.run("child prompt", context=context)

    assert state.task_id == "parent-task"
    assert state.transcript[-1].prompt[-1].content == "child prompt"


@pytest.mark.asyncio
async def test_v1_group_rewards_and_advantages_apply_to_turns(mock_client) -> None:
    mock_client.set_default_response("ok")
    env = vf.Env(taskset=GroupTaskset(), harness=vf.Harness(), advantage="grpo")
    model = attach_mock_model(env, mock_client)
    row = env.get_dataset()[0]
    base_task = env.taskset.to_task(row)
    tasks, states = await env.taskset.init_group(base_task, 2)

    states = await asyncio.gather(
        *[
            env.run_rollout(task, model=model, state=state)
            for task, state in zip(tasks, states, strict=True)
        ]
    )
    states = await env.score_group(tasks, states)
    outputs = [state.to_output(task) for task, state in zip(tasks, states, strict=True)]

    assert [output["reward"] for output in outputs] == [2.0, 1.0]
    assert "advantage" not in outputs[0]
    assert "advantage" not in outputs[0]["transcript"][0]


@pytest.mark.asyncio
async def test_v1_empty_prompt_bootstraps_from_user_server(mock_client) -> None:
    mock_client.set_default_response("ok")
    env = vf.Env(taskset=EmptyPromptTaskset(), harness=vf.Harness())
    model = attach_mock_model(env, mock_client)
    row = env.get_dataset()[0]
    task = env.taskset.to_task(row)

    state = await env.run_rollout(row, model=model)
    output = state.to_output(task)

    prompt = mock_client.last_call_kwargs["prompt"]
    assert prompt[-1].role == "user"
    assert prompt[-1].content == "server prompt"
    assert output["completion"][0]["content"] == "ok"
    assert output["transcript"][0]["prompt"][-1]["content"] == "server prompt"


@pytest.mark.asyncio
async def test_user_server_can_return_only_bound_state_updates(mock_client) -> None:
    mock_client.set_default_response("should not run")
    env = vf.Env(taskset=PatchOnlyUserTaskset(), harness=vf.Harness())
    model = attach_mock_model(env, mock_client)

    state = await env.run_rollout(env.get_dataset()[0], model=model)

    assert state.extras["done"] is True
    assert state.stop_condition == "user_bootstrap_done"
    assert state.transcript == []
    assert mock_client.last_call_kwargs == {}
    await env.close()


@pytest.mark.asyncio
async def test_user_server_hides_respond_and_exposes_user_tools() -> None:
    async with MCPToolRegistry({"user": ToolUserConfig()}) as registry:
        names = [tool.name for tool in registry.tools() or []]
        prompt = await registry.call_hidden("respond", {})
        result = await registry.call("user_ping", {})
        with pytest.raises(ToolError, match="disabled"):
            await registry.call("user_respond", {})

    assert names == ["user_ping"]
    assert prompt.response.messages[0].content == "server prompt"
    assert result.response.content == "pong"


@pytest.mark.asyncio
async def test_v1_setup_can_materialize_task_local_state_from_env_server(
    mock_client,
) -> None:
    mock_client.set_default_response("ok")
    env = vf.Env(taskset=ServerSetupTaskset(), harness=vf.Harness())
    model = attach_mock_model(env, mock_client)
    rows = list(env.get_dataset())

    states = [await env.run_rollout(row, model=model) for row in rows]

    assert [state.extras["setup_count"] for state in states] == [1, 2]
    assert [state.artifacts["setup"]["count"] for state in states] == [1, 2]
    assert states[0].metadata["grader_case"].endswith(":1")
    assert states[1].metadata["grader_case"].endswith(":2")
    assert "setup_materialize" not in [
        tool.name for tool in mock_client.last_call_kwargs["tools"] or []
    ]
    await env.close()


@pytest.mark.asyncio
async def test_migrated_hello_rlm_v1_runs_without_old_runtime(mock_client) -> None:
    from environments.hello_rlm_v1.hello_rlm_v1 import taskset as module

    env = load_environment_from_components(
        module,
        {
            "config": {
                "harness": {
                    "command": [
                        sys.executable,
                        "-c",
                        "import os; print(os.environ['VF_PROMPT'].split('exactly ', 1)[1].rstrip('.'))",
                    ]
                }
            }
        },
    )
    model = attach_mock_model(env, mock_client, "unused-model")
    row = env.get_dataset()[0]
    task = env.taskset.to_task(row)

    state = await env.run_rollout(row, model=model)
    output = state.to_output(task)

    assert output["reward"] == 1.0
    assert output["metrics"]["exact_answer"] == 1.0
    assert output["transcript"][0]["completion"][0]["content"] == output["answer"]


@pytest.mark.asyncio
async def test_bfcl_multi_turn_respects_completed_state(
    monkeypatch, mock_client
) -> None:
    from environments.bfcl_v3_v1.bfcl_v3_v1 import taskset as bfcl

    for package_name in [
        "bfcl_eval",
        "bfcl_eval.constants",
        "bfcl_eval.eval_checker",
        "bfcl_eval.eval_checker.multi_turn_eval",
        "bfcl_eval.model_handler",
    ]:
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    default_prompts = ModuleType("bfcl_eval.constants.default_prompts")
    default_prompts.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC = "add tools"
    monkeypatch.setitem(
        sys.modules,
        "bfcl_eval.constants.default_prompts",
        default_prompts,
    )

    simulator_calls: list[list[str]] = []
    multi_turn_utils = ModuleType(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"
    )

    def execute_multi_turn_func_call(
        func_call_list: list[str],
        initial_config: vf.JsonData,
        involved_classes: list[str],
        model_name: str,
        test_entry_id: str,
        *,
        long_context: bool,
    ) -> tuple[list[str], None]:
        _ = initial_config, involved_classes, model_name, test_entry_id, long_context
        simulator_calls.append(func_call_list)
        return [], None

    multi_turn_utils.execute_multi_turn_func_call = execute_multi_turn_func_call
    monkeypatch.setitem(
        sys.modules,
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils",
        multi_turn_utils,
    )

    base_handler = ModuleType("bfcl_eval.model_handler.base_handler")
    base_handler.is_empty_execute_response = lambda value: not value
    monkeypatch.setitem(
        sys.modules,
        "bfcl_eval.model_handler.base_handler",
        base_handler,
    )
    monkeypatch.setattr(bfcl, "bfcl_tool_defs", lambda _functions: [])
    monkeypatch.setattr(bfcl, "bfcl_missed_function", lambda _task: {})
    monkeypatch.setattr(bfcl, "bfcl_involved_classes", lambda _task: [])

    task = bfcl.BFCLTask(
        row_id=0,
        prompt="first",
        category="multi_turn_base",
        question=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        function=[],
        initial_config={},
        involved_classes=[],
        max_steps_per_turn=1,
        max_turns=2,
    )
    state = vf.State(task_id=task.task_id)
    state.stop("precompleted")
    model = vf.ModelConfig(client=ClientConfig(), model="test-model")
    context = vf.Context(
        task=task,
        state=state,
        model_client=vf.ModelClient(config=model, client=mock_client),
    )

    await bfcl.BFCLHarness(config=bfcl.BFCLHarnessConfig()).run_multi_turn(
        context, task, state
    )

    assert simulator_calls == [[]]
    assert mock_client.last_call_kwargs == {}
    assert state.stop_condition == "precompleted"

    bounded_task = task.model_copy(update={"task_id": "bounded", "max_turns": 1})
    bounded_state = vf.State(task_id=bounded_task.task_id)
    bounded_context = vf.Context(
        task=bounded_task,
        state=bounded_state,
        model_client=vf.ModelClient(config=model, client=mock_client),
    )

    mock_client.set_default_response("done")
    await bfcl.BFCLHarness(config=bfcl.BFCLHarnessConfig()).run_multi_turn(
        bounded_context, bounded_task, bounded_state
    )

    assert mock_client.call_count == 1
    assert len(bounded_state.transcript) == 1
    assert bounded_state.transcript[0].timing.start > 0.0
    assert (
        bounded_state.transcript[0].timing.end
        >= bounded_state.transcript[0].timing.start
    )
    assert bounded_state.stop_condition == "max_turns"


@pytest.mark.asyncio
async def test_mcp_toolset_exposes_multiple_server_tools() -> None:
    toolsets = {"demo": DemoToolsetConfig()}

    async with MCPToolRegistry(toolsets) as registry:
        names = [tool.name for tool in registry.tools() or []]
        result = await registry.call("demo_echo", {"text": "ok"})

    assert names == ["demo_echo", "demo_suffix"]
    assert result.response.content == "OK"


@pytest.mark.asyncio
async def test_mcp_owned_runtime_stops_when_server_start_fails(monkeypatch) -> None:
    mcp_module = importlib.import_module("verifiers.v1.mcp")
    events: list[str] = []

    class FailingRuntime(vf.Runtime):
        async def start(self) -> None:
            events.append("start")

        async def stop(self) -> None:
            events.append("stop")

        async def expose(self, port: int) -> str:
            return f"http://127.0.0.1:{port}"

        async def run(
            self,
            command: list[str],
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout: float | None = None,
        ) -> vf.CommandResult:
            _ = command, cwd, env, timeout
            return vf.CommandResult(returncode=0)

        async def read(self, path: str) -> bytes:
            _ = path
            return b""

        async def write(self, path: str, data: bytes) -> None:
            _ = path, data

        async def run_background(
            self,
            command: list[str],
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            log: str | None = None,
        ) -> None:
            _ = command, cwd, env, log
            events.append("run_background")
            raise RuntimeError("server failed")

    class FailingProvider(vf.RuntimeProvider):
        def create_runtime(self) -> vf.Runtime:
            return FailingRuntime()

    monkeypatch.setattr(
        mcp_module,
        "make_runtime_provider",
        lambda _: FailingProvider(),
    )
    registry = MCPToolRegistry({})

    with pytest.raises(RuntimeError, match="server failed"):
        async with registry.open_server("demo", DemoToolsetConfig()):
            pass

    assert events == ["start", "run_background", "stop"]


@pytest.mark.asyncio
async def test_mcp_registry_closes_partial_stack_when_enter_fails(monkeypatch) -> None:
    import mcp.client.session as session_module

    events: list[str] = []

    class FailingClientSession:
        def __init__(self, read: object, write: object) -> None:
            _ = read, write

        async def __aenter__(self) -> "FailingClientSession":
            events.append("session_enter")
            return self

        async def __aexit__(self, *exc: object) -> None:
            _ = exc
            events.append("session_exit")

        async def initialize(self) -> None:
            events.append("initialize")
            raise RuntimeError("initialize failed")

    @asynccontextmanager
    async def open_server(_name: str, _server: vf.ServerConfig):
        events.append("server_enter")
        try:
            yield object(), object()
        finally:
            events.append("server_exit")

    monkeypatch.setattr(session_module, "ClientSession", FailingClientSession)
    registry = MCPToolRegistry({"demo": DemoToolsetConfig()})
    monkeypatch.setattr(registry, "open_server", open_server)

    with pytest.raises(RuntimeError, match="initialize failed"):
        async with registry:
            pass

    assert events == [
        "server_enter",
        "session_enter",
        "initialize",
        "session_exit",
        "server_exit",
    ]
    assert registry.tools() is None


@pytest.mark.asyncio
async def test_mcp_tool_registry_applies_task_visibility() -> None:
    toolsets = {"demo": DemoToolsetConfig()}

    async with MCPToolRegistry(toolsets) as registry:
        registry.set_visibility(
            toolsets=vf.TaskVisibility(show=["demo"]),
            tools=vf.TaskVisibility(hide=["demo_suffix"]),
        )
        names = [tool.name for tool in registry.tools() or []]
        result = await registry.call("demo_echo", {"text": "ok"})
        with pytest.raises(ToolError, match="disabled"):
            await registry.call("demo_suffix", {"text": "ok"})

    assert names == ["demo_echo"]
    assert result.response.content == "OK"


@pytest.mark.asyncio
async def test_mcp_bindings_hide_args_and_bind_returns() -> None:
    toolsets = {"bound": BoundToolsetConfig()}
    state = vf.State()
    task = vf.Task(name="demo", prompt="say ok")
    harness = vf.Harness()

    async with MCPToolRegistry(toolsets) as registry:
        registry.set_context(harness.binding_context(task, state))
        tools = registry.tools() or []
        result = await registry.call("bound_record", {"name": "alpha"})

    properties = tools[0].parameters["properties"]
    assert "name" in properties
    assert "task_name" not in properties
    assert result.response.content == "demo:alpha"

    harness.apply_bound_updates(state, list(result.updates))
    assert state.extras == {
        "events": [{"task": "demo", "name": "alpha"}],
        "profile": {"task": "demo"},
    }


@pytest.mark.asyncio
async def test_mcp_toolset_setup_registers_dynamic_tools() -> None:
    toolsets = {"dynamic": DynamicToolsetConfig()}
    state = vf.State()
    task = vf.Task(name="demo", prompt="say ok")
    harness = vf.Harness()

    async with MCPToolRegistry(toolsets) as registry:
        await registry.resolve(
            context=harness.binding_context(task, state),
            resolution_key=f"{state.id}:{task.task_id}",
            apply_updates=lambda updates: harness.apply_bound_updates(state, updates),
        )
        names = [tool.name for tool in registry.tools() or []]
        assert registry.has_hidden("setup")
        assert registry.has_hidden("call_tool")
        result = await registry.call("dynamic_echo", {"text": "ok"})

    harness.apply_tool_result(state, result)

    assert names == ["dynamic_echo"]
    assert result.response.content == "dynamic_echo:ok"
    assert state.extras["dynamic_ready"] is True
    assert state.extras["dynamic_called"] is True


@pytest.mark.asyncio
async def test_dynamic_tools_obey_toolset_config_visibility() -> None:
    toolsets = {"dynamic": DynamicToolsetConfig(hide=["dynamic_echo"])}
    state = vf.State()
    task = vf.Task(name="demo", prompt="say ok")
    harness = vf.Harness()

    async with MCPToolRegistry(toolsets) as registry:
        await registry.resolve(
            context=harness.binding_context(task, state),
            resolution_key=f"{state.id}:{task.task_id}",
            apply_updates=lambda updates: harness.apply_bound_updates(state, updates),
        )
        assert registry.tools() is None
        with pytest.raises(ToolError, match="Unknown MCP tool"):
            await registry.call("dynamic_echo", {"text": "ok"})


@pytest.mark.asyncio
async def test_user_setup_registers_dynamic_model_tools() -> None:
    state = vf.State()
    task = vf.Task(name="demo", prompt="say ok")
    harness = vf.Harness()

    async with MCPToolRegistry({"user": DynamicUserConfig()}) as user_registry:
        async with MCPToolRegistry({}, parents=[user_registry]) as registry:
            await registry.resolve(
                context=harness.binding_context(task, state),
                resolution_key=f"{state.id}:{task.task_id}",
                apply_updates=lambda updates: harness.apply_bound_updates(
                    state, updates
                ),
            )
            tools = registry.tools() or []
            result = await registry.call("user_echo", {"text": "ok"})

    harness.apply_tool_result(state, result)

    assert [tool.name for tool in tools] == ["user_echo"]
    assert result.response.content == "user_echo:ok"
    assert state.extras["user_dynamic_ready"] is True
    assert state.extras["user_dynamic_called"] is True


@pytest.mark.asyncio
async def test_env_user_startup_failure_closes_started_env_toolsets(
    monkeypatch,
) -> None:
    harness_module = importlib.import_module("verifiers.v1.harness")
    events: list[str] = []

    class FakeRegistry:
        def __init__(self, servers, **_: object) -> None:
            self.kind = "user" if "user" in servers else "toolsets"

        async def __aenter__(self) -> "FakeRegistry":
            events.append(f"enter:{self.kind}")
            if self.kind == "user":
                raise RuntimeError("user failed")
            return self

        async def __aexit__(self, *_: object) -> None:
            events.append(f"exit:{self.kind}")

    monkeypatch.setattr(harness_module, "MCPToolRegistry", FakeRegistry)
    harness = vf.Harness()
    harness.bind(taskset=EnvServerTaskset())

    with pytest.raises(RuntimeError, match="user failed"):
        await harness.ensure_env_servers()

    assert events == ["enter:toolsets", "enter:user", "exit:user", "exit:toolsets"]
    assert harness._env_toolsets is None
    assert harness._env_user is None


def test_bound_state_updates_allow_extends_and_reject_set_conflicts() -> None:
    harness = vf.Harness()
    state = vf.State()

    harness.apply_bound_updates(
        state,
        [
            BoundUpdate("state.extras.events", [{"name": "a"}], "extend"),
            BoundUpdate("state.extras.events", [{"name": "b"}], "extend"),
        ],
    )
    assert sorted(event["name"] for event in state.extras["events"]) == ["a", "b"]

    with pytest.raises(ValueError, match="Conflicting bound state updates"):
        harness.apply_bound_updates(
            state,
            [
                BoundUpdate("state.extras.profile", {"name": "a"}),
                BoundUpdate("state.extras.profile.name", "b"),
            ],
        )

    with pytest.raises(ValueError, match="state.advantage"):
        harness.apply_bound_updates(state, [BoundUpdate("state.advantage", 1.0)])


def test_bound_state_updates_write_latest_turn_reward() -> None:
    harness = vf.Harness()
    state = vf.State(
        transcript=[
            vf.Turn(
                prompt=[vf.UserMessage(content="go")],
                completion=[vf.AssistantMessage(content="done")],
            )
        ]
    )

    harness.apply_bound_updates(
        state,
        [BoundUpdate("state.transcript.last.reward", 0.75)],
    )

    assert state.reward == 0.0
    assert state.transcript[-1].reward == 0.75


def test_state_to_output_rejects_metric_reserved_field_collision() -> None:
    state = vf.State(metrics={"reward": 1.0})
    task = vf.Task(prompt="hello")

    with pytest.raises(ValueError, match="Metric name 'reward' conflicts"):
        state.to_output(task)


def test_state_to_output_rejects_state_column_metric_collision() -> None:
    state = vf.State(metrics={"custom": 1.0})
    task = vf.Task(prompt="hello")

    with pytest.raises(ValueError, match="State column 'custom' conflicts"):
        state.to_output(task, state_columns=["custom"])


def test_split_result_copies_unbound_content() -> None:
    content: vf.JsonData = {"messages": [{"role": "user", "content": "before"}]}
    result = split_result(content, ToolBinding())

    messages = content["messages"]
    assert isinstance(messages, list)
    first = messages[0]
    assert isinstance(first, dict)
    first["content"] = "after"

    assert result.value == {"messages": [{"role": "user", "content": "before"}]}


def test_parse_anthropic_user_messages_preserves_structured_content() -> None:
    messages = parse_anthropic_user_messages(
        [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abc",
                },
            }
        ]
    )

    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, vf.UserMessage)
    assert message.content == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc",
            },
        }
    ]


def test_harbor_runtime_preserves_configured_resources_without_env_override() -> None:
    from tasksets.harbor import harbor_runtime

    runtime = harbor_runtime(
        {
            "image": "python:3.13",
            "cpu_cores": 8.0,
            "memory_gb": 12.0,
            "disk_size_gb": 99.0,
        },
        environment={},
        agent_config={},
    )

    assert runtime["cpu_cores"] == 8.0
    assert runtime["memory_gb"] == 12.0
    assert runtime["disk_size_gb"] == 99.0


@pytest.mark.asyncio
async def test_score_group_empty_group_returns_empty_states() -> None:
    env = vf.Env(taskset=NanoTaskset())

    assert await env.score_group([], []) == []


@pytest.mark.asyncio
async def test_run_rollout_retries_copy_supplied_state() -> None:
    class RetryHarness(vf.Harness):
        attempts: int = 0

        async def run(
            self,
            task: vf.Task | str,
            state: vf.State | None = None,
            *,
            model: vf.ModelConfig | str | None = None,
            teacher: vf.ModelConfig | str | None = None,
            context: vf.Context | None = None,
            score: bool = False,
        ) -> vf.State:
            _ = task, model, teacher, context, score
            self.attempts += 1
            rollout_state = state or vf.State()
            rollout_state.transcript.append(vf.Turn(prompt=[], completion=[]))
            if self.attempts == 1:
                raise InfraError("retry")
            return rollout_state

    initial_state = vf.State()
    env = vf.Env(taskset=NanoTaskset(), harness=RetryHarness())

    result = await env.run_rollout(
        NanoTask(prompt="hello", answer="ok"),
        model=vf.ModelConfig(model="student"),
        state=initial_state,
        max_retries=1,
    )

    assert len(result.transcript) == 1
    assert initial_state.transcript == []


@pytest.mark.asyncio
async def test_score_group_closes_model_client_if_teacher_load_fails(
    mock_client,
) -> None:
    env = vf.Env(taskset=NanoTaskset())
    task = NanoTask(prompt="hello", answer="ok")
    state = vf.State(task_id=task.task_id)
    model = vf.ModelConfig(model="student")
    teacher = vf.ModelConfig(model="teacher")
    closed: list[str] = []

    def load_model_client(config: vf.ModelConfig) -> vf.ModelClient:
        if config.model == "teacher":
            raise RuntimeError("teacher failed")
        return vf.ModelClient(config=config, client=mock_client)

    async def close_model_client(model_client: vf.ModelClient) -> None:
        closed.append(model_client.config.model)

    env.harness.load_model_client = load_model_client
    env.harness.close_model_client = close_model_client

    with pytest.raises(RuntimeError, match="teacher failed"):
        await env.score_group([task], [state], model=model, teacher=teacher)

    assert closed == ["student"]


@pytest.mark.asyncio
async def test_score_group_closes_model_clients_if_cleanup_fails(mock_client) -> None:
    env = vf.Env(taskset=NanoTaskset())
    task = NanoTask(prompt="hello", answer="ok")
    state = vf.State(task_id=task.task_id)
    model = vf.ModelConfig(model="student")
    teacher = vf.ModelConfig(model="teacher")
    closed: list[str] = []

    def load_model_client(config: vf.ModelConfig) -> vf.ModelClient:
        return vf.ModelClient(config=config, client=mock_client)

    async def close_model_client(model_client: vf.ModelClient) -> None:
        closed.append(model_client.config.model)

    async def run_handlers_for_group(
        kind: str, *args: object, **kwargs: object
    ) -> None:
        if kind == "cleanup":
            raise RuntimeError("cleanup failed")

    env.harness.load_model_client = load_model_client
    env.harness.close_model_client = close_model_client
    env.run_handlers_for_group = run_handlers_for_group

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await env.score_group([task], [state], model=model, teacher=teacher)

    assert closed == ["teacher", "student"]


@pytest.mark.asyncio
async def test_openenv_and_openreward_rewards_sum_turn_rewards() -> None:
    from tasksets.openenv import OpenEnvTaskset, OpenEnvTasksetConfig
    from tasksets.openreward import OpenRewardTaskset, OpenRewardTasksetConfig

    state = vf.State(
        reward=99.0,
        transcript=[
            vf.Turn(prompt=[], completion=[], reward=0.25),
            vf.Turn(prompt=[], completion=[], reward=0.5),
        ],
    )

    assert (
        await OpenEnvTaskset(OpenEnvTasksetConfig()).openenv_reward(state)
    ) == pytest.approx(0.75)
    assert (
        await OpenRewardTaskset(
            OpenRewardTasksetConfig(environment="test")
        ).openreward_reward(state)
    ) == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_openenv_user_tool_returns_bound_turn_reward_payload() -> None:
    from tasksets.openenv import OpenEnvUser, OpenEnvUserConfig

    class FakeOpenEnvResult:
        observation = {"result": {"data": {"echo": "ok"}}}
        reward = 1.25
        done = True

    class FakeOpenEnvSession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, vf.JsonData]] = []

        async def call_tool(self, name: str, input: vf.JsonData) -> FakeOpenEnvResult:
            self.calls.append((name, input))
            return FakeOpenEnvResult()

    session = FakeOpenEnvSession()
    user = OpenEnvUser(OpenEnvUserConfig())
    user.session = session

    payload = await user.call_tool("echo", {"message": "hi"})

    assert session.calls == [("echo", {"message": "hi"})]
    assert json.loads(str(payload["content"])) == {"echo": "ok"}
    assert payload["openenv_done"] is True
    assert payload["reward"] == pytest.approx(1.25)
    assert payload["finished"] is True
    assert payload["stop_condition"] == "openenv_done"


@pytest.mark.asyncio
async def test_openreward_user_tool_returns_bound_turn_reward_payload() -> None:
    from tasksets.openreward import OpenRewardUser, OpenRewardUserConfig

    class FakeOpenRewardOutput:
        blocks = ["raw"]
        reward = 0.5
        finished = True

    class FakeOpenRewardSession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, vf.JsonData]] = []

        async def call_tool(
            self, name: str, input: vf.JsonData
        ) -> FakeOpenRewardOutput:
            self.calls.append((name, input))
            return FakeOpenRewardOutput()

        def content(self, blocks: object) -> str:
            assert blocks == ["raw"]
            return "judged"

    session = FakeOpenRewardSession()
    user = OpenRewardUser(OpenRewardUserConfig())
    user.session = session

    payload = await user.call_tool("score", {"answer": "ok"})

    assert session.calls == [("score", {"answer": "ok"})]
    assert payload == {
        "content": "judged",
        "reward": 0.5,
        "finished": True,
        "stop_condition": "openreward_finished",
    }


@pytest.mark.asyncio
async def test_v1_subprocess_runtime_session_read_write_run() -> None:
    async with vf.make_runtime_provider(
        vf.SubprocessRuntimeConfig()
    ).create_runtime() as runtime:
        await runtime.write("payload.txt", b"hello runtime")
        result = await runtime.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; print(Path('payload.txt').read_text())",
            ]
        )
        payload = await runtime.read("payload.txt")

    assert result.returncode == 0
    assert result.stdout.strip() == "hello runtime"
    assert payload == b"hello runtime"


@pytest.mark.asyncio
async def test_v1_docker_runtime_read_preserves_binary_bytes() -> None:
    payload = b"\xff\x00binary"

    class ReadOnlyDockerRuntime(vf.DockerRuntime):
        async def run(
            self,
            command: list[str],
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout: float | None = None,
        ) -> vf.CommandResult:
            _ = cwd, env, timeout
            assert command == ["sh", "-c", "base64 < /payload.bin"]
            return vf.CommandResult(
                returncode=0,
                stdout=base64.b64encode(payload).decode(),
            )

    runtime = ReadOnlyDockerRuntime(vf.DockerRuntimeConfig())

    assert await runtime.read("/payload.bin") == payload


def test_v1_runtime_image_is_container_only() -> None:
    task = vf.Task({"prompt": [], "image": "python:3.12-slim"})
    subprocess_env = vf.Env(taskset=NanoTaskset(), runtime=vf.SubprocessRuntimeConfig())
    docker_env = vf.Env(
        taskset=NanoTaskset(),
        runtime=vf.DockerRuntimeConfig(image="python:3.11-slim"),
    )

    with pytest.raises(ValueError, match="declares an image"):
        subprocess_env.harness.runtime_for(task)
    docker_config = docker_env.harness.runtime_for(task)

    assert isinstance(docker_config, vf.DockerRuntimeConfig)
    assert docker_config.image == "python:3.12-slim"


def test_v1_runtime_config_applies_task_resources_with_config_precedence() -> None:
    task = vf.Task(
        {
            "prompt": [],
            "resources": {
                "cpu_cores": 2.0,
                "memory_gb": 4.0,
                "gpu_count": 1,
                "disk_gb": 12.0,
            },
        }
    )
    default_env = vf.Env(taskset=NanoTaskset(), runtime=vf.DockerRuntimeConfig())
    configured_env = vf.Env(
        taskset=NanoTaskset(),
        runtime=vf.DockerRuntimeConfig(cpu_cores=8.0, memory_gb=16.0),
    )
    prime_env = vf.Env(taskset=NanoTaskset(), runtime=vf.PrimeRuntimeConfig())
    subprocess_env = vf.Env(taskset=NanoTaskset(), runtime=vf.SubprocessRuntimeConfig())

    docker_config = default_env.harness.runtime_for(task)
    configured_config = configured_env.harness.runtime_for(task)
    prime_config = prime_env.harness.runtime_for(task)

    assert isinstance(docker_config, vf.DockerRuntimeConfig)
    assert docker_config.cpu_cores == 2.0
    assert docker_config.memory_gb == 4.0
    assert docker_config.gpu_count == 1
    assert docker_config.disk_gb == 12.0

    assert isinstance(configured_config, vf.DockerRuntimeConfig)
    assert configured_config.cpu_cores == 8.0
    assert configured_config.memory_gb == 16.0
    assert configured_config.gpu_count == 1
    assert configured_config.disk_gb == 12.0

    assert isinstance(prime_config, vf.PrimeRuntimeConfig)
    assert prime_config.cpu_cores == 2.0
    assert prime_config.memory_gb == 4.0
    assert prime_config.gpu_count == 1
    assert prime_config.disk_gb == 12.0

    with pytest.raises(ValueError, match="does not support"):
        subprocess_env.harness.runtime_for(task)


def test_v1_default_advantages_fill_turn_tokens() -> None:
    states = [
        vf.State(
            transcript=[
                vf.Turn(
                    prompt=[{"role": "user", "content": "p"}],
                    completion=[{"role": "assistant", "content": "a"}],
                    tokens=vf.TurnTokens(
                        prompt_ids=[1, 2],
                        prompt_mask=[1, 1],
                        completion_ids=[3, 4, 5],
                        completion_mask=[1, 1, 1],
                        completion_logprobs=[0.1, 0.2, 0.3],
                    ),
                )
            ],
            reward=2.0,
        ),
        vf.State(
            transcript=[
                vf.Turn(
                    prompt=[{"role": "user", "content": "p"}],
                    completion=[{"role": "assistant", "content": "b"}],
                    tokens=vf.TurnTokens(
                        prompt_ids=[6],
                        prompt_mask=[1],
                        completion_ids=[7, 8],
                        completion_mask=[1, 1],
                        completion_logprobs=[0.4, 0.5],
                    ),
                )
            ],
            reward=0.0,
        ),
    ]

    tasks = [vf.Task(prompt="p"), vf.Task(prompt="p")]
    vf.advantages.grpo(tasks, states)

    assert states[0].transcript[0].tokens is not None
    assert states[0].transcript[0].tokens.prompt_advantages == pytest.approx([0.0, 0.0])
    assert states[0].transcript[0].tokens.completion_advantages == pytest.approx(
        [1.0, 1.0, 1.0]
    )

    vf.advantages.rl(tasks, states)
    assert states[0].transcript[0].tokens.completion_advantages == pytest.approx(
        [1.0, 1.0, 1.0]
    )

    vf.advantages.sft(tasks, states)
    assert states[0].transcript[0].tokens.prompt_advantages == pytest.approx([1.0, 1.0])
    assert states[0].transcript[0].tokens.completion_advantages == pytest.approx(
        [1.0, 1.0, 1.0]
    )


def test_env_advantage_defaults_to_rl() -> None:
    env = vf.Env(taskset=NanoTaskset())

    assert env.advantage == "rl"
    assert env.provides_advantages
    assert env.requires_group_rollouts


@pytest.mark.asyncio
async def test_env_advantage_config_sets_group_default() -> None:
    tasks = [vf.Task(prompt="p"), vf.Task(prompt="p")]
    states = [
        vf.State(
            transcript=[
                vf.Turn(
                    prompt=[{"role": "user", "content": "p"}],
                    completion=[{"role": "assistant", "content": "a"}],
                    tokens=vf.TurnTokens(
                        prompt_ids=[1],
                        prompt_mask=[1],
                        completion_ids=[2, 3],
                        completion_mask=[1, 1],
                        completion_logprobs=[0.1, 0.2],
                    ),
                )
            ],
            reward=2.0,
        ),
        vf.State(
            transcript=[
                vf.Turn(
                    prompt=[{"role": "user", "content": "p"}],
                    completion=[{"role": "assistant", "content": "b"}],
                    tokens=vf.TurnTokens(
                        prompt_ids=[4],
                        prompt_mask=[1],
                        completion_ids=[5],
                        completion_mask=[1],
                        completion_logprobs=[0.3],
                    ),
                )
            ],
            reward=0.0,
        ),
    ]
    env = vf.Env(
        taskset=NanoTaskset(),
        advantage="reinforce",
    )

    await env.score_group(tasks, states)

    assert states[0].transcript[0].tokens is not None
    assert states[0].transcript[0].tokens.completion_advantages == pytest.approx(
        [2.0, 2.0]
    )
    assert states[1].transcript[0].tokens is not None
    assert states[1].transcript[0].tokens.completion_advantages == pytest.approx([0.0])


@pytest.mark.asyncio
async def test_env_advantage_path_supports_user_authored_group_logic() -> None:
    tasks = [vf.Task(prompt="p"), vf.Task(prompt="p")]
    states = [
        vf.State(
            transcript=[
                vf.Turn(
                    prompt=[{"role": "user", "content": "p"}],
                    completion=[{"role": "assistant", "content": "a"}],
                    tokens=vf.TurnTokens(
                        prompt_ids=[1],
                        prompt_mask=[1],
                        completion_ids=[2],
                        completion_mask=[1],
                        completion_logprobs=[0.1],
                    ),
                )
            ],
            reward=2.0,
        ),
        vf.State(
            transcript=[
                vf.Turn(
                    prompt=[{"role": "user", "content": "p"}],
                    completion=[{"role": "assistant", "content": "b"}],
                    tokens=vf.TurnTokens(
                        prompt_ids=[3],
                        prompt_mask=[1],
                        completion_ids=[4],
                        completion_mask=[1],
                        completion_logprobs=[0.2],
                    ),
                )
            ],
            reward=0.0,
        ),
    ]
    env = vf.Env(taskset=NanoTaskset(), advantage=f"{__name__}:custom_env_advantage")

    await env.score_group(tasks, states)

    assert states[0].transcript[0].tokens is not None
    assert states[0].transcript[0].tokens.completion_advantages == pytest.approx([10.0])
    assert states[1].transcript[0].tokens is not None
    assert states[1].transcript[0].tokens.completion_advantages == pytest.approx([11.0])


def test_v1_runtime_default_resolves_taskset_and_harness_fields() -> None:
    class RuntimeTasksetConfig(vf.TasksetConfig):
        runtime: vf.RuntimeConfig | None = vf.DockerRuntimeConfig(
            image="python:3.12-slim"
        )

    class RuntimeTaskset(vf.Taskset[RuntimeTasksetConfig]):
        config: RuntimeTasksetConfig

        def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
            return NanoTaskset().load_tasks(split)

    class RuntimeHarnessConfig(vf.HarnessConfig):
        runtime: vf.RuntimeConfig | None = vf.DockerRuntimeConfig(workdir="/workspace")

    class RuntimeHarness(vf.Harness[RuntimeHarnessConfig]):
        pass

    env = vf.Env(taskset=RuntimeTaskset(), harness=RuntimeHarness())

    assert isinstance(env.runtime_config, vf.DockerRuntimeConfig)
    assert env.runtime_config.image == "python:3.12-slim"
    assert env.runtime_config.workdir == "/workspace"


def test_v1_runtime_default_rejects_provider_conflicts() -> None:
    class RuntimeTasksetConfig(vf.TasksetConfig):
        runtime: vf.RuntimeConfig | None = vf.DockerRuntimeConfig()

    class RuntimeTaskset(vf.Taskset[RuntimeTasksetConfig]):
        config: RuntimeTasksetConfig

        def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
            return NanoTaskset().load_tasks(split)

    class RuntimeHarnessConfig(vf.HarnessConfig):
        runtime: vf.RuntimeConfig | None = vf.PrimeRuntimeConfig()

    class RuntimeHarness(vf.Harness[RuntimeHarnessConfig]):
        pass

    with pytest.raises(ValueError, match="single provider type"):
        vf.Env(taskset=RuntimeTaskset(), harness=RuntimeHarness())


@pytest.mark.asyncio
async def test_v1_interception_supports_custom_protocols(mock_client) -> None:
    class CustomProtocol(vf.EndpointProtocol):
        name = "custom_json"
        routes = (vf.ProtocolRoute("POST", "/custom/generate"),)

        def env(self, *, base_url: str, api_key: str, model: str) -> dict[str, str]:
            return {
                "CUSTOM_BASE_URL": base_url,
                "CUSTOM_API_KEY": api_key,
                "CUSTOM_MODEL": model,
            }

        async def parse(
            self, request: web.Request, body: vf.JsonData
        ) -> vf.InterceptedRequest:
            _ = request
            prompt = body.get("input")
            if not isinstance(prompt, str):
                raise TypeError("input must be a string.")
            sampling_args: dict[str, vf.JsonValue] = {}
            temperature = body.get("temperature")
            if isinstance(temperature, int | float) and not isinstance(
                temperature, bool
            ):
                sampling_args["temperature"] = temperature
            return vf.InterceptedRequest(
                protocol=self.name,
                prompt=[vf.UserMessage(content=prompt)],
                model=body.get("model") if isinstance(body.get("model"), str) else None,
                sampling_args=sampling_args,
                body=body,
            )

        def serialize(
            self, response: Response, request: vf.InterceptedRequest
        ) -> vf.JsonData:
            content = response.message.content
            if not isinstance(content, str):
                content = ""
            return {
                "protocol": request.protocol,
                "text": content,
                "model": response.model or request.model or "",
            }

    task = vf.Task(prompt=[])
    state = vf.State(task_id=task.task_id)
    model = vf.ModelConfig(client=ClientConfig(), model="fallback-model")
    ctx = vf.Context(
        task=task,
        state=state,
        model_client=vf.ModelClient(config=model, client=mock_client),
    )

    async with vf.InterceptionServer(
        ctx, task, state, protocols=[CustomProtocol()]
    ) as server:
        url = f"http://127.0.0.1:{server.port}/custom/generate"
        async with ClientSession() as session:
            response = await session.post(
                url,
                headers={"Authorization": f"Bearer {server.secret}"},
                json={"input": "hello protocol", "model": "custom-model"},
            )
            payload = await response.json()

    assert response.status == 200
    assert payload == {
        "protocol": "custom_json",
        "text": "This is a test response",
        "model": "test-model",
    }
    assert mock_client.last_call_kwargs["model"] == "custom-model"
    assert mock_client.last_call_kwargs["prompt"][0].content == "hello protocol"
    assert state.transcript[0].prompt[0].content == "hello protocol"
    assert state.completion[-1].content == "This is a test response"


@pytest.mark.asyncio
async def test_eval_runner_uses_v1_native_rollouts(
    tmp_path, monkeypatch, mock_client
) -> None:
    mock_client.set_default_response("ok")
    env = vf.Env(taskset=NanoTaskset(), harness=vf.Harness())
    env.env_id = "nano-env"
    attach_mock_model(env, mock_client)

    monkeypatch.setattr(eval_utils.vf, "load_environment", lambda **_: env)

    config = EvalConfig(
        env_id="nano-env",
        env_args={},
        env_dir_path="./environments",
        model="test-model",
        client_config=ClientConfig(),
        sampling_args={},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
        output_dir=str(tmp_path),
        save_results=True,
    )

    results = await eval_utils.run_evaluation(config)
    result_path = results["metadata"]["path_to_save"]

    assert results["outputs"][0]["reward"] == 1.0
    assert results["outputs"][0]["transcript"][0]["completion"][0]["content"] == "ok"
    assert (result_path / "results.jsonl").exists()
    assert (result_path / "metadata.json").exists()

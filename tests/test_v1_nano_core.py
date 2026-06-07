from __future__ import annotations

import sys
import asyncio
from types import ModuleType

from aiohttp import ClientSession, web
import pytest

import verifiers.v1 as vf
from verifiers.types import ClientConfig, EvalConfig, Response
from verifiers.utils import eval_utils
from verifiers.v1.loaders import (
    load_environment_from_components,
    load_harness_from_module,
    load_taskset_from_module,
)
from verifiers.v1.mcp import BoundUpdate, MCPToolRegistry


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
            GroupNanoTask.model_validate({**task.to_record(), "candidate": index})
            for index in range(num_rollouts)
        ]
        return tasks, [vf.State(task_id=task.task_id) for task in tasks]

    @vf.reward(stage="group")
    async def relative(
        self, tasks: list[GroupNanoTask], states: list[vf.State]
    ) -> list[float]:
        _ = states
        return [float(task.candidate == 0) for task in tasks]

    grpo = staticmethod(vf.advantages.grpo)


class EmptyPromptUser(vf.User):
    @vf.tool
    def respond(self) -> dict:
        return {"messages": [{"role": "user", "content": "server prompt"}]}


class EmptyPromptTasksetConfig(vf.TasksetConfig):
    user: vf.UserConfig | None = vf.UserConfig(
        loader=f"{EmptyPromptUser.__module__}:EmptyPromptUser"
    )


class EmptyPromptTaskset(vf.Taskset[EmptyPromptTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [{"example_id": 0, "prompt": [], "max_turns": 1}]


class PatchOnlyUser(vf.User):
    @vf.tool(
        sets={
            "done": "state.extras.done",
            "stop_condition": "state.stop_condition",
        }
    )
    def respond(self) -> dict:
        return {"done": True, "stop_condition": "user_bootstrap_done"}


class PatchOnlyUserTasksetConfig(vf.TasksetConfig):
    user: vf.UserConfig | None = vf.UserConfig(
        loader=f"{PatchOnlyUser.__module__}:PatchOnlyUser"
    )


class PatchOnlyUserTaskset(vf.Taskset[PatchOnlyUserTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [{"example_id": 0, "prompt": [], "max_turns": 1}]


class TaskSetupToolset(vf.Toolset):
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

    def load_toolsets(self, config: vf.TasksetConfig) -> list[vf.ToolsetConfig]:
        _ = config
        return [
            vf.ToolsetConfig(
                loader=f"{TaskSetupToolset.__module__}:TaskSetupToolset",
                name="setup",
                scope="env",
                hide=["materialize"],
            )
        ]

    @vf.setup
    async def materialize_task(
        self,
        task: vf.Task,
        state: vf.State,
        harness: vf.Harness,
        runtime: vf.Runtime,
        tools: MCPToolRegistry,
    ) -> None:
        _ = task
        result = await tools.call_hidden("materialize", {})
        harness.apply_tool_result(state, result)
        case = state.metadata["grader_case"]
        assert isinstance(case, str)
        await runtime.write("grader_case.txt", case.encode())


class DemoToolset(vf.Toolset):
    @vf.tool
    def echo(self, text: str) -> str:
        return text.upper()

    @vf.tool
    def suffix(self, text: str) -> str:
        return text + "!"


class BoundToolset(vf.Toolset):
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


@pytest.mark.asyncio
async def test_v1_model_client_uses_serialized_transcript_record(mock_client) -> None:
    config = vf.ModelConfig(client=ClientConfig(), model="test-model")
    state = vf.State(task_id="task-1")
    state.add_turn(
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
async def test_v1_group_rewards_and_advantages_apply_to_turns(mock_client) -> None:
    mock_client.set_default_response("ok")
    env = vf.Env(taskset=GroupTaskset(), harness=vf.Harness())
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
    assert sum(float(output["advantage"]) for output in outputs) == pytest.approx(0.0)
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
            "config": vf.EnvConfig(
                harness=module.HelloRLMHarnessConfig(
                    command=[
                        sys.executable,
                        "-c",
                        "import os; print(os.environ['VF_PROMPT'].split('exactly ', 1)[1].rstrip('.'))",
                    ]
                )
            )
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
async def test_mcp_toolset_exposes_multiple_server_tools() -> None:
    toolset = vf.ToolsetConfig(
        loader=f"{DemoToolset.__module__}:DemoToolset",
        name="demo",
    )

    async with MCPToolRegistry([toolset]) as registry:
        names = [tool.name for tool in registry.tool_defs() or []]
        result = await registry.call("demo_echo", {"text": "ok"})

    assert names == ["demo_echo", "demo_suffix"]
    assert result.response.messages[0].content == "OK"


@pytest.mark.asyncio
async def test_mcp_bindings_hide_args_and_bind_returns() -> None:
    toolset = vf.ToolsetConfig(
        loader=f"{BoundToolset.__module__}:BoundToolset",
        name="bound",
    )
    state = vf.State()
    task = vf.Task(name="demo", prompt="say ok")
    harness = vf.Harness()

    async with MCPToolRegistry([toolset]) as registry:
        registry.set_context(harness.binding_context(task, state))
        tools = registry.tool_defs() or []
        result = await registry.call("bound_record", {"name": "alpha"})

    properties = tools[0].parameters["properties"]
    assert "name" in properties
    assert "task_name" not in properties
    assert result.response.messages[0].content == "demo:alpha"

    harness.apply_bound_updates(state, list(result.updates))
    assert state.extras == {
        "events": [{"task": "demo", "name": "alpha"}],
        "profile": {"task": "demo"},
    }


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


@pytest.mark.asyncio
async def test_v1_local_runtime_session_read_write_run() -> None:
    async with vf.make_runtime_provider(
        vf.LocalRuntimeConfig()
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


def test_v1_runtime_image_is_container_only() -> None:
    task = vf.Task({"prompt": [], "image": "python:3.12-slim"})
    local_env = vf.Env(taskset=NanoTaskset(), runtime=vf.LocalRuntimeConfig())
    docker_env = vf.Env(
        taskset=NanoTaskset(),
        runtime=vf.DockerRuntimeConfig(image="python:3.11-slim"),
    )

    with pytest.raises(ValueError, match="declares an image"):
        local_env.harness.runtime_for(task)
    docker_config = docker_env.harness.runtime_for(task)

    assert isinstance(docker_config, vf.DockerRuntimeConfig)
    assert docker_config.image == "python:3.12-slim"


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

    assert [state.advantage for state in states] == pytest.approx([1.0, -1.0])
    assert states[0].transcript[0].tokens is not None
    assert states[0].transcript[0].tokens.prompt_advantages == pytest.approx([0.0, 0.0])
    assert states[0].transcript[0].tokens.completion_advantages == pytest.approx(
        [1.0, 1.0, 1.0]
    )


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

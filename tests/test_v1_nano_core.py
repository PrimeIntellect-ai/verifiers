from __future__ import annotations

import sys
import asyncio

from aiohttp import ClientSession, web
import pytest

import verifiers.v1 as vf
from verifiers.types import ClientConfig, EvalConfig, Response
from verifiers.utils import eval_utils
from verifiers.v1.mcp import MCPToolRegistry


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
    async def setup_scratch(self, state: vf.State) -> None:
        state.scratch["seen_setup"] = True

    @vf.reward
    async def exact(self, task: NanoTask, state: vf.State) -> float:
        completion = state.completion
        text = str(completion[-1].content if completion else "")
        return float(text.strip() == task.answer)


class CandidateNanoTask(NanoTask):
    candidate: int


class GroupTaskset(NanoTaskset):
    async def init_group(
        self, task: NanoTask, num_rollouts: int
    ) -> tuple[list[CandidateNanoTask], list[vf.State]]:
        tasks = [
            CandidateNanoTask.model_validate({**task.to_record(), "candidate": index})
            for index in range(num_rollouts)
        ]
        return tasks, [vf.State(task_id=task.task_id) for task in tasks]

    @vf.reward(stage="group")
    async def relative(
        self, tasks: list[CandidateNanoTask], states: list[vf.State]
    ) -> list[float]:
        _ = states
        return [float(task.candidate == 0) for task in tasks]

    grpo = staticmethod(vf.advantages.grpo)


EMPTY_PROMPT_USER_SCRIPT = """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("bootstrap-user")

@mcp.tool()
def respond(task: dict, state: dict, transcript: list[dict]) -> dict:
    return {"messages": [{"role": "user", "content": "server prompt"}]}

mcp.run(transport="stdio")
"""


class EmptyPromptTasksetConfig(vf.TasksetConfig):
    user: vf.User | None = vf.User(
        server=vf.MCPServerSpec(
            command=[sys.executable, "-c", EMPTY_PROMPT_USER_SCRIPT]
        )
    )


class EmptyPromptTaskset(vf.Taskset[EmptyPromptTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [{"example_id": 0, "prompt": [], "max_turns": 1}]


def test_v1_state_is_pydantic_and_scratch_owned() -> None:
    task = NanoTask(prompt="hello", answer="world")
    state = vf.State(task_id=task.task_id)

    state.scratch["x"] = 1
    state.reward += 0.5
    state.assert_serializable()

    assert state.scratch == {"x": 1}
    assert state.reward == 0.5
    with pytest.raises(TypeError):
        state["x"] = 2
    assert state.to_output(task)["transcript"] == []


@pytest.mark.asyncio
async def test_v1_model_client_derives_legacy_client_record(mock_client) -> None:
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

    await vf.ModelClient(config=config, client=mock_client).get_response(
        prompt=[vf.UserMessage(content="next")],
        state=state,
    )

    client_state = mock_client.last_call_kwargs["state"]
    assert "transcript" not in client_state
    assert client_state["task_id"] == "task-1"
    assert client_state["trajectory"][0]["prompt"][0].content == "first"
    assert client_state["trajectory"][0]["tokens"]["completion_ids"] == [2]


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
async def test_migrated_hello_rlm_v1_runs_without_old_runtime(mock_client) -> None:
    from environments.hello_rlm_v1.hello_rlm_v1 import load_environment

    env = load_environment(vf.EnvConfig())
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
    script = """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")

@mcp.tool()
def echo(text: str) -> str:
    return text.upper()

@mcp.tool()
def suffix(text: str) -> str:
    return text + "!"

mcp.run(transport="stdio")
"""
    toolset = vf.Toolset(
        name="demo",
        server=vf.MCPServerSpec(command=[sys.executable, "-c", script]),
    )

    async with MCPToolRegistry([toolset]) as registry:
        names = [tool.name for tool in registry.tool_defs() or []]
        result = await registry.call("demo_echo", {"text": "ok"})

    assert names == ["demo_echo", "demo_suffix"]
    assert result == "OK"


@pytest.mark.asyncio
async def test_v1_local_runtime_session_read_write_run() -> None:
    async with vf.make_runtime_provider(vf.LocalRuntimeConfig()).session() as runtime:
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
    ctx = vf.RolloutContext(
        model_client=vf.ModelClient(config=model, client=mock_client)
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

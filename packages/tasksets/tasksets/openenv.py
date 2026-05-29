import asyncio
import importlib.util
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import verifiers as vf
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
)
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.message_utils import get_messages, normalize_messages
from verifiers.utils.tool_utils import is_valid_tool_content_parts
from verifiers.v1.config import import_config_ref
from verifiers.v1.utils.serialization_utils import serializable

from tasksets.utils.openenv_utils import PrimeSandboxOpenEnvProvider


def default_openenv_prompt_renderer(
    observation: object,
    *,
    context: str = "reset",
    action_schema: vf.ConfigData | None = None,
    contract: str = "gym",
    seed: int = 0,
) -> object:
    del context, action_schema, contract, seed
    if isinstance(observation, str):
        return [{"role": "user", "content": observation}]
    if isinstance(observation, Mapping):
        observation_map = cast(vf.ConfigMap, observation)
        messages = observation_map.get("messages")
        if messages is not None:
            return messages
        for key in ("prompt", "question", "instruction", "content", "text"):
            value = observation_map.get(key)
            if isinstance(value, str) and value.strip():
                return [{"role": "user", "content": value}]
        return [{"role": "user", "content": json.dumps(serializable(observation))}]
    return [{"role": "user", "content": str(observation)}]


class OpenEnvRuntimeConfig(vf.Config):
    openenv_project: str
    prompt_renderer: str
    image: str
    port: int
    start_command: str
    contract: Literal["gym", "mcp"]
    seed: int
    startup_timeout_seconds: int
    startup_poll_interval_seconds: float
    health_request_timeout_seconds: float
    schema_request_timeout_seconds: float
    wait_for_creation_max_attempts: int
    max_retries: int
    base_delay: float
    backoff_factor: float
    max_backoff_seconds: float
    jitter: float


class OpenEnvBuildConfig(vf.Config):
    image: str
    port: int = 8000
    start_command: str
    contract: Literal["gym", "mcp"]


class OpenEnvUserConfig(vf.UserConfig):
    pass


class OpenEnvTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "openenv"
    user: OpenEnvUserConfig | None = None
    prompt_renderer: str = "tasksets.openenv:default_openenv_prompt_renderer"
    openenv_project: str = "proj"
    num_train_examples: int = 100
    num_eval_examples: int = 50
    seed: int = 0
    startup_timeout_seconds: int = 30
    startup_poll_interval_seconds: float = 1.0
    health_request_timeout_seconds: float = 2.0
    schema_request_timeout_seconds: float = 5.0
    wait_for_creation_max_attempts: int = 20
    max_retries: int = 5
    base_delay: float = 0.5
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 30.0
    jitter: float = 1e-3


class OpenEnvTaskset(vf.Taskset[OpenEnvTasksetConfig]):
    def load_user(self) -> vf.UserConfig:
        return self.config.user or OpenEnvUserConfig()

    def load_toolsets(self) -> vf.Toolsets:
        return {"openenv": vf.Toolset(scope="rollout", handler=self.call_tool)}

    def load_tasks(self, split: vf.TaskSplit = "train") -> list[vf.ConfigData]:
        num_examples = (
            self.config.num_train_examples
            if split == "train"
            else self.config.num_eval_examples
        )
        if num_examples <= 0:
            return []
        project = Path(self.config.openenv_project).expanduser()
        if not project.is_absolute():
            spec = importlib.util.find_spec(type(self.config).__module__)
            assert spec is not None
            assert spec.origin is not None
            project = Path(spec.origin).parent / project
        project = project.resolve()
        build = OpenEnvBuildConfig.model_validate(
            json.loads((project / ".build.json").read_text())
        )
        runtime_config = OpenEnvRuntimeConfig(
            openenv_project=str(project),
            prompt_renderer=self.config.prompt_renderer,
            image=build.image,
            port=build.port,
            start_command=build.start_command,
            contract=build.contract,
            seed=self.config.seed,
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            startup_poll_interval_seconds=self.config.startup_poll_interval_seconds,
            health_request_timeout_seconds=self.config.health_request_timeout_seconds,
            schema_request_timeout_seconds=self.config.schema_request_timeout_seconds,
            wait_for_creation_max_attempts=self.config.wait_for_creation_max_attempts,
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            backoff_factor=self.config.backoff_factor,
            max_backoff_seconds=self.config.max_backoff_seconds,
            jitter=self.config.jitter,
        )
        first_seed = (
            self.config.seed
            if split == "train"
            else self.config.seed + self.config.num_train_examples
        )
        return [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "OpenEnv rollout is initializing.",
                    }
                ],
                "openenv": runtime_config.model_copy(
                    update={"seed": first_seed + index}
                ).model_dump(),
                "info": {"seed": first_seed + index, "contract": build.contract},
            }
            for index in range(num_examples)
        ]

    @vf.setup
    async def setup_openenv(self, task: vf.Task, state: vf.State) -> None:
        task_config = task["openenv"]
        assert isinstance(task_config, Mapping)
        config = OpenEnvRuntimeConfig.model_validate(task_config)
        provider = PrimeSandboxOpenEnvProvider(config)
        client_class = MCPToolClient if config.contract == "mcp" else GenericEnvClient
        try:
            client = await client_class.from_docker_image(
                config.image,
                provider=provider,
                port=config.port,
                start_command=config.start_command,
                env_vars={"ENABLE_WEB_INTERFACE": "false"},
            )
            schema = await asyncio.to_thread(provider.fetch_schema)
        except Exception:
            provider.stop_container()
            raise
        action_schema = schema.get("action", {}) if isinstance(schema, Mapping) else {}
        assert isinstance(action_schema, Mapping)
        state["openenv_client"] = client
        state["openenv_contract"] = config.contract
        state["openenv_action_schema"] = dict(action_schema)
        result = await client.reset(seed=config.seed)
        if config.contract == "mcp":
            assert isinstance(client, MCPToolClient)
            for tool in await client.list_tools():
                schema = tool.input_schema or {"type": "object", "properties": {}}
                tool_def = vf.Tool(
                    name=tool.name,
                    description=tool.description,
                    parameters={str(key): value for key, value in schema.items()},
                )
                state.add_tool("openenv", tool_def)
        state["openenv_done"] = bool(result.done)
        renderer = import_config_ref(config.prompt_renderer)
        assert callable(renderer)
        rendered = await maybe_await(
            cast(Callable[..., object], renderer),
            result.observation,
            context="reset",
            action_schema=dict(action_schema),
            contract=config.contract,
            seed=config.seed,
        )
        state["prompt"] = normalize_messages(
            cast(vf.PromptInput, rendered), field_name="openenv"
        )

    @vf.cleanup
    async def cleanup_openenv(self, state: vf.State) -> None:
        client = state.pop("openenv_client", None)
        state.pop("openenv_action_schema", None)
        state.pop("openenv_contract", None)
        if client is not None:
            assert isinstance(client, GenericEnvClient | MCPToolClient)
            await maybe_await(client.close)

    @vf.stop
    async def openenv_done(self, state: vf.State) -> bool:
        return bool(state.get("openenv_done"))

    @vf.reward(weight=1.0)
    async def openenv_reward(self, state: vf.State) -> float:
        return float(
            sum(
                float(step.get("reward", 0.0) or 0.0)
                for step in state.get("trajectory", [])
                if isinstance(step, Mapping)
            )
        )

    async def call_tool(
        self, state: vf.State, tool: vf.Tool, arguments: vf.ConfigData
    ) -> vf.MessageContent:
        client = state["openenv_client"]
        assert isinstance(client, MCPToolClient)
        result = await client.step(
            CallToolAction(tool_name=tool.name, arguments=dict(arguments))
        )
        if result.reward is not None:
            trajectory = state["trajectory"]
            assert isinstance(trajectory, list)
            step = trajectory[-1]
            assert isinstance(step, dict)
            step["reward"] = float(step.get("reward", 0.0) or 0.0) + float(
                result.reward
            )
        state["openenv_done"] = bool(result.done)
        if result.done:
            state.stop("openenv_done")
        observation = result.observation
        if isinstance(observation, CallToolObservation):
            content: object = (
                {"error": observation.error.message}
                if observation.error is not None
                else observation.result.data
            )
        elif isinstance(observation, Mapping):
            observation_map = cast(vf.ConfigMap, observation)
            if observation_map.get("error") is not None:
                content = {"error": observation_map.get("error")}
            else:
                result_value = observation_map["result"]
                assert isinstance(result_value, Mapping)
                result_data = cast(vf.ConfigMap, result_value)
                content = result_data["data"]
        else:
            content = observation
        if is_valid_tool_content_parts(content):
            return cast(vf.MessageContent, content)
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True)


class OpenEnvUser(vf.User[OpenEnvUserConfig]):
    async def get_response(
        self, task: vf.Task, state: vf.State, messages: Sequence[vf.Message]
    ) -> list[vf.UserMessage]:
        task_config = task["openenv"]
        assert isinstance(task_config, Mapping)
        config = OpenEnvRuntimeConfig.model_validate(task_config)
        if config.contract == "mcp":
            return []
        assistant_messages = get_messages(messages, role="assistant")
        last_message = assistant_messages[-1] if assistant_messages else None
        text = str(last_message.content or "").strip() if last_message else ""
        action = json.loads(text)
        assert isinstance(action, Mapping)
        client = state["openenv_client"]
        assert isinstance(client, GenericEnvClient)
        result = await client.step(action)
        if result.reward is not None:
            trajectory = state["trajectory"]
            assert isinstance(trajectory, list)
            step = trajectory[-1]
            assert isinstance(step, dict)
            step["reward"] = float(step.get("reward", 0.0) or 0.0) + float(
                result.reward
            )
        state["openenv_done"] = bool(result.done)
        if result.done:
            state.stop("openenv_done")
        schema = state["openenv_action_schema"]
        assert isinstance(schema, Mapping)
        renderer = import_config_ref(config.prompt_renderer)
        assert callable(renderer)
        rendered = await maybe_await(
            cast(Callable[..., object], renderer),
            result.observation,
            context="step",
            action_schema=dict(schema),
            contract=config.contract,
            seed=config.seed,
        )
        return cast(
            list[vf.UserMessage],
            normalize_messages(cast(vf.PromptInput, rendered), field_name="openenv"),
        )


def load_taskset(config: OpenEnvTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)

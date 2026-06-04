import asyncio
import importlib.util
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

import verifiers as vf
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    Tool as OpenEnvToolSpec,
)
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.message_utils import get_messages, normalize_messages
from verifiers.utils.tool_utils import is_valid_tool_content_parts
from verifiers.v1.config import import_config_ref
from verifiers.v1.utils.serialization_utils import serializable

from tasksets.utils.openenv_utils import PrimeSandboxOpenEnvProvider

OpenEnvPromptRenderer: TypeAlias = Callable[
    ..., vf.PromptInput | Awaitable[vf.PromptInput]
]


class OpenEnvResult(Protocol):
    observation: object
    reward: float | int | None
    done: bool


def default_openenv_prompt_renderer(
    observation: object,
) -> vf.PromptInput:
    if isinstance(observation, str):
        return [{"role": "user", "content": observation}]
    if isinstance(observation, dict):
        observation_map = cast(vf.JsonData, observation)
        messages = observation_map.get("messages")
        if messages is not None:
            assert isinstance(messages, list)
            return cast(vf.PromptInput, messages)
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
    bindings: vf.BindingsConfig = vf.BindingsConfig.model_validate(
        {"session": "taskset.objects.session"}
    )


class OpenEnvTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "openenv"
    bindings: vf.BindingsConfig = vf.BindingsConfig.model_validate(
        {"session.task": "task"}
    )
    objects: vf.ObjectsConfig = vf.ObjectsConfig.model_validate(
        {"session": "tasksets.openenv:OpenEnvSession"}
    )
    user: OpenEnvUserConfig | None = OpenEnvUserConfig()
    prompt_renderer: str = "tasksets.openenv:default_openenv_prompt_renderer"
    openenv_project: str = "proj"
    num_train_examples: int = 100
    num_eval_examples: int = 50
    seed: int = 0
    max_retries: int = 5
    base_delay: float = 0.5
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 30.0
    jitter: float = 1e-3


class OpenEnvSession:
    def __init__(self, task: vf.Task):
        task_config = task["openenv"]
        assert isinstance(task_config, dict)
        self.config = OpenEnvRuntimeConfig.model_validate(task_config)
        self.provider: PrimeSandboxOpenEnvProvider | None = None
        self.client: GenericEnvClient | MCPToolClient | None = None
        self.action_schema: vf.JsonData = {}

    async def start(self) -> GenericEnvClient | MCPToolClient:
        if self.client is not None:
            return self.client
        config = self.config
        provider = PrimeSandboxOpenEnvProvider(config)
        self.provider = provider
        client_class = MCPToolClient if config.contract == "mcp" else GenericEnvClient
        try:
            self.client = await client_class.from_docker_image(
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
        action_schema = schema.get("action", {})
        assert isinstance(action_schema, dict)
        self.action_schema = cast(vf.JsonData, dict(action_schema))
        return self.client

    async def reset(self) -> OpenEnvResult:
        client = await self.start()
        return cast(OpenEnvResult, await client.reset(seed=self.config.seed))

    async def list_tools(self) -> Sequence[OpenEnvToolSpec]:
        client = await self.start()
        assert isinstance(client, MCPToolClient)
        return await client.list_tools()

    async def call_tool(self, name: str, arguments: vf.JsonData) -> OpenEnvResult:
        client = await self.start()
        assert isinstance(client, MCPToolClient)
        result = await client.step(
            CallToolAction(tool_name=name, arguments=dict(arguments))
        )
        return cast(OpenEnvResult, result)

    async def step(self, action: vf.JsonData) -> OpenEnvResult:
        client = await self.start()
        assert isinstance(client, GenericEnvClient)
        return cast(OpenEnvResult, await client.step(action))

    async def close(self) -> None:
        if self.client is not None:
            await maybe_await(self.client.close)
        self.client = None
        self.provider = None


class OpenEnvTaskset(vf.Taskset[OpenEnvTasksetConfig]):
    def load_toolsets(self, config: OpenEnvTasksetConfig) -> vf.Toolsets:
        return {"openenv": vf.Toolset(scope="rollout", handler=self.call_tool)}

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return self.openenv_tasks(
                num_examples=self.config.num_eval_examples,
                first_seed=self.config.seed + self.config.num_train_examples,
            )
        return self.openenv_tasks(
            num_examples=self.config.num_train_examples,
            first_seed=self.config.seed,
        )

    def openenv_tasks(
        self,
        *,
        num_examples: int,
        first_seed: int,
    ) -> vf.Tasks:
        config = self.config
        if num_examples <= 0:
            return []
        project = Path(config.openenv_project).expanduser()
        if not project.is_absolute():
            spec = importlib.util.find_spec(type(config).__module__)
            assert spec is not None
            assert spec.origin is not None
            project = Path(spec.origin).parent / project
        project = project.resolve()
        build = OpenEnvBuildConfig.model_validate(
            json.loads((project / ".build.json").read_text())
        )
        runtime_config = OpenEnvRuntimeConfig(
            openenv_project=str(project),
            prompt_renderer=config.prompt_renderer,
            image=build.image,
            port=build.port,
            start_command=build.start_command,
            contract=build.contract,
            seed=config.seed,
            max_retries=config.max_retries,
            base_delay=config.base_delay,
            backoff_factor=config.backoff_factor,
            max_backoff_seconds=config.max_backoff_seconds,
            jitter=config.jitter,
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
        session = await self.get_object("session", task, state)
        assert isinstance(session, OpenEnvSession)
        result = await session.reset()
        config = session.config
        if config.contract == "mcp":
            for tool in await session.list_tools():
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
        rendered = await maybe_call_with_named_args(
            cast(OpenEnvPromptRenderer, renderer),
            observation=result.observation,
            context="reset",
            action_schema=dict(session.action_schema),
            contract=config.contract,
            seed=config.seed,
        )
        state["prompt"] = normalize_messages(
            cast(vf.PromptInput, rendered), field_name="openenv"
        )

    @vf.stop
    async def openenv_done(self, state: vf.State) -> bool:
        return bool(state.get("openenv_done"))

    @vf.reward(weight=1.0)
    async def openenv_reward(self, state: vf.State) -> float:
        return state.total_step_reward()

    async def call_tool(
        self, task: vf.Task, state: vf.State, tool: vf.Tool, arguments: vf.JsonData
    ) -> vf.MessageContent:
        session = await self.get_object("session", task, state)
        assert isinstance(session, OpenEnvSession)
        result = await session.call_tool(tool.name, arguments)
        state.add_step_reward(result.reward)
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
        elif isinstance(observation, dict):
            observation_map = cast(vf.JsonData, observation)
            if observation_map.get("error") is not None:
                content = {"error": observation_map.get("error")}
            else:
                result_value = observation_map["result"]
                assert isinstance(result_value, dict)
                result_data = cast(vf.JsonData, result_value)
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
        self,
        task: vf.Task,
        state: vf.State,
        messages: Sequence[vf.Message],
        session: OpenEnvSession | None = None,
    ) -> list[vf.UserMessage]:
        assert session is not None
        config = session.config
        if config.contract == "mcp":
            return []
        assistant_messages = get_messages(messages, role="assistant")
        last_message = assistant_messages[-1] if assistant_messages else None
        text = str(last_message.content or "").strip() if last_message else ""
        action = json.loads(text)
        assert isinstance(action, dict)
        result = await session.step(cast(vf.JsonData, action))
        state.add_step_reward(result.reward)
        state["openenv_done"] = bool(result.done)
        if result.done:
            state.stop("openenv_done")
        renderer = import_config_ref(config.prompt_renderer)
        assert callable(renderer)
        rendered = await maybe_call_with_named_args(
            cast(OpenEnvPromptRenderer, renderer),
            observation=result.observation,
            context="step",
            action_schema=dict(session.action_schema),
            contract=config.contract,
            seed=config.seed,
        )
        response: list[vf.UserMessage] = []
        for message in normalize_messages(
            cast(vf.PromptInput, rendered), field_name="openenv"
        ):
            assert isinstance(message, vf.UserMessage)
            response.append(message)
        return response


def load_taskset(config: OpenEnvTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)

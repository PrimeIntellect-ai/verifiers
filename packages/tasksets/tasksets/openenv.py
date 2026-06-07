import asyncio
import importlib.util
import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

from openenv.core.generic_client import GenericEnvClient
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient
from pydantic import Field, TypeAdapter

import verifiers.v1 as vf
from verifiers.utils.async_utils import maybe_await, maybe_call_with_named_args
from verifiers.v1.config import import_config_ref
from verifiers.v1.utils.json_utils import json_data, json_value

from tasksets.utils.openenv_utils import PrimeSandboxOpenEnvProvider

_MESSAGES_ADAPTER = TypeAdapter(vf.Messages)

OpenEnvPromptRenderer: TypeAlias = Callable[
    ..., vf.PromptInput | Awaitable[vf.PromptInput]
]


class OpenEnvResult(Protocol):
    observation: object
    reward: float | int | None
    done: bool


def default_openenv_prompt_renderer(observation: object) -> vf.PromptInput:
    if isinstance(observation, str):
        return [{"role": "user", "content": observation}]
    if isinstance(observation, dict):
        observation_data = json_data(observation)
        messages = observation_data.get("messages")
        if messages is not None:
            if not isinstance(messages, list):
                raise TypeError("OpenEnv observation messages must be a list.")
            return _MESSAGES_ADAPTER.validate_python(messages)
        for key in ("prompt", "question", "instruction", "content", "text"):
            value = observation_data.get(key)
            if isinstance(value, str) and value.strip():
                return [{"role": "user", "content": value}]
        return [{"role": "user", "content": json.dumps(observation_data)}]
    return [{"role": "user", "content": str(observation)}]


class OpenEnvRuntimeConfig(vf.Config):
    openenv_project: str
    prompt_renderer: str
    image: str
    port: int
    start_command: str
    contract: Literal["gym", "mcp"]
    tools: list[vf.JsonData] = Field(default_factory=list)
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
    app: str | None = None
    image: str
    environment_id: str | None = None
    image_status: str | None = None
    port: int = 8000
    schema_version: int | None = None
    start_command: str
    contract: Literal["gym", "mcp"]
    tools: list[vf.JsonData] = Field(default_factory=list)


class OpenEnvUserConfig(vf.UserConfig):
    pass


class OpenEnvTasksetConfig(vf.TasksetConfig):
    id: str | None = "openenv"
    user: vf.UserConfig | None = OpenEnvUserConfig()
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


class OpenEnvTask(vf.Task, frozen=True):
    openenv: vf.JsonData
    info: vf.JsonData


class OpenEnvSession:
    def __init__(self, config: OpenEnvRuntimeConfig):
        self.config = config
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
            if isinstance(self.client, MCPToolClient):
                self.client.use_production_mode = True
            schema = await asyncio.to_thread(provider.fetch_schema)
        except Exception:
            provider.stop_container()
            raise
        action_schema = schema.get("action", {})
        if isinstance(action_schema, dict):
            self.action_schema = json_data(action_schema)
        return self.client

    async def reset(self) -> OpenEnvResult:
        client = await self.start()
        return cast(OpenEnvResult, await client.reset(seed=self.config.seed))

    async def step(self, action: vf.JsonData) -> OpenEnvResult:
        client = await self.start()
        if not isinstance(client, GenericEnvClient):
            raise RuntimeError("MCP OpenEnv tasks require an MCP tool server.")
        return cast(OpenEnvResult, await client.step(action))

    async def call_tool(self, name: str, input: vf.JsonData) -> OpenEnvResult:
        client = await self.start()
        if not isinstance(client, MCPToolClient):
            raise RuntimeError("Gym OpenEnv tasks require assistant JSON actions.")
        return cast(
            OpenEnvResult,
            await client.step(CallToolAction(tool_name=name, arguments=input)),
        )

    async def tool_defs(self) -> list[vf.JsonData]:
        client = await self.start()
        if not isinstance(client, MCPToolClient):
            return []
        return openenv_tool_defs(await client.list_tools())

    async def close(self) -> None:
        if self.client is not None:
            await maybe_await(self.client.close)
        if self.provider is not None:
            self.provider.stop_container()
        self.client = None
        self.provider = None


class OpenEnvTaskset(vf.Taskset[OpenEnvTasksetConfig]):
    task_type = OpenEnvTask

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

    def openenv_tasks(self, *, num_examples: int, first_seed: int) -> vf.Tasks:
        if num_examples <= 0:
            return []
        project = self.openenv_project_path()
        build = OpenEnvBuildConfig.model_validate(
            json.loads((project / ".build.json").read_text())
        )
        return [
            {
                "prompt": [],
                "openenv": self.runtime_config(
                    project=project, build=build, seed=first_seed + index
                ).model_dump(mode="json"),
                "info": {"seed": first_seed + index, "contract": build.contract},
            }
            for index in range(num_examples)
        ]

    def openenv_project_path(self) -> Path:
        project = Path(self.config.openenv_project).expanduser()
        if project.is_absolute():
            return project.resolve()
        spec = importlib.util.find_spec(type(self.config).__module__)
        if spec is None or spec.origin is None:
            return project.resolve()
        return (Path(spec.origin).parent / project).resolve()

    def runtime_config(
        self, *, project: Path, build: OpenEnvBuildConfig, seed: int
    ) -> OpenEnvRuntimeConfig:
        config = self.config
        return OpenEnvRuntimeConfig(
            openenv_project=str(project),
            prompt_renderer=config.prompt_renderer,
            image=build.image,
            port=build.port,
            start_command=build.start_command,
            contract=build.contract,
            tools=list(build.tools),
            seed=seed,
            startup_timeout_seconds=config.startup_timeout_seconds,
            startup_poll_interval_seconds=config.startup_poll_interval_seconds,
            health_request_timeout_seconds=config.health_request_timeout_seconds,
            schema_request_timeout_seconds=config.schema_request_timeout_seconds,
            wait_for_creation_max_attempts=config.wait_for_creation_max_attempts,
            max_retries=config.max_retries,
            base_delay=config.base_delay,
            backoff_factor=config.backoff_factor,
            max_backoff_seconds=config.max_backoff_seconds,
            jitter=config.jitter,
        )

    @vf.reward(weight=1.0)
    async def openenv_reward(self, state: vf.State) -> float:
        return sum(float(turn.reward or 0.0) for turn in state.transcript)


async def render_messages(
    config: OpenEnvRuntimeConfig,
    session: OpenEnvSession,
    observation: object,
    context: str,
) -> list[vf.JsonData]:
    renderer = import_config_ref(config.prompt_renderer)
    if not callable(renderer):
        raise TypeError("OpenEnv prompt_renderer must be callable.")
    rendered = await maybe_call_with_named_args(
        cast(OpenEnvPromptRenderer, renderer),
        observation=observation,
        context=context,
        action_schema=dict(session.action_schema),
        contract=config.contract,
        seed=config.seed,
    )
    messages = (
        [vf.UserMessage(content=rendered)]
        if isinstance(rendered, str)
        else _MESSAGES_ADAPTER.validate_python(rendered)
    )
    return [
        json_data(message.model_dump(mode="json", exclude_none=True))
        for message in messages
    ]


def latest_assistant_json(raw_completion: list[dict]) -> vf.JsonData:
    messages = [message for message in raw_completion if isinstance(message, dict)]
    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    if not assistant_messages:
        return {}
    content = assistant_messages[-1].get("content")
    text = content if isinstance(content, str) else json.dumps(content)
    action = json.loads(text.strip())
    return json_data(action, context="OpenEnv assistant action")


def mcp_tool_content(observation: object) -> vf.JsonValue:
    model_dump = getattr(observation, "model_dump", None)
    if callable(model_dump):
        observation = model_dump()
    if not isinstance(observation, dict):
        return json_value(observation)
    observation_data = json_data(observation)
    if observation_data.get("error") is not None:
        return {"error": json_value(observation_data.get("error"))}
    value = observation_data.get("result")
    data = getattr(value, "data", None)
    if data is not None:
        return json_value(data)
    if isinstance(value, dict) and "data" in value:
        return json_value(value["data"])
    return json_value(value)


def openenv_tool_defs(tools: Iterable[object]) -> list[vf.JsonData]:
    tool_defs: list[vf.JsonData] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError("OpenEnv MCP tools must have non-empty names.")
        description = getattr(tool, "description", "") or ""
        schema = getattr(tool, "input_schema", None)
        if schema is None:
            schema = getattr(tool, "inputSchema", None)
        parameters = (
            json_data(schema, context=f"OpenEnv tool {name} schema")
            if isinstance(schema, dict)
            else {"type": "object", "properties": {}}
        )
        tool_defs.append(
            json_data(
                {
                    "name": name,
                    "description": str(description),
                    "parameters": parameters,
                },
                context=f"OpenEnv tool {name}",
            )
        )
    return tool_defs


def result_payload(
    *,
    messages: list[vf.JsonData],
    result: OpenEnvResult,
    include_reward: bool,
) -> vf.JsonData:
    payload: dict[str, object] = {
        "messages": messages,
        "observation": json_value(result.observation),
        "openenv_done": bool(result.done),
    }
    if include_reward:
        payload["reward"] = float(result.reward or 0.0)
    if result.done:
        payload["stop_condition"] = "openenv_done"
    return json_data(payload)


class OpenEnvUser(vf.User[OpenEnvUserConfig]):
    session: OpenEnvSession | None

    def start(self) -> None:
        self.session = None

    def stop(self) -> None:
        if self.session is not None:
            asyncio.run(self.session.close())
        self.session = None

    @vf.tool(
        hidden=True,
        args={"openenv": "task.openenv"},
        sets={
            "observation": "state.extras.openenv.observation",
            "openenv_done": "state.extras.openenv.done",
            "finished": "state.is_completed",
            "stop_condition": "state.stop_condition",
        },
    )
    async def setup(self, openenv: vf.JsonData) -> vf.JsonData:
        config = OpenEnvRuntimeConfig.model_validate(openenv)
        self.session = OpenEnvSession(config)
        if config.contract == "mcp":
            await self.session.start()
            tools = list(config.tools) or await self.session.tool_defs()
            return json_data(
                {
                    "observation": {},
                    "openenv_done": False,
                    "tools": tools,
                }
            )
        result = await self.session.reset()
        payload: dict[str, object] = {
            "observation": json_value(result.observation),
            "openenv_done": bool(result.done),
        }
        if result.done:
            payload["finished"] = True
            payload["stop_condition"] = "openenv_done"
        return json_data(payload)

    @vf.user(
        args={
            "openenv": "task.openenv",
            "observation": "state.extras.openenv.observation",
            "completion": "state.completion",
        },
        sets={
            "observation": "state.extras.openenv.observation",
            "openenv_done": "state.extras.openenv.done",
            "reward": "state.transcript.last.reward",
            "stop_condition": "state.stop_condition",
        },
    )
    async def respond(
        self,
        openenv: vf.JsonData,
        observation: vf.JsonValue,
        completion: list[vf.JsonData],
    ) -> vf.JsonData:
        config = OpenEnvRuntimeConfig.model_validate(openenv)
        if self.session is None:
            raise RuntimeError("OpenEnv setup has not started.")
        if not completion:
            return json_data(
                {
                    "messages": await render_messages(
                        config, self.session, observation, "reset"
                    )
                }
            )
        if config.contract == "mcp":
            return json_data(
                {"messages": [], "stop_condition": "openenv_no_tool_calls"}
            )
        action = latest_assistant_json(completion)
        result = await self.session.step(action)
        return result_payload(
            messages=await render_messages(
                config, self.session, result.observation, "step"
            ),
            result=result,
            include_reward=True,
        )

    @vf.tool(
        hidden=True,
        sets={
            "observation": "state.extras.openenv.observation",
            "openenv_done": "state.extras.openenv.done",
            "reward": "state.transcript.last.reward",
            "finished": "state.is_completed",
            "stop_condition": "state.stop_condition",
        },
    )
    async def call_tool(self, name: str, input: vf.JsonData) -> vf.JsonData:
        """Call an OpenEnv MCP tool by name with JSON input."""
        if self.session is None:
            raise RuntimeError("OpenEnv session has not started.")
        result = await self.session.call_tool(name, input)
        content = mcp_tool_content(result.observation)
        payload: dict[str, object] = {
            "content": content
            if isinstance(content, str)
            else json.dumps(json_value(content)),
            "observation": json_value(result.observation),
            "openenv_done": bool(result.done),
        }
        if result.reward is not None:
            payload["reward"] = float(result.reward)
        if result.done:
            payload["finished"] = True
            payload["stop_condition"] = "openenv_done"
        return json_data(payload)


def load_taskset(config: OpenEnvTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)

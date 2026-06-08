import asyncio
import contextlib
import importlib.util
import json
import weakref
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


class OpenEnvTool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str | None: ...


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
    scope: vf.Scope = "env"


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
    max_retries: int = 12
    base_delay: float = 0.5
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 60.0
    jitter: float = 1e-3


class OpenEnvTask(vf.Task, frozen=True):
    openenv: vf.JsonData
    info: vf.JsonData


class OpenEnvSession:
    _startup_slots: weakref.WeakKeyDictionary[
        asyncio.AbstractEventLoop, asyncio.Semaphore
    ] = weakref.WeakKeyDictionary()

    def __init__(self, config: OpenEnvRuntimeConfig, server: "OpenEnvServer"):
        self.config = config
        self.server = server
        self.client: GenericEnvClient | MCPToolClient | None = None

    @property
    def action_schema(self) -> vf.JsonData:
        return self.server.action_schema

    @classmethod
    @contextlib.asynccontextmanager
    async def startup_slot(cls):
        loop = asyncio.get_running_loop()
        semaphore = cls._startup_slots.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(1)
            cls._startup_slots[loop] = semaphore
        async with semaphore:
            yield

    async def start(self) -> GenericEnvClient | MCPToolClient:
        if self.client is not None:
            return self.client
        self.client = await self.server.open_client()
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
        return await self.server.tool_defs()

    async def close(self) -> None:
        if self.client is not None:
            await maybe_await(self.client.close)
        self.client = None


class OpenEnvServer:
    def __init__(self, config: OpenEnvRuntimeConfig):
        self.config = config
        self.provider: PrimeSandboxOpenEnvProvider | None = None
        self.base_url: str | None = None
        self.action_schema: vf.JsonData = {}
        self._lock = asyncio.Lock()
        self._tools_lock = asyncio.Lock()
        self._tool_defs: list[vf.JsonData] | None = None

    @property
    def client_class(self) -> type[GenericEnvClient] | type[MCPToolClient]:
        return MCPToolClient if self.config.contract == "mcp" else GenericEnvClient

    async def start(self) -> None:
        if self.base_url is not None:
            return
        async with self._lock:
            if self.base_url is not None:
                return
            config = self.config
            provider = PrimeSandboxOpenEnvProvider(config)
            schema: vf.JsonData | None = None
            try:
                async with OpenEnvSession.startup_slot():
                    base_url = await asyncio.to_thread(
                        provider.start_container,
                        config.image,
                        port=config.port,
                        start_command=config.start_command,
                        env_vars={"ENABLE_WEB_INTERFACE": "false"},
                    )
                    await asyncio.to_thread(provider.wait_for_ready, base_url)
                    schema = await asyncio.to_thread(provider.fetch_schema)
                action_schema = schema.get("action", {})
                if isinstance(action_schema, dict):
                    self.action_schema = json_data(action_schema)
                self.provider = provider
                self.base_url = base_url
            except BaseException:
                provider.stop_container()
                raise

    async def open_client(self) -> GenericEnvClient | MCPToolClient:
        await self.start()
        if self.base_url is None:
            raise RuntimeError("OpenEnv server did not start.")
        client = self.client_class(base_url=self.base_url)
        await client.connect()
        return client

    async def tool_defs(self) -> list[vf.JsonData]:
        if self.config.contract != "mcp":
            return []
        if self._tool_defs is not None:
            return list(self._tool_defs)
        async with self._tools_lock:
            if self._tool_defs is not None:
                return list(self._tool_defs)
            client = await self.open_client()
            try:
                if not isinstance(client, MCPToolClient):
                    self._tool_defs = []
                else:
                    self._tool_defs = openenv_tool_defs(
                        await client.list_tools(use_cache=False)
                    )
            finally:
                await maybe_await(client.close)
            return list(self._tool_defs)

    async def close(self) -> None:
        if self.provider is not None:
            self.provider.stop_container()
        self.provider = None
        self.base_url = None
        self.action_schema = {}
        self._tool_defs = None


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


def openenv_tool_defs(tools: Iterable[OpenEnvTool]) -> list[vf.JsonData]:
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
    done = openenv_result_done(result)
    payload: dict[str, object] = {
        "messages": messages,
        "observation": json_value(result.observation),
        "openenv_done": done,
    }
    if include_reward:
        payload["reward"] = openenv_result_reward(result)
    if done:
        payload["stop_condition"] = "openenv_done"
    return json_data(payload)


def openenv_result_done(result: OpenEnvResult) -> bool:
    observation_done = getattr(result.observation, "done", None)
    return bool(result.done or observation_done)


def openenv_result_reward(result: OpenEnvResult) -> float:
    reward = result.reward
    if reward is None:
        reward = getattr(result.observation, "reward", None)
    return float(reward or 0.0)


class OpenEnvUser(vf.User[OpenEnvUserConfig]):
    sessions: dict[str, OpenEnvSession]
    servers: dict[str, OpenEnvServer]

    def start(self) -> None:
        self.sessions = {}
        self.servers = {}

    def stop(self) -> None:
        if self.sessions or self.servers:
            asyncio.run(self.close_resources())
        self.sessions = {}
        self.servers = {}

    async def close_resources(self) -> None:
        sessions = list(self.sessions.values())
        servers = list(self.servers.values())
        self.sessions.clear()
        self.servers.clear()
        await asyncio.gather(*(session.close() for session in sessions))
        await asyncio.gather(*(server.close() for server in servers))

    def session_for(self, state_id: str) -> OpenEnvSession:
        try:
            return self.sessions[state_id]
        except KeyError as exc:
            raise RuntimeError("OpenEnv setup has not started.") from exc

    def server_for(self, config: OpenEnvRuntimeConfig) -> OpenEnvServer:
        key = self.server_key(config)
        server = self.servers.get(key)
        if server is None:
            server = OpenEnvServer(config)
            self.servers[key] = server
        return server

    @staticmethod
    def server_key(config: OpenEnvRuntimeConfig) -> str:
        data = config.model_dump(mode="json")
        data.pop("seed", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @vf.tool(
        hidden=True,
        args={"state_id": "state.id", "openenv": "task.openenv"},
        sets={
            "observation": "state.extras.openenv.observation",
            "openenv_done": "state.extras.openenv.done",
            "finished": "state.is_completed",
            "stop_condition": "state.stop_condition",
        },
    )
    async def setup(self, state_id: str, openenv: vf.JsonData) -> vf.JsonData:
        config = OpenEnvRuntimeConfig.model_validate(openenv)
        existing = self.sessions.get(state_id)
        if existing is not None:
            await existing.close()
        session = OpenEnvSession(config, self.server_for(config))
        self.sessions[state_id] = session
        if config.contract == "mcp":
            tools = list(config.tools) or await session.tool_defs()
            return json_data(
                {
                    "observation": {},
                    "openenv_done": False,
                    "tools": tools,
                }
            )
        result = await session.reset()
        payload: dict[str, object] = {
            "observation": json_value(result.observation),
            "openenv_done": openenv_result_done(result),
        }
        if openenv_result_done(result):
            payload["finished"] = True
            payload["stop_condition"] = "openenv_done"
        return json_data(payload)

    @vf.user(
        args={
            "state_id": "state.id",
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
        state_id: str,
        observation: vf.JsonValue,
        completion: list[vf.JsonData],
    ) -> vf.JsonData:
        session = self.session_for(state_id)
        config = session.config
        if not completion:
            return json_data(
                {
                    "messages": await render_messages(
                        config, session, observation, "reset"
                    )
                }
            )
        if config.contract == "mcp":
            return json_data(
                {"messages": [], "stop_condition": "openenv_no_tool_calls"}
            )
        action = latest_assistant_json(completion)
        result = await session.step(action)
        return result_payload(
            messages=await render_messages(config, session, result.observation, "step"),
            result=result,
            include_reward=True,
        )

    @vf.tool(
        hidden=True,
        args={"state_id": "state.id"},
        sets={
            "observation": "state.extras.openenv.observation",
            "openenv_done": "state.extras.openenv.done",
            "reward": "state.transcript.last.reward",
            "finished": "state.is_completed",
            "stop_condition": "state.stop_condition",
        },
    )
    async def call_tool(
        self, state_id: str, name: str, input: vf.JsonData
    ) -> vf.JsonData:
        """Call an OpenEnv MCP tool by name with JSON input."""
        session = self.session_for(state_id)
        result = await session.call_tool(name, input)
        content = mcp_tool_content(result.observation)
        payload: dict[str, object] = {
            "content": content
            if isinstance(content, str)
            else json.dumps(json_value(content)),
            "observation": json_value(result.observation),
            "openenv_done": openenv_result_done(result),
        }
        payload["reward"] = openenv_result_reward(result)
        if openenv_result_done(result):
            payload["finished"] = True
            payload["stop_condition"] = "openenv_done"
        return json_data(payload)


def load_taskset(config: OpenEnvTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)

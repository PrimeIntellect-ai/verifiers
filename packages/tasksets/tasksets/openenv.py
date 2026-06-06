import asyncio
import importlib.util
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient

import verifiers.v1 as vf
from verifiers.utils.async_utils import maybe_await, maybe_call_with_named_args
from verifiers.utils.message_utils import get_messages, normalize_messages
from verifiers.v1.config import import_config_ref
from verifiers.v1.utils.json_utils import jsonable

from tasksets.utils.openenv_utils import PrimeSandboxOpenEnvProvider

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
        observation_map = cast(vf.JsonData, observation)
        messages = observation_map.get("messages")
        if messages is not None:
            if not isinstance(messages, list):
                raise TypeError("OpenEnv observation messages must be a list.")
            return cast(vf.PromptInput, messages)
        for key in ("prompt", "question", "instruction", "content", "text"):
            value = observation_map.get(key)
            if isinstance(value, str) and value.strip():
                return [{"role": "user", "content": value}]
        return [{"role": "user", "content": json.dumps(jsonable(observation))}]
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


class OpenEnvTasksetConfig(vf.TasksetConfig):
    id: str | None = "openenv"
    user: vf.User | None = vf.User(
        server=vf.MCPServerSpec(
            command=[sys.executable, "-m", "tasksets.openenv", "--user-server"]
        )
    )
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
            schema = await asyncio.to_thread(provider.fetch_schema)
        except Exception:
            provider.stop_container()
            raise
        action_schema = schema.get("action", {})
        if isinstance(action_schema, dict):
            self.action_schema = cast(vf.JsonData, dict(action_schema))
        return self.client

    async def reset(self) -> OpenEnvResult:
        client = await self.start()
        return cast(OpenEnvResult, await client.reset(seed=self.config.seed))

    async def step(self, action: vf.JsonData) -> OpenEnvResult:
        client = await self.start()
        if not isinstance(client, GenericEnvClient):
            raise RuntimeError("MCP OpenEnv tasks require an MCP tool server.")
        return cast(OpenEnvResult, await client.step(action))

    async def close(self) -> None:
        if self.client is not None:
            await maybe_await(self.client.close)
        if self.provider is not None:
            self.provider.stop_container()
        self.client = None
        self.provider = None


SESSION: OpenEnvSession | None = None


class OpenEnvTaskset(vf.Taskset[OpenEnvTasksetConfig]):
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
        return float(state.reward)


async def render_messages(
    config: OpenEnvRuntimeConfig,
    session: OpenEnvSession,
    observation: object,
    context: str,
) -> list[dict[str, object]]:
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
    return [
        cast(dict[str, object], message.model_dump(mode="json", exclude_none=True))
        for message in normalize_messages(
            cast(vf.PromptInput, rendered), field_name="openenv"
        )
    ]


def latest_assistant_json(state: dict) -> vf.JsonData:
    raw_completion = state["completion"]
    if not isinstance(raw_completion, list):
        raise TypeError("OpenEnv state.completion must be a list.")
    messages = [
        cast(dict[str, object], message)
        for message in raw_completion
        if isinstance(message, dict)
    ]
    assistant_messages = get_messages(messages, role="assistant")
    if not assistant_messages:
        return {}
    content = assistant_messages[-1].content
    text = content if isinstance(content, str) else json.dumps(content)
    action = json.loads(text.strip())
    if not isinstance(action, dict):
        raise TypeError("OpenEnv assistant action must decode to an object.")
    return cast(vf.JsonData, action)


def result_payload(
    *,
    messages: list[dict[str, object]],
    result: OpenEnvResult,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "messages": messages,
        "scratch": {"openenv_done": bool(result.done)},
        "reward_delta": float(result.reward or 0.0),
    }
    if result.done:
        payload["stop_condition"] = "openenv_done"
    return payload


def run_user_server() -> None:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("openenv-user")

    @mcp.tool()
    async def respond(task: dict, state: dict, transcript: list[dict]) -> dict:
        _ = transcript
        global SESSION
        config = OpenEnvRuntimeConfig.model_validate(task["openenv"])
        if SESSION is None:
            SESSION = OpenEnvSession(config)
            result = await SESSION.reset()
            return result_payload(
                messages=await render_messages(
                    config, SESSION, result.observation, "reset"
                ),
                result=result,
            )
        action = latest_assistant_json(state)
        result = await SESSION.step(action)
        return result_payload(
            messages=await render_messages(config, SESSION, result.observation, "step"),
            result=result,
        )

    mcp.run(transport="stdio")


def load_taskset(config: OpenEnvTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)


if __name__ == "__main__" and sys.argv[1:] == ["--user-server"]:
    run_user_server()

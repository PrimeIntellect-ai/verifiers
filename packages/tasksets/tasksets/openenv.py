import asyncio
import inspect
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import requests
import tenacity as tc
from openenv.core.containers.runtime.providers import ContainerProvider
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    Tool as OpenEnvToolSpec,
)
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient
from pydantic import field_validator
from verifiers.decorators import cleanup, reward, setup, stop
from verifiers.types import Message, MessageContent, Tool, UserMessage
from verifiers.utils.message_utils import get_messages, normalize_messages
from verifiers.utils.tool_utils import is_valid_tool_content_parts
from verifiers.v1.config import ConfigSource, import_config_ref
from verifiers.v1.state import State
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.toolset import Toolset
from verifiers.v1.types import ConfigData, ConfigMap, Handler, PromptInput
from verifiers.v1.user import User
from verifiers.v1.utils.config_utils import coerce_config
from verifiers.v1.utils.serialization_utils import serializable


@dataclass(frozen=True)
class OpenEnvRuntimeSpec:
    openenv_project: str
    prompt_renderer: str
    image: str
    port: int
    start_command: str
    contract: str
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


class OpenEnvTasksetConfig(TasksetConfig):
    prompt_renderer: str = "tasksets.openenv:default_openenv_prompt_renderer"
    openenv_project: str | None = None
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

    @field_validator("prompt_renderer")
    @classmethod
    def validate_prompt_renderer(cls, value: str) -> str:
        if not value:
            raise ValueError("OpenEnvTasksetConfig.prompt_renderer is required.")
        return value


class OpenEnvTaskset(Taskset[OpenEnvTasksetConfig]):
    def __init__(self, config: ConfigSource = None):
        config_value = coerce_config(OpenEnvTasksetConfig, config)
        if config_value.openenv_project is None:
            config_value = config_value.model_copy(
                update={"openenv_project": discover_caller_openenv_project()}
            )
        super().__init__(config=config_value)
        if "user" not in self.config.model_fields_set:
            self.user = User(fn=self.openenv_user)
        self.taskset_id = self.config.taskset_id or "openenv"

    def load_tasks(self) -> list[ConfigData]:
        return load_tasks(self.config)

    def load_eval_tasks(self) -> list[ConfigData]:
        return load_eval_tasks(self.config)

    @setup
    async def setup_openenv(self, task: Task, state: State) -> None:
        spec = openenv_runtime_spec(task)
        client = await ensure_openenv_runtime(spec, state)
        result = await client.reset(seed=spec.seed)
        state["openenv_done"] = bool(result.done)
        state["prompt"] = await render_openenv_observation_messages(
            result.observation,
            spec,
            context="reset",
            action_schema=openenv_action_schema(state),
        )

    @cleanup
    async def cleanup_openenv(self, state: State) -> None:
        client = state.pop("openenv_client", None)
        state.pop("openenv_action_schema", None)
        state.pop("openenv_contract", None)
        if client is not None:
            await close_openenv_client(client)

    @stop
    async def openenv_done(self, state: State) -> bool:
        return bool(state.get("openenv_done"))

    @reward(weight=1.0)
    async def openenv_reward(self, state: State) -> float:
        return float(
            sum(
                float(step.get("reward", 0.0) or 0.0)
                for step in state.get("trajectory", [])
                if isinstance(step, Mapping)
            )
        )

    async def openenv_user(
        self, task: Task, state: State, messages: Sequence[Message]
    ) -> list[UserMessage]:
        spec = openenv_runtime_spec(task)
        if spec.contract == "mcp":
            return []
        assistant_messages = get_messages(messages, role="assistant")
        last_message = assistant_messages[-1] if assistant_messages else None
        text = str(last_message.content or "").strip() if last_message else ""
        schema = openenv_action_schema(state)
        action = parse_openenv_action(text, schema)
        client = openenv_generic_client_from_state(state)
        result = await client.step(action)
        record_openenv_step_reward(state, result.reward)
        state["openenv_done"] = bool(result.done)
        if result.done:
            state.stop("openenv_done")
        return cast(
            list[UserMessage],
            await render_openenv_observation_messages(
                result.observation,
                spec,
                context="step",
                action_schema=schema,
            ),
        )


def load_taskset(config: OpenEnvTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)


def load_tasks(config: OpenEnvTasksetConfig) -> list[ConfigData]:
    return openenv_rows(config, config.num_train_examples, config.seed)


def load_eval_tasks(config: OpenEnvTasksetConfig) -> list[ConfigData]:
    if config.num_eval_examples <= 0:
        return []
    return openenv_rows(
        config,
        config.num_eval_examples,
        config.seed + config.num_train_examples,
    )


async def openenv_toolset(task: Task, state: State) -> Toolset:
    spec = openenv_runtime_spec(task)
    if spec.contract != "mcp":
        return Toolset()
    client = await ensure_openenv_runtime(spec, state)
    if not isinstance(client, MCPToolClient):
        raise RuntimeError("OpenEnv MCP task did not initialize an MCPToolClient.")
    tool_defs = convert_openenv_mcp_tools(await client.list_tools())
    return Toolset(tools=[OpenEnvMCPTool(tool_def) for tool_def in tool_defs])


class OpenEnvMCPTool:
    def __init__(self, tool_def: Tool):
        self.tool_def = tool_def
        self.name = tool_def.name
        self.__name__ = tool_def.name
        self.__doc__ = tool_def.description

    async def __call__(self, state: State, **kwargs: object) -> MessageContent:
        client = openenv_mcp_client_from_state(state)
        result = await client.step(
            CallToolAction(tool_name=self.name, arguments=dict(kwargs))
        )
        record_openenv_step_reward(state, result.reward)
        state["openenv_done"] = bool(result.done)
        if result.done:
            state.stop("openenv_done")
        content = extract_openenv_mcp_tool_content(result.observation)
        if is_valid_tool_content_parts(content):
            return cast(MessageContent, content)
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True)


def discover_caller_openenv_project() -> str:
    current_file = Path(__file__).resolve()
    for frame_info in inspect.stack()[2:]:
        frame_path = Path(frame_info.filename).resolve()
        if frame_path != current_file:
            return str(frame_path.parent / "proj")
    return str(Path.cwd() / "proj")


def openenv_rows(
    config: OpenEnvTasksetConfig, num_examples: int, first_seed: int
) -> list[ConfigData]:
    if num_examples <= 0:
        return []
    project = resolved_openenv_project(config)
    image, port, start_command, contract = resolve_openenv_runtime_config(project)
    rows: list[ConfigData] = []
    for index in range(num_examples):
        seed = first_seed + index
        row: ConfigData = {
            "example_id": index,
            "prompt": [
                {
                    "role": "user",
                    "content": "OpenEnv rollout is initializing.",
                }
            ],
            "openenv": {
                "openenv_project": str(project),
                "prompt_renderer": config.prompt_renderer,
                "image": image,
                "port": port,
                "start_command": start_command,
                "contract": contract,
                "seed": seed,
                "startup_timeout_seconds": config.startup_timeout_seconds,
                "startup_poll_interval_seconds": config.startup_poll_interval_seconds,
                "health_request_timeout_seconds": config.health_request_timeout_seconds,
                "schema_request_timeout_seconds": config.schema_request_timeout_seconds,
                "wait_for_creation_max_attempts": config.wait_for_creation_max_attempts,
                "max_retries": config.max_retries,
                "base_delay": config.base_delay,
                "backoff_factor": config.backoff_factor,
                "max_backoff_seconds": config.max_backoff_seconds,
                "jitter": config.jitter,
            },
            "info": {"seed": seed, "contract": contract},
        }
        if contract == "mcp":
            row["toolsets"] = {"openenv": {"fn": "tasksets.openenv:openenv_toolset"}}
        rows.append(row)
    return rows


def resolved_openenv_project(config: OpenEnvTasksetConfig) -> Path:
    project_value = config.openenv_project
    if project_value is None:
        project_value = discover_caller_openenv_project()
    project = Path(project_value).expanduser().resolve()
    if not project.exists() or not project.is_dir():
        raise ValueError(f"OpenEnv project directory not found: {project}")
    return project


def resolve_openenv_runtime_config(project_path: Path) -> tuple[str, int, str, str]:
    manifest = read_openenv_build_manifest(project_path)
    image = manifest.get("image")
    port = manifest.get("port", 8000)
    start_command = manifest.get("start_command")
    contract = manifest.get("contract")
    if not isinstance(image, str) or not image.strip():
        raise RuntimeError(
            "Invalid .build.json: image must be a non-empty string. "
            "Run vf-build for this OpenEnv project."
        )
    if isinstance(port, bool) or not isinstance(port, int | float | str):
        raise RuntimeError("Invalid .build.json: port must be an integer.")
    port_num = int(port)
    if not isinstance(start_command, str) or not start_command.strip():
        raise RuntimeError(
            "Invalid .build.json: start_command must be a non-empty string. "
            "Run vf-build for this OpenEnv project."
        )
    if not isinstance(contract, str) or contract not in {"gym", "mcp"}:
        raise RuntimeError("Invalid .build.json: contract must be gym or mcp.")
    return image.strip(), port_num, start_command.strip(), contract


def read_openenv_build_manifest(project_path: Path) -> ConfigData:
    manifest_path = project_path / ".build.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"OpenEnv project {project_path} is missing .build.json. "
            "Run vf-build for this OpenEnv project."
        )
    data = json.loads(manifest_path.read_text())
    if not isinstance(data, Mapping):
        raise RuntimeError(f"Invalid OpenEnv build manifest at {manifest_path}.")
    return {str(key): value for key, value in data.items()}


def openenv_runtime_spec(task: Task) -> OpenEnvRuntimeSpec:
    data = task.get("openenv")
    if not isinstance(data, Mapping):
        raise TypeError("OpenEnv tasks must contain an openenv mapping.")
    return OpenEnvRuntimeSpec(
        openenv_project=str(data["openenv_project"]),
        prompt_renderer=str(data["prompt_renderer"]),
        image=str(data["image"]),
        port=int(cast(int | str, data["port"])),
        start_command=str(data["start_command"]),
        contract=str(data["contract"]),
        seed=int(cast(int | str, data["seed"])),
        startup_timeout_seconds=int(cast(int | str, data["startup_timeout_seconds"])),
        startup_poll_interval_seconds=float(
            cast(float | int | str, data["startup_poll_interval_seconds"])
        ),
        health_request_timeout_seconds=float(
            cast(float | int | str, data["health_request_timeout_seconds"])
        ),
        schema_request_timeout_seconds=float(
            cast(float | int | str, data["schema_request_timeout_seconds"])
        ),
        wait_for_creation_max_attempts=int(
            cast(int | str, data["wait_for_creation_max_attempts"])
        ),
        max_retries=int(cast(int | str, data["max_retries"])),
        base_delay=float(cast(float | int | str, data["base_delay"])),
        backoff_factor=float(cast(float | int | str, data["backoff_factor"])),
        max_backoff_seconds=float(cast(float | int | str, data["max_backoff_seconds"])),
        jitter=float(cast(float | int | str, data["jitter"])),
    )


async def ensure_openenv_runtime(
    spec: OpenEnvRuntimeSpec, state: State
) -> GenericEnvClient | MCPToolClient:
    existing = state.get("openenv_client")
    if isinstance(existing, GenericEnvClient | MCPToolClient):
        return existing
    provider = PrimeSandboxOpenEnvProvider(spec)
    client_type = MCPToolClient if spec.contract == "mcp" else GenericEnvClient
    try:
        client = await client_type.from_docker_image(
            spec.image,
            provider=provider,
            port=spec.port,
            start_command=spec.start_command,
            env_vars={"ENABLE_WEB_INTERFACE": "false"},
        )
        schema = await asyncio.to_thread(fetch_openenv_schema, provider.base_url, spec)
        action_schema = schema.get("action", {}) if isinstance(schema, Mapping) else {}
        if not isinstance(action_schema, Mapping):
            action_schema = {}
        assert_openenv_contract_matches_schema(
            spec.contract, cast(ConfigMap, action_schema)
        )
        state["openenv_client"] = client
        state["openenv_contract"] = spec.contract
        state["openenv_action_schema"] = dict(action_schema)
        return client
    except Exception:
        try:
            provider.stop_container()
        except Exception:
            pass
        raise


class PrimeSandboxOpenEnvProvider(ContainerProvider):
    def __init__(self, spec: OpenEnvRuntimeSpec):
        self.spec = spec
        self.sandbox_id: str | None = None
        self.exposure_id: str | None = None
        self._base_url: str | None = None
        self._client: object | None = None

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("OpenEnv sandbox has not started.")
        return self._base_url

    def start_container(
        self,
        image: str,
        port: int | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: object,
    ) -> str:
        from prime_sandboxes import APIClient, CreateSandboxRequest, SandboxClient

        api_client = APIClient()
        client = SandboxClient(api_client)
        self._client = client
        start_command = kwargs.get("start_command")
        if not isinstance(start_command, str) or not start_command:
            start_command = self.spec.start_command
        container_port = port or self.spec.port
        environment_vars = {"ENABLE_WEB_INTERFACE": "false", **dict(env_vars or {})}
        request = CreateSandboxRequest(
            name="openenv-env",
            docker_image=image,
            start_command=start_command,
            cpu_cores=2,
            memory_gb=4,
            disk_size_gb=10,
            timeout_minutes=60,
            environment_vars=environment_vars,
        )
        sandbox_id: str | None = None
        exposure_id: str | None = None
        try:
            sandbox = openenv_retry_sync(self.spec, client.create, request)
            sandbox_id = sandbox_handle_id(sandbox)
            self.sandbox_id = sandbox_id
            client.wait_for_creation(
                sandbox_id,
                max_attempts=self.spec.wait_for_creation_max_attempts,
            )
            exposure = client.expose(
                sandbox_id,
                port=container_port,
                name="openenv-env",
                protocol="TCP",
            )
            exposure_id = exposure_handle_id(exposure)
            self.exposure_id = exposure_id
            self._base_url = openenv_exposure_base_url(exposure)
            return self._base_url
        except Exception as exc:
            details = (
                openenv_sandbox_failure_details(client, sandbox_id, container_port)
                if sandbox_id is not None
                else None
            )
            if sandbox_id is not None:
                if exposure_id is not None:
                    try:
                        client.unexpose(sandbox_id, exposure_id)
                    except Exception:
                        pass
                try:
                    client.delete(sandbox_id)
                except Exception:
                    pass
            message = f"OpenEnv sandbox failed during startup for image {image}."
            if details:
                message = f"{message}\n{details}"
            raise RuntimeError(message) from exc

    def stop_container(self) -> None:
        client = self._client
        if client is None:
            return
        if self.sandbox_id is not None and self.exposure_id is not None:
            unexpose = getattr(client, "unexpose", None)
            try:
                if callable(unexpose):
                    unexpose(self.sandbox_id, self.exposure_id)
            except Exception:
                pass
        if self.sandbox_id is not None:
            delete = getattr(client, "delete", None)
            try:
                if callable(delete):
                    delete(self.sandbox_id)
            except Exception:
                pass
        api_client = getattr(client, "client", None)
        close = getattr(api_client, "close", None)
        if callable(close):
            close()
        self.sandbox_id = None
        self.exposure_id = None
        self._base_url = None
        self._client = None

    def wait_for_ready(self, base_url: str, timeout_s: float = 30.0) -> None:
        del timeout_s
        wait_for_openenv_ready(base_url, self.spec)


def openenv_retry_sync(
    spec: OpenEnvRuntimeSpec, fn: Callable[..., object], *args: object
) -> object:
    retrying = tc.Retrying(
        stop=tc.stop_after_attempt(spec.max_retries),
        wait=tc.wait_exponential_jitter(
            initial=spec.base_delay,
            exp_base=spec.backoff_factor,
            max=spec.max_backoff_seconds,
            jitter=spec.jitter,
        ),
        reraise=True,
    )
    return retrying(fn, *args)


def sandbox_handle_id(sandbox: object) -> str:
    sandbox_id = getattr(sandbox, "id", None)
    if not isinstance(sandbox_id, str) or not sandbox_id:
        raise RuntimeError("Prime sandbox response did not include an id.")
    return sandbox_id


def exposure_handle_id(exposure: object) -> str:
    exposure_id = getattr(exposure, "exposure_id", None)
    if not isinstance(exposure_id, str) or not exposure_id:
        raise RuntimeError("Prime sandbox exposure response did not include an id.")
    return exposure_id


def openenv_exposure_base_url(exposure: object) -> str:
    endpoint = getattr(exposure, "external_endpoint", None)
    if isinstance(endpoint, str) and endpoint.strip():
        return f"http://{endpoint.strip()}"
    raw_url = str(getattr(exposure, "url", "") or "").strip()
    if raw_url.startswith("tcp://"):
        host_port = raw_url[len("tcp://") :].rstrip("/")
        if host_port:
            return f"http://{host_port}"
    if raw_url.startswith(("http://", "https://")):
        return raw_url.rstrip("/")
    raise RuntimeError("OpenEnv sandbox exposure did not provide a usable URL.")


def wait_for_openenv_ready(base_url: str, spec: OpenEnvRuntimeSpec) -> None:
    start_time = time.monotonic()
    last_error = "no attempts"
    while time.monotonic() - start_time < spec.startup_timeout_seconds:
        ok, detail = check_openenv_health(base_url, spec)
        if ok:
            return
        last_error = detail
        time.sleep(spec.startup_poll_interval_seconds)
    raise RuntimeError(
        "OpenEnv server not ready. "
        f"Health timeout={spec.startup_timeout_seconds}s, "
        f"url={base_url}, last error: {last_error}"
    )


def check_openenv_health(base_url: str, spec: OpenEnvRuntimeSpec) -> tuple[bool, str]:
    try:
        response = requests.get(
            f"{base_url}/health",
            timeout=spec.health_request_timeout_seconds,
        )
        if response.status_code == 200:
            return True, "ok"
        return False, f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def openenv_sandbox_failure_details(
    client: object, sandbox_id: str, port: int | None
) -> str | None:
    details: list[str] = []
    get_logs = getattr(client, "get_logs", None)
    if callable(get_logs):
        try:
            logs = str(get_logs(sandbox_id) or "")
        except Exception:
            logs = ""
        if logs:
            details.append(f"Logs tail:\n{logs[-4000:]}")
    execute_command = getattr(client, "execute_command", None)
    if callable(execute_command) and port is not None:
        try:
            result = execute_command(
                sandbox_id,
                f'sh -lc "curl -sS -m 2 http://localhost:{int(port)}/health 2>&1 || true"',
                timeout=5,
            )
        except Exception as exc:
            details.append(f"Local /health probe failed: {type(exc).__name__}: {exc}")
        else:
            stdout = str(getattr(result, "stdout", "") or "").strip()
            stderr = str(getattr(result, "stderr", "") or "").strip()
            if stdout:
                details.append(f"Local /health probe stdout: {stdout}")
            if stderr:
                details.append(f"Local /health probe stderr: {stderr}")
            if not stdout and not stderr:
                details.append("Local /health probe returned no output.")
    return "\n".join(details) or None


def fetch_openenv_schema(base_url: str, spec: OpenEnvRuntimeSpec) -> ConfigData:
    def request_schema() -> ConfigData:
        response = requests.get(
            f"{base_url}/schema",
            timeout=spec.schema_request_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, Mapping):
            raise TypeError("OpenEnv /schema must return a JSON object.")
        return {str(key): value for key, value in data.items()}

    return cast(ConfigData, openenv_retry_sync(spec, request_schema))


def assert_openenv_contract_matches_schema(
    contract: str, action_schema: ConfigMap
) -> None:
    looks_mcp = False
    properties = action_schema.get("properties")
    if isinstance(properties, Mapping):
        looks_mcp = (
            "tool_name" in properties and "arguments" in properties
        ) or schema_contains_openenv_values(action_schema, {"list_tools", "call_tool"})
    if contract == "mcp" and not looks_mcp:
        raise RuntimeError(
            "OpenEnv manifest contract is mcp but action schema does not match MCP."
        )
    if contract == "gym" and looks_mcp:
        raise RuntimeError(
            "OpenEnv manifest contract is gym but action schema looks like MCP."
        )


def schema_contains_openenv_values(value: object, values: set[str]) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "enum" and isinstance(item, list):
                if any(option in values for option in item):
                    return True
            if schema_contains_openenv_values(item, values):
                return True
    elif isinstance(value, list):
        return any(schema_contains_openenv_values(item, values) for item in value)
    return False


async def close_openenv_client(client: object) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def openenv_generic_client_from_state(state: State) -> GenericEnvClient:
    client = state.get("openenv_client")
    if not isinstance(client, GenericEnvClient):
        raise RuntimeError("OpenEnv gym client is not initialized.")
    return client


def openenv_mcp_client_from_state(state: State) -> MCPToolClient:
    client = state.get("openenv_client")
    if not isinstance(client, MCPToolClient):
        raise RuntimeError("OpenEnv MCP client is not initialized.")
    return client


def openenv_action_schema(state: State) -> ConfigMap:
    schema = state.get("openenv_action_schema") or {}
    if not isinstance(schema, Mapping):
        raise TypeError("state.openenv_action_schema must be a mapping.")
    return cast(ConfigMap, schema)


async def render_openenv_observation_messages(
    observation: object,
    spec: OpenEnvRuntimeSpec,
    *,
    context: str,
    action_schema: ConfigMap,
) -> list[Message]:
    renderer = import_config_ref(spec.prompt_renderer)
    if not callable(renderer):
        raise TypeError("OpenEnv prompt_renderer must resolve to a callable.")
    kwargs: ConfigData = {
        "context": context,
        "action_schema": dict(action_schema),
        "contract": spec.contract,
        "seed": spec.seed,
    }
    result = await call_openenv_renderer(cast(Handler, renderer), observation, kwargs)
    messages = normalize_messages(cast(PromptInput, result), field_name="openenv")
    if not messages:
        raise RuntimeError("OpenEnv prompt_renderer returned an empty message list.")
    for index, message in enumerate(messages):
        if message.content is None:
            raise RuntimeError(
                "OpenEnv prompt_renderer returned a message with null content "
                f"at index {index}."
            )
    return messages


def default_openenv_prompt_renderer(observation: object, **kwargs: object) -> object:
    del kwargs
    if isinstance(observation, str):
        return [{"role": "user", "content": observation}]
    if isinstance(observation, Mapping):
        observation_map = cast(ConfigMap, observation)
        messages = observation_map.get("messages")
        if messages is not None:
            return messages
        for key in ("prompt", "question", "instruction", "content", "text"):
            value = observation_map.get(key)
            if isinstance(value, str) and value.strip():
                return [{"role": "user", "content": value}]
        return [{"role": "user", "content": json.dumps(serializable(observation))}]
    return [{"role": "user", "content": str(observation)}]


async def call_openenv_renderer(
    renderer: Handler, observation: object, kwargs: ConfigData
) -> object:
    try:
        signature = inspect.signature(renderer)
    except (TypeError, ValueError):
        result = renderer(observation, **kwargs)
    else:
        has_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        accepted_kwargs = (
            kwargs
            if has_var_kwargs
            else {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
        )
        result = renderer(observation, **accepted_kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def parse_openenv_action(text: str, schema: ConfigMap) -> ConfigData:
    cleaned = strip_openenv_json_fence(text)
    try:
        action = json.loads(cleaned)
        if isinstance(action, Mapping):
            return {str(key): value for key, value in action.items()}
    except Exception:
        pass
    field_name = single_openenv_string_field(schema)
    if field_name is not None:
        return {field_name: cleaned}
    raise ValueError(
        "Failed to parse OpenEnv action JSON. Provide a JSON object matching "
        "the action schema."
    )


def strip_openenv_json_fence(text: str) -> str:
    if text.startswith("```") and text.endswith("```"):
        return "\n".join(text.split("\n")[1:-1]).strip()
    return text


def single_openenv_string_field(schema: ConfigMap) -> str | None:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    property_map = cast(ConfigMap, properties)
    required = schema.get("required")
    if isinstance(required, list):
        required_names = [name for name in required if isinstance(name, str)]
        if len(required_names) == 1:
            name = required_names[0]
            spec = property_map.get(name)
            if isinstance(spec, Mapping):
                spec_map = cast(ConfigMap, spec)
                if spec_map.get("type") == "string":
                    return name
    if len(property_map) == 1:
        name, spec = next(iter(property_map.items()))
        if isinstance(name, str) and isinstance(spec, Mapping):
            spec_map = cast(ConfigMap, spec)
            if spec_map.get("type") == "string":
                return name
    return None


def convert_openenv_mcp_tools(tools: Iterable[OpenEnvToolSpec]) -> list[Tool]:
    tool_defs: list[Tool] = []
    for tool in tools:
        schema = tool.input_schema or {"type": "object", "properties": {}}
        tool_defs.append(
            Tool(
                name=tool.name,
                description=tool.description,
                parameters={str(key): value for key, value in schema.items()},
            )
        )
    return tool_defs


def extract_openenv_mcp_tool_content(observation: object) -> object:
    if isinstance(observation, CallToolObservation):
        if observation.error is not None:
            return {"error": observation.error.message}
        result = observation.result
    elif isinstance(observation, Mapping):
        observation_map = cast(ConfigMap, observation)
        if observation_map.get("error") is not None:
            return {"error": observation_map.get("error")}
        result = observation_map.get("result")
    else:
        return observation
    data = getattr(result, "data", None)
    if data is not None:
        return data
    if isinstance(result, Mapping):
        result_map = cast(ConfigMap, result)
        if "data" in result_map:
            return result_map["data"]
    return result


def record_openenv_step_reward(state: State, reward_value: float | int | None) -> None:
    if reward_value is None:
        return
    trajectory = state.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return
    step = trajectory[-1]
    if not isinstance(step, dict):
        return
    current = float(step.get("reward", 0.0) or 0.0)
    step["reward"] = current + float(reward_value)

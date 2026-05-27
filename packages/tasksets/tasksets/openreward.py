import asyncio
import json
from collections.abc import Iterable, Mapping
from importlib import import_module
from typing import Protocol, cast

from pydantic import field_validator
from verifiers.decorators import cleanup, reward, setup, stop
from verifiers.types import Message, MessageContent, Tool
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.tool_utils import is_valid_tool_content_parts
from verifiers.v1.config import ConfigSource
from verifiers.v1.state import State
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.toolset import Toolset
from verifiers.v1.types import ConfigData, ConfigMap
from verifiers.v1.utils.config_utils import coerce_config
from verifiers.v1.utils.serialization_utils import serializable


class OpenRewardSession(Protocol):
    def get_prompt(self) -> object: ...

    def call_tool(self, tool_name: str, input: ConfigMap) -> object: ...


class OpenRewardSessionContext(Protocol):
    def __enter__(self) -> OpenRewardSession: ...

    def __exit__(self, *exc: object) -> object: ...


class OpenRewardEnvironment(Protocol):
    def list_tasks(self, split: str) -> list[object]: ...

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[object]: ...

    def list_tools(self, format: str | None = None) -> list[object]: ...

    def session(self, task: object) -> OpenRewardSessionContext: ...


class OpenRewardEnvironmentsAPI(Protocol):
    def get(
        self,
        name: str,
        variant: str | None = None,
        base_url: str | None = None,
    ) -> OpenRewardEnvironment: ...


class OpenRewardClient(Protocol):
    environments: OpenRewardEnvironmentsAPI

    def __enter__(self) -> "OpenRewardClient": ...

    def __exit__(self, *exc: object) -> object: ...

    def close(self) -> object: ...


class OpenRewardTasksetConfig(TasksetConfig):
    environment: str
    variant: str | None = None
    base_url: str | None = None
    split: str = "train"
    eval_split: str | None = None
    num_train_examples: int | None = None
    num_eval_examples: int = 0

    @field_validator("environment", "split")
    @classmethod
    def validate_required_name(cls, value: str) -> str:
        if not value:
            raise ValueError("OpenReward environment and split must be non-empty.")
        return value


class OpenRewardTaskset(Taskset[OpenRewardTasksetConfig]):
    def __init__(self, config: ConfigSource = None):
        config_value = coerce_config(OpenRewardTasksetConfig, config)
        super().__init__(config=config_value)
        self.taskset_id = self.config.taskset_id or "openreward"

    def load_tasks(self) -> list[ConfigData]:
        return load_tasks(self.config)

    def load_eval_tasks(self) -> list[ConfigData]:
        return load_eval_tasks(self.config)

    @setup
    async def setup_openreward(self, task: Task, state: State) -> None:
        session = await asyncio.to_thread(openreward_session, task)
        entered_session = await asyncio.to_thread(session.__enter__)
        state["openreward_session"] = entered_session
        state["openreward_session_context"] = session
        prompt = await asyncio.to_thread(
            cast(OpenRewardSession, entered_session).get_prompt
        )
        state["prompt"] = openreward_prompt_messages(prompt)

    @cleanup
    async def cleanup_openreward(self, state: State) -> None:
        session_context = state.pop("openreward_session_context", None)
        state.pop("openreward_session", None)
        if session_context is None:
            return
        exit_fn = getattr(session_context, "__exit__", None)
        if callable(exit_fn):
            await asyncio.to_thread(exit_fn, None, None, None)

    @stop
    async def openreward_done(self, state: State) -> bool:
        return bool(state.get("openreward_finished"))

    @reward(weight=1.0)
    async def openreward_reward(self, state: State) -> float:
        return float(
            sum(
                float(step.get("reward", 0.0) or 0.0)
                for step in state.get("trajectory", [])
                if isinstance(step, Mapping)
            )
        )


class OpenRewardTool:
    def __init__(self, tool_def: Tool):
        self.tool_def = tool_def
        self.name = tool_def.name
        self.__name__ = tool_def.name
        self.__doc__ = tool_def.description

    async def __call__(self, state: State, **kwargs: object) -> MessageContent:
        session = state.get("openreward_session")
        if session is None:
            raise RuntimeError("OpenReward session is not initialized.")
        result = await asyncio.to_thread(
            cast(OpenRewardSession, session).call_tool,
            self.name,
            kwargs,
        )
        reward_value = getattr(result, "reward", None)
        if isinstance(reward_value, int | float):
            record_openreward_step_reward(state, reward_value)
        finished = bool(getattr(result, "finished", False))
        state["openreward_finished"] = finished
        if finished:
            state.stop("openreward_done")
        metadata = getattr(result, "metadata", None)
        if metadata is not None:
            state["openreward_metadata"] = serializable(metadata)
        content = openreward_blocks_content(getattr(result, "blocks", []))
        if is_valid_tool_content_parts(content):
            return cast(MessageContent, content)
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True)


def load_taskset(config: OpenRewardTasksetConfig) -> OpenRewardTaskset:
    return OpenRewardTaskset(config=config)


def load_tasks(config: OpenRewardTasksetConfig) -> list[ConfigData]:
    return openreward_rows(
        config=config,
        split=config.split,
        num_examples=config.num_train_examples,
    )


def load_eval_tasks(config: OpenRewardTasksetConfig) -> list[ConfigData]:
    if config.num_eval_examples <= 0:
        return []
    return openreward_rows(
        config=config,
        split=config.eval_split or config.split,
        num_examples=config.num_eval_examples,
    )


def openreward_toolset(task: Task, state: State) -> Toolset:
    del state
    spec = openreward_task_spec(task)
    tool_defs_value = spec.get("tools")
    if not isinstance(tool_defs_value, list):
        raise TypeError("OpenReward task data must include a tools list.")
    tool_defs = [openreward_tool_def(tool) for tool in tool_defs_value]
    return Toolset(tools=[OpenRewardTool(tool_def) for tool_def in tool_defs])


def openreward_rows(
    config: OpenRewardTasksetConfig,
    split: str,
    num_examples: int | None,
) -> list[ConfigData]:
    with openreward_client() as client:
        environment = client.environments.get(
            name=config.environment,
            variant=config.variant,
            base_url=config.base_url,
        )
        tasks = openreward_tasks(environment, split, num_examples)
        tools = environment.list_tools(format="openai")
    return [
        openreward_row(config, split, task, tools, index)
        for index, task in enumerate(tasks)
    ]


def openreward_tasks(
    environment: OpenRewardEnvironment, split: str, num_examples: int | None
) -> list[object]:
    if num_examples is None:
        return cast(list[object], environment.list_tasks(split=split))
    return cast(
        list[object],
        environment.get_task_range(split=split, start=0, stop=num_examples),
    )


def openreward_row(
    config: OpenRewardTasksetConfig,
    split: str,
    task: object,
    tools: Iterable[object],
    index: int,
) -> ConfigData:
    return {
        "example_id": index,
        "prompt": [
            {
                "role": "user",
                "content": "OpenReward rollout is initializing.",
            }
        ],
        "openreward": {
            "environment": config.environment,
            "variant": config.variant,
            "base_url": config.base_url,
            "split": split,
            "task": openreward_task_data(task),
            "tools": [serializable_openreward_tool(tool) for tool in tools],
        },
        "toolsets": {"openreward": {"fn": "tasksets.openreward:openreward_toolset"}},
    }


def openreward_client() -> OpenRewardClient:
    client_type = openreward_type("openreward", "OpenReward")
    if client_type is None:
        raise ImportError(
            "OpenRewardTaskset requires openreward. "
            "Install with: uv add 'tasksets[openreward]'"
        )
    return cast(OpenRewardClient, client_type())


def openreward_task_spec(task: Task) -> ConfigMap:
    spec = task.get("openreward")
    if not isinstance(spec, Mapping):
        raise TypeError("OpenReward tasks must contain an openreward mapping.")
    return cast(ConfigMap, spec)


def openreward_session(task: Task) -> OpenRewardSessionContext:
    spec = openreward_task_spec(task)
    config = OpenRewardTasksetConfig(
        environment=str(spec["environment"]),
        variant=cast(str | None, spec.get("variant")),
        base_url=cast(str | None, spec.get("base_url")),
    )
    client = openreward_client()
    environment = client.environments.get(
        name=config.environment,
        variant=config.variant,
        base_url=config.base_url,
    )
    session = environment.session(task=openreward_task_object(spec["task"]))
    return OpenRewardClientSession(client, session)


class OpenRewardClientSession:
    def __init__(self, client: OpenRewardClient, session: OpenRewardSessionContext):
        self.client = client
        self.session = session
        self.entered_session: object | None = None

    def __enter__(self) -> OpenRewardSession:
        entered = self.session.__enter__()
        self.entered_session = entered
        return entered

    def __exit__(self, *exc: object) -> object:
        try:
            return self.session.__exit__(*exc)
        finally:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()


def openreward_task_data(task: object) -> ConfigData:
    value = serializable_object_mapping(task)
    if not isinstance(value, Mapping):
        raise TypeError("OpenReward task must serialize to a mapping.")
    data = {str(key): item for key, item in value.items()}
    if "task_spec" not in data:
        return {"task_spec": dict(data)}
    return data


def openreward_task_object(value: object) -> object:
    if not isinstance(value, Mapping):
        raise TypeError("OpenReward task payload must be a mapping.")
    data = {str(key): item for key, item in value.items()}
    task_type = openreward_type("openreward.api.environments.types", "Task")
    required = {"server_name", "environment_name", "task_spec", "namespace"}
    if task_type is not None and required.issubset(data):
        return task_type(
            server_name=data["server_name"],
            environment_name=data["environment_name"],
            task_spec=data["task_spec"],
            namespace=data["namespace"],
        )
    return data.get("task_spec", data)


def openreward_prompt_messages(prompt: object) -> list[Message]:
    blocks = openreward_blocks_content(prompt)
    if isinstance(blocks, str):
        return normalize_messages(
            [{"role": "user", "content": blocks}], field_name="openreward.prompt"
        )
    return normalize_messages(
        [{"role": "user", "content": blocks}], field_name="openreward.prompt"
    )


def openreward_blocks_content(blocks: object) -> object:
    if not isinstance(blocks, Iterable) or isinstance(blocks, str | bytes | Mapping):
        return block_content(blocks)
    content: list[object] = []
    for block in blocks:
        content.append(block_content(block))
    text_blocks = [item for item in content if isinstance(item, str)]
    if len(text_blocks) == len(content):
        return "\n".join(text_blocks)
    return content


def block_content(block: object) -> object:
    block_type = getattr(block, "type", None)
    if block_type is None and isinstance(block, Mapping):
        block_map = cast(ConfigMap, block)
        block_type = block_map.get("type")
    if block_type == "text":
        text = getattr(block, "text", None)
        if text is None and isinstance(block, Mapping):
            block_map = cast(ConfigMap, block)
            text = block_map.get("text")
        return str(text or "")
    if block_type == "image":
        data = getattr(block, "data", None)
        mime_type = getattr(block, "mimeType", None)
        if isinstance(block, Mapping):
            block_map = cast(ConfigMap, block)
            data = block_map.get("data", data)
            mime_type = block_map.get("mimeType", mime_type)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{data}"},
        }
    mapped = serializable_object_mapping(block)
    if isinstance(mapped, Mapping):
        return {str(key): item for key, item in mapped.items()}
    return str(block)


def serializable_openreward_tool(tool: object) -> ConfigData:
    data = serializable_object_mapping(tool)
    if not isinstance(data, Mapping):
        raise TypeError("OpenReward tool must serialize to a mapping.")
    return {str(key): item for key, item in data.items()}


def openreward_tool_def(tool: object) -> Tool:
    if not isinstance(tool, Mapping):
        raise TypeError("OpenReward tool data must be a mapping.")
    tool_map = cast(ConfigMap, tool)
    data = {str(key): value for key, value in tool_map.items()}
    function_data = data.get("function")
    if isinstance(function_data, Mapping):
        data = {str(key): value for key, value in function_data.items()}
        if "parameters" not in data and "parameters" in tool_map:
            data["parameters"] = tool_map["parameters"]
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("OpenReward tool is missing a name.")
    description = data.get("description")
    parameters = data.get("parameters")
    if parameters is None:
        parameters = data.get("input_schema") or data.get("inputSchema")
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    if not isinstance(parameters, Mapping):
        raise TypeError(f"OpenReward tool {name!r} parameters must be a mapping.")
    return Tool(
        name=name,
        description=str(description or ""),
        parameters={str(key): value for key, value in parameters.items()},
    )


def record_openreward_step_reward(
    state: State, reward_value: float | int | None
) -> None:
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


def openreward_type(module_name: str, attr: str) -> type[object] | None:
    try:
        return cast(type[object], getattr(import_module(module_name), attr))
    except ImportError:
        return None


def serializable_object_mapping(value: object) -> object:
    dumped = serializable(value)
    if isinstance(dumped, Mapping):
        return dumped
    try:
        return serializable(vars(value))
    except TypeError:
        return dumped

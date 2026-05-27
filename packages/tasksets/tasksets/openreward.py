import asyncio
import json
from collections.abc import Iterable, Mapping
from typing import cast

from openreward import OpenReward
from openreward.api.environments.client import (
    Environment as OpenRewardEnvironment,
    Session as OpenRewardSession,
)
from openreward.api.environments.types import (
    ImageBlock as OpenRewardImageBlock,
    JSONObject as OpenRewardJSONObject,
    Task as OpenRewardTask,
    TextBlock as OpenRewardTextBlock,
    ToolOutput as OpenRewardToolOutput,
)
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
        session = await ensure_openreward_session(task, state)
        prompt = await asyncio.to_thread(session.get_prompt)
        state["prompt"] = openreward_prompt_messages(prompt)

    @cleanup
    async def cleanup_openreward(self, state: State) -> None:
        session_context = state.pop("openreward_session_context", None)
        state.pop("openreward_session", None)
        if session_context is None:
            return
        await asyncio.to_thread(session_context.__exit__, None, None, None)

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
        session = openreward_session_from_state(state)
        result = await asyncio.to_thread(
            session.call_tool,
            self.name,
            openreward_json_object(kwargs, "OpenReward tool input"),
        )
        record_openreward_tool_output(state, result)
        content = openreward_blocks_content(result.blocks)
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


async def openreward_toolset(task: Task, state: State) -> Toolset:
    session = await ensure_openreward_session(task, state)
    tool_specs = await asyncio.to_thread(session.list_tools, "openai")
    return Toolset(
        tools=[
            OpenRewardTool(openreward_tool_def(cast(ConfigMap, tool_spec)))
            for tool_spec in tool_specs
        ]
    )


def openreward_rows(
    config: OpenRewardTasksetConfig,
    split: str,
    num_examples: int | None,
) -> list[ConfigData]:
    with OpenReward() as client:
        environment = client.environments.get(
            name=config.environment,
            variant=config.variant,
            base_url=config.base_url,
        )
        tasks = openreward_tasks(environment, split, num_examples)
    return [
        openreward_row(config, split, task, index) for index, task in enumerate(tasks)
    ]


def openreward_tasks(
    environment: OpenRewardEnvironment, split: str, num_examples: int | None
) -> list[OpenRewardTask]:
    if num_examples is None:
        return environment.list_tasks(split=split)
    return environment.get_task_range(split=split, start=0, stop=num_examples)


def openreward_row(
    config: OpenRewardTasksetConfig,
    split: str,
    task: OpenRewardTask,
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
        },
        "toolsets": {"openreward": {"fn": "tasksets.openreward:openreward_toolset"}},
    }


async def ensure_openreward_session(task: Task, state: State) -> OpenRewardSession:
    session = state.get("openreward_session")
    if session is not None:
        return cast(OpenRewardSession, session)
    session_context = await asyncio.to_thread(openreward_session_context, task)
    entered_session = await asyncio.to_thread(session_context.__enter__)
    state["openreward_session"] = entered_session
    state["openreward_session_context"] = session_context
    return entered_session


def openreward_session_from_state(state: State) -> OpenRewardSession:
    session = state.get("openreward_session")
    if session is None:
        raise RuntimeError("OpenReward session is not initialized.")
    return cast(OpenRewardSession, session)


def openreward_session_context(task: Task) -> "OpenRewardClientSession":
    spec = openreward_task_spec(task)
    client = OpenReward()
    environment = client.environments.get(
        name=str(spec["environment"]),
        variant=cast(str | None, spec.get("variant")),
        base_url=cast(str | None, spec.get("base_url")),
    )
    session = environment.session(task=openreward_task_object(spec["task"]))
    return OpenRewardClientSession(client, session)


class OpenRewardClientSession:
    def __init__(self, client: OpenReward, session: OpenRewardSession):
        self.client = client
        self.session = session

    def __enter__(self) -> OpenRewardSession:
        return self.session.__enter__()

    def __exit__(self, *exc: object) -> object:
        try:
            return self.session.__exit__(*exc)
        finally:
            self.client.close()


def openreward_task_spec(task: Task) -> ConfigMap:
    spec = task.get("openreward")
    if not isinstance(spec, Mapping):
        raise TypeError("OpenReward tasks must contain an openreward mapping.")
    return cast(ConfigMap, spec)


def openreward_task_data(task: OpenRewardTask) -> ConfigData:
    task_spec = serializable(task.task_spec)
    if not isinstance(task_spec, Mapping):
        raise TypeError("OpenReward task_spec must serialize to a mapping.")
    return {
        "server_name": task.server_name,
        "environment_name": task.environment_name,
        "namespace": task.namespace,
        "task_spec": {str(key): value for key, value in task_spec.items()},
    }


def openreward_task_object(value: object) -> OpenRewardTask:
    if not isinstance(value, Mapping):
        raise TypeError("OpenReward task payload must be a mapping.")
    data = {str(key): item for key, item in value.items()}
    missing = sorted(
        {"server_name", "environment_name", "namespace", "task_spec"} - set(data)
    )
    if missing:
        raise ValueError(f"OpenReward task payload missing fields: {missing}.")
    task_spec = data["task_spec"]
    if not isinstance(task_spec, Mapping):
        raise TypeError("OpenReward task payload task_spec must be a mapping.")
    task_spec_data = {str(key): item for key, item in task_spec.items()}
    namespace = data["namespace"]
    if namespace is not None and not isinstance(namespace, str):
        raise TypeError("OpenReward task payload namespace must be a string or None.")
    return OpenRewardTask(
        server_name=str(data["server_name"]),
        environment_name=str(data["environment_name"]),
        namespace=namespace,
        task_spec=openreward_json_object(task_spec_data, "OpenReward task_spec"),
    )


def openreward_prompt_messages(prompt: object) -> list[Message]:
    blocks = openreward_blocks_content(prompt)
    return normalize_messages(
        [{"role": "user", "content": blocks}], field_name="openreward.prompt"
    )


def record_openreward_tool_output(state: State, output: OpenRewardToolOutput) -> None:
    if output.reward is not None:
        record_openreward_step_reward(state, output.reward)
    state["openreward_finished"] = output.finished
    if output.finished:
        state.stop("openreward_done")
    if output.metadata is not None:
        state["openreward_metadata"] = serializable(output.metadata)


def openreward_blocks_content(blocks: object) -> object:
    if isinstance(blocks, OpenRewardTextBlock | OpenRewardImageBlock):
        return openreward_block_content(blocks)
    if not isinstance(blocks, Iterable) or isinstance(blocks, str | bytes | Mapping):
        return openreward_block_content(blocks)
    content = [openreward_block_content(block) for block in blocks]
    text_blocks = [item for item in content if isinstance(item, str)]
    if len(text_blocks) == len(content):
        return "\n".join(text_blocks)
    return content


def openreward_block_content(block: object) -> object:
    if isinstance(block, OpenRewardTextBlock):
        return block.text
    if isinstance(block, OpenRewardImageBlock):
        return openreward_image_content(block.data, block.mimeType)
    if isinstance(block, Mapping):
        block_map = cast(ConfigMap, block)
        block_type = block_map.get("type")
        if block_type == "text":
            return str(block_map.get("text") or "")
        if block_type == "image":
            return openreward_image_content(
                str(block_map.get("data") or ""),
                str(block_map.get("mimeType") or ""),
            )
        return {str(key): value for key, value in block_map.items()}
    return str(block)


def openreward_image_content(data: str, mime_type: str) -> ConfigData:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{data}"},
    }


def openreward_tool_def(tool: ConfigMap) -> Tool:
    data = {str(key): value for key, value in tool.items()}
    function_data = data.get("function")
    if isinstance(function_data, Mapping):
        data = {str(key): value for key, value in function_data.items()}
        if "parameters" not in data and "parameters" in tool:
            data["parameters"] = tool["parameters"]
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


def openreward_json_object(
    value: Mapping[str, object], context: str
) -> OpenRewardJSONObject:
    data = serializable(value)
    if not isinstance(data, Mapping):
        raise TypeError(f"{context} must serialize to a JSON object.")
    return cast(OpenRewardJSONObject, {str(key): item for key, item in data.items()})


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

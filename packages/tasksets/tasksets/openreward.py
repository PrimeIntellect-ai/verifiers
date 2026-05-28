import asyncio
from collections.abc import Iterable, Mapping
from typing import cast

from openreward import OpenReward
from openreward.api.environments.client import Session as OpenRewardSession
from openreward.api.environments.types import (
    ImageBlock as OpenRewardImageBlock,
    JSONObject as OpenRewardJSONObject,
    Task as OpenRewardTask,
    TextBlock as OpenRewardTextBlock,
    ToolOutput as OpenRewardToolOutput,
)
import verifiers as vf
from verifiers.types import MessageContent, Tool
from verifiers.utils.message_utils import normalize_messages
from verifiers.v1.config import ConfigSource
from verifiers.v1.state import State
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.toolset import Toolset, Toolsets
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


class OpenRewardTaskset(Taskset[OpenRewardTasksetConfig]):
    def __init__(self, config: ConfigSource = None):
        config_value = coerce_config(OpenRewardTasksetConfig, config)
        super().__init__(config=config_value)
        self.taskset_id = self.config.taskset_id or "openreward"

    def load_toolsets(self) -> Toolsets:
        return {"openreward": Toolset(scope="rollout")}

    def load_tasks(self) -> list[ConfigData]:
        with OpenReward() as client:
            environment = client.environments.get(
                name=self.config.environment,
                variant=self.config.variant,
                base_url=self.config.base_url,
            )
            tasks = (
                environment.list_tasks(split=self.config.split)
                if self.config.num_train_examples is None
                else environment.get_task_range(
                    split=self.config.split,
                    start=0,
                    stop=self.config.num_train_examples,
                )
            )
        return [
            self._task(self.config.split, task, index)
            for index, task in enumerate(tasks)
        ]

    def load_eval_tasks(self) -> list[ConfigData]:
        if self.config.num_eval_examples <= 0:
            return []
        split = self.config.eval_split or self.config.split
        with OpenReward() as client:
            environment = client.environments.get(
                name=self.config.environment,
                variant=self.config.variant,
                base_url=self.config.base_url,
            )
            tasks = environment.get_task_range(
                split=split,
                start=0,
                stop=self.config.num_eval_examples,
            )
        return [self._task(split, task, index) for index, task in enumerate(tasks)]

    def _task(self, split: str, task: OpenRewardTask, index: int) -> ConfigData:
        task_spec = serializable(task.task_spec)
        assert isinstance(task_spec, Mapping)
        return {
            "example_id": index,
            "prompt": [
                {"role": "user", "content": "OpenReward rollout is initializing."}
            ],
            "openreward": {
                "environment": self.config.environment,
                "variant": self.config.variant,
                "base_url": self.config.base_url,
                "split": split,
                "task": {
                    "server_name": task.server_name,
                    "environment_name": task.environment_name,
                    "namespace": task.namespace,
                    "task_spec": {str(key): value for key, value in task_spec.items()},
                },
            },
        }

    @vf.setup
    async def setup_openreward(self, task: Task, state: State) -> None:
        session = await self._session(task, state)
        prompt = await asyncio.to_thread(session.get_prompt)
        state["prompt"] = normalize_messages(
            [{"role": "user", "content": self._content(prompt)}],
            field_name="openreward.prompt",
        )
        tool_specs = await asyncio.to_thread(session.list_tools, "openai")
        for tool_spec in tool_specs:
            state.add_tool(
                "openreward",
                OpenRewardTool(self._tool_def(cast(ConfigMap, tool_spec))),
            )

    @vf.cleanup
    async def cleanup_openreward(self, state: State) -> None:
        session = state.pop("openreward_session", None)
        client = state.pop("openreward_client", None)
        if session is not None:
            await asyncio.to_thread(session.__exit__, None, None, None)
        if client is not None:
            await asyncio.to_thread(client.close)

    @vf.stop
    async def openreward_done(self, state: State) -> bool:
        return bool(state.get("openreward_finished"))

    @vf.reward(weight=1.0)
    async def openreward_reward(self, state: State) -> float:
        return float(
            sum(
                float(step.get("reward", 0.0) or 0.0)
                for step in state.get("trajectory", [])
                if isinstance(step, Mapping)
            )
        )

    async def _session(self, task: Task, state: State) -> OpenRewardSession:
        session = state.get("openreward_session")
        if session is not None:
            return cast(OpenRewardSession, session)
        spec = task["openreward"]
        assert isinstance(spec, Mapping)
        task_data = spec["task"]
        assert isinstance(task_data, Mapping)
        task_spec = task_data["task_spec"]
        assert isinstance(task_spec, Mapping)
        client = OpenReward()
        state["openreward_client"] = client
        environment = client.environments.get(
            name=str(spec["environment"]),
            variant=cast(str | None, spec["variant"]),
            base_url=cast(str | None, spec["base_url"]),
        )
        session = environment.session(
            task=OpenRewardTask(
                server_name=str(task_data["server_name"]),
                environment_name=str(task_data["environment_name"]),
                namespace=cast(str | None, task_data["namespace"]),
                task_spec=cast(
                    OpenRewardJSONObject,
                    {str(key): value for key, value in task_spec.items()},
                ),
            )
        )
        entered_session = await asyncio.to_thread(session.__enter__)
        state["openreward_session"] = entered_session
        return entered_session

    @classmethod
    def _tool_def(cls, tool: ConfigMap) -> Tool:
        function_data = tool.get("function")
        data = (
            cast(ConfigMap, function_data)
            if isinstance(function_data, Mapping)
            else tool
        )
        name = data["name"]
        assert isinstance(name, str)
        parameters = (
            data.get("parameters")
            or data.get("input_schema")
            or data.get("inputSchema")
            or {"type": "object", "properties": {}}
        )
        assert isinstance(parameters, Mapping)
        return Tool(
            name=name,
            description=str(data.get("description") or ""),
            parameters={str(key): value for key, value in parameters.items()},
        )

    @classmethod
    def _content(cls, blocks: object) -> MessageContent:
        block_list = (
            list(blocks)
            if isinstance(blocks, Iterable) and not isinstance(blocks, str | bytes)
            else [blocks]
        )
        if all(isinstance(block, OpenRewardTextBlock) for block in block_list):
            text_blocks = cast(list[OpenRewardTextBlock], block_list)
            return "\n".join(block.text for block in text_blocks)
        content: list[ConfigData] = []
        for block in block_list:
            if isinstance(block, OpenRewardTextBlock):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, OpenRewardImageBlock):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.mimeType};base64,{block.data}"
                        },
                    }
                )
            else:
                assert False, f"Unexpected OpenReward block: {block!r}"
        return cast(MessageContent, content)


class OpenRewardTool:
    def __init__(self, tool_def: Tool):
        self.tool_def = tool_def
        self.name = tool_def.name
        self.__name__ = tool_def.name
        self.__doc__ = tool_def.description

    async def __call__(self, state: State, **kwargs: object) -> MessageContent:
        session = cast(OpenRewardSession, state["openreward_session"])
        arguments = serializable(kwargs)
        assert isinstance(arguments, Mapping)
        result = await asyncio.to_thread(
            session.call_tool,
            self.name,
            cast(
                OpenRewardJSONObject,
                {str(key): value for key, value in arguments.items()},
            ),
        )
        self._record_output(state, result)
        return OpenRewardTaskset._content(result.blocks)

    def _record_output(self, state: State, output: OpenRewardToolOutput) -> None:
        if output.reward is not None:
            trajectory = state["trajectory"]
            assert isinstance(trajectory, list)
            step = trajectory[-1]
            assert isinstance(step, dict)
            step["reward"] = float(step.get("reward", 0.0) or 0.0) + float(
                output.reward
            )
        state["openreward_finished"] = output.finished
        if output.finished:
            state.stop("openreward_done")
        if output.metadata is not None:
            state["openreward_metadata"] = serializable(output.metadata)


def load_taskset(config: OpenRewardTasksetConfig) -> OpenRewardTaskset:
    return OpenRewardTaskset(config=config)

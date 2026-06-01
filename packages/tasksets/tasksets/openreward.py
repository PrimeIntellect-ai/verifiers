import asyncio
from collections.abc import Iterable
from typing import cast

from openreward import OpenReward
from openreward.api.environments.client import Session as OpenRewardAPISession
from openreward.api.environments.types import (
    ImageBlock as OpenRewardImageBlock,
    JSONObject as OpenRewardJSONObject,
    Task as OpenRewardTask,
    TextBlock as OpenRewardTextBlock,
    ToolOutput as OpenRewardToolOutput,
)
import verifiers as vf
from verifiers.v1.utils.serialization_utils import serializable


class OpenRewardTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "openreward"
    bindings: vf.BindingsConfig = vf.BindingsConfig.model_validate(
        {"session.task": "task"}
    )
    objects: vf.ObjectsConfig = vf.ObjectsConfig.model_validate(
        {"session": "tasksets.openreward:OpenRewardSession"}
    )
    environment: str
    variant: str | None = None
    base_url: str | None = None
    split: str = "train"
    eval_split: str | None = None
    num_train_examples: int | None = None
    num_eval_examples: int = 0


class OpenRewardSession:
    def __init__(self, task: vf.Task):
        self.task = task
        self.client: OpenReward | None = None
        self.session_context: OpenRewardAPISession | None = None
        self.session: OpenRewardAPISession | None = None

    async def start(self) -> OpenRewardAPISession:
        if self.session is not None:
            return self.session
        spec = self.task["openreward"]
        assert isinstance(spec, dict)
        task_data = spec["task"]
        assert isinstance(task_data, dict)
        task_spec = task_data["task_spec"]
        assert isinstance(task_spec, dict)
        client = OpenReward()
        self.client = client
        environment = await asyncio.to_thread(
            client.environments.get,
            name=str(spec["environment"]),
            variant=cast(str | None, spec["variant"]),
            base_url=cast(str | None, spec["base_url"]),
        )
        self.session_context = environment.session(
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
        self.session = await asyncio.to_thread(self.session_context.__enter__)
        return self.session

    async def prompt(self) -> object:
        session = await self.start()
        return await asyncio.to_thread(session.get_prompt)

    async def tool_specs(self) -> Iterable[object]:
        session = await self.start()
        return cast(
            Iterable[object], await asyncio.to_thread(session.list_tools, "openai")
        )

    async def call_tool(
        self, name: str, arguments: vf.JsonData
    ) -> OpenRewardToolOutput:
        session = await self.start()
        result = await asyncio.to_thread(
            session.call_tool, name, cast(OpenRewardJSONObject, dict(arguments))
        )
        return cast(OpenRewardToolOutput, result)

    def content(self, blocks: object) -> vf.MessageContent:
        block_list = (
            list(blocks)
            if isinstance(blocks, Iterable) and not isinstance(blocks, str | bytes)
            else [blocks]
        )
        if all(isinstance(block, OpenRewardTextBlock) for block in block_list):
            text_blocks = cast(list[OpenRewardTextBlock], block_list)
            return "\n".join(block.text for block in text_blocks)
        content: list[vf.JsonData] = []
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
        return cast(vf.MessageContent, content)

    async def close(self) -> None:
        if self.session_context is not None:
            await asyncio.to_thread(self.session_context.__exit__, None, None, None)
        if self.client is not None:
            await asyncio.to_thread(self.client.close)
        self.session_context = None
        self.session = None
        self.client = None


class OpenRewardTaskset(vf.Taskset[OpenRewardTasksetConfig]):
    def load_toolsets(self, config: OpenRewardTasksetConfig) -> vf.Toolsets:
        return {"openreward": vf.Toolset(scope="rollout", handler=self.call_tool)}

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        config = self.config
        task_split = config.split
        num_examples = config.num_train_examples
        if split == "eval":
            if config.num_eval_examples <= 0:
                return []
            task_split = config.eval_split or config.split
            num_examples = config.num_eval_examples
        with OpenReward() as client:
            environment = client.environments.get(
                name=config.environment,
                variant=config.variant,
                base_url=config.base_url,
            )
            tasks = (
                environment.list_tasks(split=task_split)
                if num_examples is None
                else environment.get_task_range(
                    split=task_split,
                    start=0,
                    stop=num_examples,
                )
            )
        data: list[vf.JsonData] = []
        for task in tasks:
            task_spec = serializable(task.task_spec)
            assert isinstance(task_spec, dict)
            data.append(
                {
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
                        "split": task_split,
                        "task": {
                            "server_name": task.server_name,
                            "environment_name": task.environment_name,
                            "namespace": task.namespace,
                            "task_spec": {
                                str(key): value for key, value in task_spec.items()
                            },
                        },
                    },
                }
            )
        return data

    @vf.setup
    async def setup_openreward(self, task: vf.Task, state: vf.State) -> None:
        session = await self.get_object("session", task, state)
        assert isinstance(session, OpenRewardSession)
        prompt = await session.prompt()
        state["prompt"] = [vf.UserMessage(content=session.content(prompt))]
        for tool_spec in await session.tool_specs():
            tool_value = serializable(tool_spec)
            assert isinstance(tool_value, dict)
            tool_data = cast(vf.ConfigData, tool_value)
            function_data = tool_data.get("function")
            data = (
                cast(vf.ConfigData, function_data)
                if isinstance(function_data, dict)
                else tool_data
            )
            name = data["name"]
            assert isinstance(name, str)
            parameters = (
                data.get("parameters")
                or data.get("input_schema")
                or data.get("inputSchema")
                or {"type": "object", "properties": {}}
            )
            assert isinstance(parameters, dict)
            state.add_tool(
                "openreward",
                vf.Tool(
                    name=name,
                    description=str(data.get("description") or ""),
                    parameters={str(key): value for key, value in parameters.items()},
                ),
            )

    @vf.stop
    async def openreward_done(self, state: vf.State) -> bool:
        return bool(state.get("openreward_finished"))

    @vf.reward(weight=1.0)
    async def openreward_reward(self, state: vf.State) -> float:
        return state.total_step_reward()

    async def call_tool(
        self, task: vf.Task, state: vf.State, tool: vf.Tool, arguments: vf.JsonData
    ) -> vf.MessageContent:
        session = await self.get_object("session", task, state)
        assert isinstance(session, OpenRewardSession)
        tool_arguments = serializable(arguments)
        assert isinstance(tool_arguments, dict)
        result = await session.call_tool(
            tool.name,
            cast(
                vf.JsonData, {str(key): value for key, value in tool_arguments.items()}
            ),
        )
        state.add_step_reward(result.reward)
        state["openreward_finished"] = result.finished
        if result.finished:
            state.stop("openreward_done")
        if result.metadata is not None:
            state["openreward_metadata"] = serializable(result.metadata)
        return session.content(result.blocks)


def load_taskset(config: OpenRewardTasksetConfig) -> OpenRewardTaskset:
    return OpenRewardTaskset(config=config)

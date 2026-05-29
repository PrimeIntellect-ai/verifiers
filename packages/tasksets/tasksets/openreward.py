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
)
import verifiers as vf
from verifiers.utils.message_utils import normalize_messages
from verifiers.v1.utils.serialization_utils import serializable


def openreward_content(blocks: object) -> vf.MessageContent:
    block_list = (
        list(blocks)
        if isinstance(blocks, Iterable) and not isinstance(blocks, str | bytes)
        else [blocks]
    )
    if all(isinstance(block, OpenRewardTextBlock) for block in block_list):
        text_blocks = cast(list[OpenRewardTextBlock], block_list)
        return "\n".join(block.text for block in text_blocks)
    content: list[vf.ConfigData] = []
    for block in block_list:
        if isinstance(block, OpenRewardTextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, OpenRewardImageBlock):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                }
            )
        else:
            assert False, f"Unexpected OpenReward block: {block!r}"
    return cast(vf.MessageContent, content)


class OpenRewardTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "openreward"
    environment: str
    variant: str | None = None
    base_url: str | None = None
    split: str = "train"
    eval_split: str | None = None
    num_train_examples: int | None = None
    num_eval_examples: int = 0


class OpenRewardTaskset(vf.Taskset[OpenRewardTasksetConfig]):
    def load_toolsets(self) -> vf.Toolsets:
        return {"openreward": vf.Toolset(scope="rollout", handler=self.call_tool)}

    def load_tasks(self, split: vf.TaskSplit = "train") -> list[vf.ConfigData]:
        task_split = self.config.split
        num_examples = self.config.num_train_examples
        if split == "eval":
            if self.config.num_eval_examples <= 0:
                return []
            task_split = self.config.eval_split or self.config.split
            num_examples = self.config.num_eval_examples
        with OpenReward() as client:
            environment = client.environments.get(
                name=self.config.environment,
                variant=self.config.variant,
                base_url=self.config.base_url,
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
        data: list[vf.ConfigData] = []
        for task in tasks:
            task_spec = serializable(task.task_spec)
            assert isinstance(task_spec, Mapping)
            data.append(
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": "OpenReward rollout is initializing.",
                        }
                    ],
                    "openreward": {
                        "environment": self.config.environment,
                        "variant": self.config.variant,
                        "base_url": self.config.base_url,
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
        session = await asyncio.to_thread(session.__enter__)
        state["openreward_session"] = session
        prompt = await asyncio.to_thread(session.get_prompt)
        state["prompt"] = normalize_messages(
            [{"role": "user", "content": openreward_content(prompt)}],
            field_name="openreward.prompt",
        )
        tool_specs = await asyncio.to_thread(session.list_tools, "openai")
        for tool_spec in tool_specs:
            tool_data = cast(vf.ConfigMap, tool_spec)
            function_data = tool_data.get("function")
            data = (
                cast(vf.ConfigMap, function_data)
                if isinstance(function_data, Mapping)
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
            assert isinstance(parameters, Mapping)
            state.add_tool(
                "openreward",
                vf.Tool(
                    name=name,
                    description=str(data.get("description") or ""),
                    parameters={str(key): value for key, value in parameters.items()},
                ),
            )

    @vf.cleanup
    async def cleanup_openreward(self, state: vf.State) -> None:
        session = state.pop("openreward_session", None)
        client = state.pop("openreward_client", None)
        if session is not None:
            await asyncio.to_thread(session.__exit__, None, None, None)
        if client is not None:
            await asyncio.to_thread(client.close)

    @vf.stop
    async def openreward_done(self, state: vf.State) -> bool:
        return bool(state.get("openreward_finished"))

    @vf.reward(weight=1.0)
    async def openreward_reward(self, state: vf.State) -> float:
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
        session = cast(OpenRewardSession, state["openreward_session"])
        tool_arguments = serializable(arguments)
        assert isinstance(tool_arguments, Mapping)
        result = await asyncio.to_thread(
            session.call_tool,
            tool.name,
            cast(
                OpenRewardJSONObject,
                {str(key): value for key, value in tool_arguments.items()},
            ),
        )
        if result.reward is not None:
            trajectory = state["trajectory"]
            assert isinstance(trajectory, list)
            step = trajectory[-1]
            assert isinstance(step, dict)
            step["reward"] = float(step.get("reward", 0.0) or 0.0) + float(
                result.reward
            )
        state["openreward_finished"] = result.finished
        if result.finished:
            state.stop("openreward_done")
        if result.metadata is not None:
            state["openreward_metadata"] = serializable(result.metadata)
        return openreward_content(result.blocks)


def load_taskset(config: OpenRewardTasksetConfig) -> OpenRewardTaskset:
    return OpenRewardTaskset(config=config)

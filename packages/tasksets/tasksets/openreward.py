import asyncio
from collections.abc import Iterable
from typing import Protocol, cast

from openreward import OpenReward
from openreward.api.environments.client import Session as OpenRewardAPISession
from openreward.api.environments.types import (
    ImageBlock as OpenRewardImageBlock,
    JSONObject as OpenRewardJSONObject,
    Task as OpenRewardTask,
    TextBlock as OpenRewardTextBlock,
    ToolOutput as OpenRewardToolOutput,
)

import verifiers.v1 as vf
from verifiers.v1.utils.json_utils import json_data, json_value


class OpenRewardSplit(Protocol):
    name: str
    type: str


class OpenRewardEnvironment(Protocol):
    def list_splits(self) -> Iterable[OpenRewardSplit]: ...

    def list_tasks(self, split: str) -> Iterable[OpenRewardTask]: ...

    def get_task_range(
        self,
        split: str,
        start: int | None = None,
        stop: int | None = None,
    ) -> Iterable[OpenRewardTask]: ...


class OpenRewardUserConfig(vf.UserConfig):
    pass


class OpenRewardTasksetConfig(vf.TasksetConfig):
    id: str | None = "openreward"
    user: vf.UserConfig | None = OpenRewardUserConfig()
    environment: str
    variant: str | None = None
    base_url: str | None = None
    split: str = "train"
    num_train_examples: int | None = None
    num_eval_examples: int = 0


class OpenRewardVFTask(vf.Task, frozen=True):
    openreward: vf.JsonData


class OpenRewardSession:
    def __init__(self, task: vf.JsonData):
        self.task = task
        self.client: OpenReward | None = None
        self.session_context: OpenRewardAPISession | None = None
        self.session: OpenRewardAPISession | None = None

    async def start(self) -> OpenRewardAPISession:
        if self.session is not None:
            return self.session
        spec = self.task["openreward"]
        if not isinstance(spec, dict):
            raise TypeError("OpenReward task requires openreward config.")
        spec_data = spec
        task_data = spec_data.get("task")
        if not isinstance(task_data, dict):
            raise TypeError("OpenReward task requires task data.")
        task_spec = task_data.get("task_spec")
        if not isinstance(task_spec, dict):
            raise TypeError("OpenReward task_spec must be a mapping.")
        variant = spec_data.get("variant")
        if variant is not None and not isinstance(variant, str):
            raise TypeError("OpenReward variant must be a string or null.")
        base_url = spec_data.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise TypeError("OpenReward base_url must be a string or null.")
        namespace = task_data.get("namespace")
        if namespace is not None and not isinstance(namespace, str):
            raise TypeError("OpenReward namespace must be a string or null.")
        client = OpenReward()
        self.client = client
        environment = await asyncio.to_thread(
            client.environments.get,
            name=str(spec_data["environment"]),
            variant=variant,
            base_url=base_url,
        )
        self.session_context = environment.session(
            task=OpenRewardTask(
                server_name=str(task_data["server_name"]),
                environment_name=str(task_data["environment_name"]),
                namespace=namespace,
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

    async def call_tool(self, name: str, input: vf.JsonData) -> OpenRewardToolOutput:
        session = await self.start()
        return cast(
            OpenRewardToolOutput,
            await asyncio.to_thread(session.call_tool, name, input),
        )

    async def tool_defs(self) -> list[vf.JsonData]:
        session = await self.start()
        tools = await asyncio.to_thread(session.list_tools)
        return openreward_tool_defs(tools)

    def content(self, blocks: object) -> vf.MessageContent:
        block_list = (
            list(blocks)
            if isinstance(blocks, Iterable) and not isinstance(blocks, str | bytes)
            else [blocks]
        )
        if all(isinstance(block, OpenRewardTextBlock) for block in block_list):
            return "\n".join(
                block.text for block in cast(list[OpenRewardTextBlock], block_list)
            )
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
                raise TypeError(f"Unexpected OpenReward block: {block!r}")
        return cast(vf.MessageContent, content)

    async def close(self) -> None:
        if self.session_context is not None:
            await asyncio.to_thread(self.session_context.__exit__, None, None, None)
        if self.client is not None:
            await asyncio.to_thread(self.client.close)
        self.session_context = None
        self.session = None
        self.client = None


def openreward_tool_defs(tools: Iterable[object]) -> list[vf.JsonData]:
    tool_defs: list[vf.JsonData] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError("OpenReward tools must have non-empty names.")
        description = getattr(tool, "description", "") or ""
        schema = getattr(tool, "input_schema", None)
        parameters = (
            json_data(schema, context=f"OpenReward tool {name} schema")
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
                context=f"OpenReward tool {name}",
            )
        )
    return tool_defs


class OpenRewardTaskset(vf.Taskset[OpenRewardTasksetConfig]):
    task_type = OpenRewardVFTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            if self.config.num_eval_examples <= 0:
                return []
            return self.openreward_test_tasks(
                num_examples=self.config.num_eval_examples
            )
        return self.openreward_tasks(
            task_split=self.config.split,
            num_examples=self.config.num_train_examples,
        )

    def openreward_tasks(
        self,
        *,
        task_split: str,
        num_examples: int | None,
    ) -> list[vf.JsonData]:
        config = self.config
        with OpenReward() as client:
            environment = cast(
                OpenRewardEnvironment,
                client.environments.get(
                    name=config.environment,
                    variant=config.variant,
                    base_url=config.base_url,
                ),
            )
            tasks = self.openreward_source_tasks(environment, task_split, num_examples)
        return self.openreward_task_records(tasks, task_split)

    def openreward_test_tasks(self, *, num_examples: int) -> list[vf.JsonData]:
        config = self.config
        with OpenReward() as client:
            environment = cast(
                OpenRewardEnvironment,
                client.environments.get(
                    name=config.environment,
                    variant=config.variant,
                    base_url=config.base_url,
                ),
            )
            task_split = self.openreward_test_split(environment)
            if task_split is None:
                return []
            tasks = self.openreward_source_tasks(environment, task_split, num_examples)
        return self.openreward_task_records(tasks, task_split)

    def openreward_test_split(self, environment: OpenRewardEnvironment) -> str | None:
        for split in environment.list_splits():
            if split.type == "test":
                return split.name
        return None

    def openreward_source_tasks(
        self,
        environment: OpenRewardEnvironment,
        task_split: str,
        num_examples: int | None,
    ) -> Iterable[OpenRewardTask]:
        if num_examples is None:
            return environment.list_tasks(split=task_split)
        return environment.get_task_range(split=task_split, start=0, stop=num_examples)

    def openreward_task_records(
        self,
        tasks: Iterable[OpenRewardTask],
        task_split: str,
    ) -> list[vf.JsonData]:
        config = self.config
        data: list[vf.JsonData] = []
        for task in tasks:
            task_spec = json_value(task.task_spec)
            if not isinstance(task_spec, dict):
                raise TypeError("OpenReward task_spec must serialize to a mapping.")
            data.append(
                json_data(
                    {
                        "prompt": [],
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
                    },
                    context="OpenReward task record",
                )
            )
        return data

    @vf.reward(weight=1.0)
    async def openreward_reward(self, state: vf.State) -> float:
        return sum(float(turn.reward or 0.0) for turn in state.transcript)


class OpenRewardUser(vf.User[OpenRewardUserConfig]):
    session: OpenRewardSession | None

    def start(self) -> None:
        self.session = None

    def stop(self) -> None:
        if self.session is not None:
            asyncio.run(self.session.close())
        self.session = None

    @vf.tool(
        hidden=True,
        args={"task": "task"},
        sets={
            "prompt": "state.extras.openreward.prompt",
        },
    )
    async def setup(self, task: vf.JsonData) -> vf.JsonData:
        self.session = OpenRewardSession(task)
        prompt = await self.session.prompt()
        return json_data(
            {
                "prompt": json_value(self.session.content(prompt)),
                "tools": await self.session.tool_defs(),
            }
        )

    @vf.user(
        args={
            "prompt": "state.extras.openreward.prompt",
            "completion": "state.completion",
        },
        sets={
            "stop_condition": "state.stop_condition",
        },
    )
    async def respond(
        self,
        prompt: vf.JsonValue,
        completion: list[vf.JsonData],
    ) -> vf.JsonData:
        if self.session is None:
            raise RuntimeError("OpenReward setup has not started.")
        if completion:
            return json_data(
                {"messages": [], "stop_condition": "openreward_waiting_for_tools"}
            )
        return json_data(
            {
                "messages": [
                    json_data(
                        vf.UserMessage(
                            content=cast(vf.MessageContent, prompt)
                        ).model_dump(mode="json")
                    )
                ],
            }
        )

    @vf.tool(
        hidden=True,
        sets={
            "reward": "state.transcript.last.reward",
            "finished": "state.is_completed",
            "stop_condition": "state.stop_condition",
        },
    )
    async def call_tool(self, name: str, input: vf.JsonData) -> vf.JsonData:
        """Call an OpenReward task tool by name with JSON input."""
        if self.session is None:
            raise RuntimeError("OpenReward session has not started.")
        output = await self.session.call_tool(name, input)
        content = self.session.content(output.blocks)
        payload: dict[str, object] = {
            "content": content,
        }
        if output.reward is not None:
            payload["reward"] = float(output.reward)
        if output.finished:
            payload["finished"] = True
            payload["stop_condition"] = "openreward_finished"
        return json_data(payload)


def load_taskset(config: OpenRewardTasksetConfig) -> OpenRewardTaskset:
    return OpenRewardTaskset(config=config)

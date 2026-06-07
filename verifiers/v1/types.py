from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING, TypeAlias, cast

from datasets import Dataset
from pydantic import Field
from verifiers.clients import Client
from verifiers.types import (
    ClientConfig,
    Message,
    MessageContent,
    Messages,
    Response,
    SamplingArgs,
    Tool,
)
from typing_extensions import TypeAliasType

from .config import Config

if TYPE_CHECKING:
    from .mcp import MCPToolRegistry
    from .runtime import Runtime
    from .state import State
    from .task import Task

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonData: TypeAlias = dict[str, JsonValue]

HandlerResult: TypeAlias = (
    JsonValue | JsonData | Message | Messages | MessageContent | Sequence[float] | None
)
Handler: TypeAlias = Callable[..., HandlerResult | Awaitable[HandlerResult]]

TaskSplit: TypeAlias = Literal["train", "eval"]
Tasks: TypeAlias = Dataset | Iterable[JsonData] | Iterable["Task"]

PromptMessage: TypeAlias = Message | JsonData
PromptInput: TypeAlias = str | Sequence[PromptMessage]


class ModelConfig(Config):
    client: ClientConfig = Field(default_factory=ClientConfig)
    model: str
    sampling_args: JsonData = Field(default_factory=dict)


@dataclass(frozen=True)
class ModelClient:
    config: ModelConfig
    client: Client

    async def get_response(
        self,
        *,
        prompt: Messages,
        state: "State | None" = None,
        model: str | None = None,
        sampling_args: SamplingArgs | None = None,
        tools: list[Tool] | None = None,
    ) -> Response:
        kwargs: dict[str, object] = {}
        if state is not None:
            kwargs["state"] = client_state_record(state)
        return await self.client.get_response(
            prompt=prompt,
            model=model or self.config.model,
            sampling_args=sampling_args or dict(self.config.sampling_args),
            tools=tools,
            **kwargs,
        )


@dataclass
class Context:
    task: Task
    state: State
    model_client: ModelClient
    teacher: ModelClient | None = None
    runtime: Runtime | None = None
    tools: MCPToolRegistry | None = None
    user: MCPToolRegistry | None = None
    parent: Context | None = None
    score: bool = False
    scoring: bool = False

    @property
    def client(self) -> Client:
        return self.model_client.client

    @property
    def model(self) -> str:
        return self.model_client.config.model

    @property
    def sampling_args(self) -> SamplingArgs:
        return dict(self.model_client.config.sampling_args)

    def has_active_scoring(self) -> bool:
        context: Context | None = self
        while context is not None:
            if context.scoring:
                return True
            context = context.parent
        return False


def client_state_record(state: "State") -> JsonData:
    from .utils.json_utils import jsonable

    transcript: list[JsonValue] = []
    for turn in state.transcript:
        tokens = jsonable(turn.tokens.model_dump(mode="json")) if turn.tokens else None
        transcript.append(
            cast(
                JsonValue,
                jsonable(
                    {
                        "prompt": turn.prompt,
                        "completion": turn.completion,
                        "tokens": tokens,
                        "reward": turn.reward,
                        "is_truncated": turn.is_truncated,
                    }
                ),
            )
        )
    record = {
        "id": state.id,
        "task_id": state.task_id,
        "group_id": state.group_id,
        "extras": state.extras,
        "metadata": state.metadata,
        "transcript": transcript,
    }
    return cast(
        JsonData,
        jsonable({key: value for key, value in record.items() if value is not None}),
    )

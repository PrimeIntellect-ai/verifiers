from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Literal, TYPE_CHECKING, TypeAlias

from datasets import Dataset
from verifiers.clients import Client
from verifiers.types import ClientConfig, Message, MessageContent, Messages
from typing_extensions import TypeAliasType

if TYPE_CHECKING:
    from .task import Task

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonData: TypeAlias = dict[str, JsonValue]
ConfigValue = TypeAliasType(
    "ConfigValue",
    JsonScalar
    | list["ConfigValue"]
    | tuple["ConfigValue", ...]
    | dict[str, "ConfigValue"],
)
ConfigData: TypeAlias = dict[str, ConfigValue]

HandlerResult: TypeAlias = (
    JsonValue
    | JsonData
    | ConfigData
    | Message
    | Messages
    | MessageContent
    | Sequence[float]
    | None
)
Handler: TypeAlias = Callable[..., HandlerResult | Awaitable[HandlerResult]]

TaskSplit: TypeAlias = Literal["train", "eval"]
Tasks: TypeAlias = Dataset | Iterable[JsonData] | Iterable["Task"]

PromptMessage: TypeAlias = Message | JsonData
PromptInput: TypeAlias = str | Sequence[PromptMessage]

ModelClient: TypeAlias = Client | ClientConfig
RuntimeObject: TypeAlias = object
RuntimeData: TypeAlias = dict[str, RuntimeObject]
RuntimeCallableResult: TypeAlias = RuntimeObject | Awaitable[RuntimeObject]
RuntimeCallable: TypeAlias = Callable[..., RuntimeCallableResult]
ObjectFactoryResult: TypeAlias = RuntimeObject | Awaitable[RuntimeObject]
ObjectFactory: TypeAlias = Callable[..., ObjectFactoryResult]
Objects: TypeAlias = dict[str, ObjectFactory]
ToolParameters: TypeAlias = dict[str, RuntimeObject]

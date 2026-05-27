from collections.abc import (
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from os import PathLike
from typing import Literal, TypeAlias

from datasets import Dataset
from typing_extensions import TypeAliasType
from verifiers.clients import Client
from verifiers.types import ClientConfig, Message

Handler: TypeAlias = Callable[..., object]
GroupHandler: TypeAlias = Callable[..., Sequence[float] | Awaitable[Sequence[float]]]
ConfigMap: TypeAlias = Mapping[str, object]
ConfigData: TypeAlias = dict[str, object]
ConfigInputMap: TypeAlias = Mapping[object, object]
MutableConfigMap: TypeAlias = MutableMapping[str, object]
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonData: TypeAlias = dict[str, JsonValue]
HandlerList: TypeAlias = Iterable[Handler]

TaskRow: TypeAlias = Mapping[str, object]
Tasks: TypeAlias = Dataset | Iterable[TaskRow]
TaskLoader: TypeAlias = Callable[..., Tasks]

PromptMessage: TypeAlias = Message | ConfigMap
PromptInput: TypeAlias = str | Sequence[PromptMessage]
SystemPrompt: TypeAlias = PromptInput | PathLike[str]
ToolSpec: TypeAlias = Handler | str | ConfigMap
ToolSpecs: TypeAlias = ToolSpec | Sequence[ToolSpec]
ToolsetSpecs: TypeAlias = ToolSpec | Sequence[ToolSpec] | ConfigMap

ModelClient: TypeAlias = Client | ClientConfig

ProgramScalar: TypeAlias = str | int | float | bool | None
ProgramConfigValue = TypeAliasType(
    "ProgramConfigValue",
    ProgramScalar | list["ProgramConfigValue"] | dict[str, "ProgramConfigValue"],
)
ProgramValue: TypeAlias = ProgramConfigValue
ProgramCommand: TypeAlias = str | list[ProgramConfigValue]
ProgramMap: TypeAlias = Mapping[str, object]
ProgramData: TypeAlias = dict[str, object]
ProgramOptionMap: TypeAlias = dict[str, ProgramConfigValue]
ProgramSetup: TypeAlias = ProgramConfigValue | list[ProgramConfigValue]
ProgramArgs: TypeAlias = list[ProgramConfigValue]
ProgramChannel: TypeAlias = Literal["callable", "mcp"]
ProgramChannelSpec: TypeAlias = ProgramChannel | dict[str, ProgramConfigValue]
ProgramChannels: TypeAlias = ProgramChannelSpec | list[ProgramChannelSpec]

ObjectLoader: TypeAlias = str | Callable[..., object | Awaitable[object]]
Objects: TypeAlias = dict[str, ObjectLoader]

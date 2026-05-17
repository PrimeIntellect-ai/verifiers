from collections.abc import (
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from importlib.abc import Traversable
from os import PathLike
from typing import Literal, TypeAlias

from pydantic import BaseModel
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
ConfigFactory: TypeAlias = Callable[[], BaseModel | ConfigMap]
ConfigSource: TypeAlias = BaseModel | ConfigMap | str | ConfigFactory
CallableConfigEntry: TypeAlias = Handler | str | ConfigMap
HandlerList: TypeAlias = Iterable[Handler]

TaskRow: TypeAlias = Mapping[str, object]
TaskRows: TypeAlias = Iterable[TaskRow]
TaskRowsSource: TypeAlias = Callable[[], TaskRows] | TaskRows
TaskSource: TypeAlias = str | TaskRowsSource

PromptMessage: TypeAlias = Message | ConfigMap
PromptInput: TypeAlias = str | Sequence[PromptMessage]
Transcript: TypeAlias = Sequence[PromptMessage]
TranscriptData: TypeAlias = list[PromptMessage]
ToolSpec: TypeAlias = Handler | str | ConfigMap
ToolSpecs: TypeAlias = ToolSpec | Sequence[ToolSpec]
ToolsetSpecs: TypeAlias = ToolSpec | Sequence[ToolSpec] | ConfigMap

ModelClient: TypeAlias = Client | ClientConfig

ProgramScalar: TypeAlias = str | int | float | bool | None
ProgramSource: TypeAlias = PathLike[str] | Traversable
ProgramValue: TypeAlias = ProgramScalar | Handler | ConfigMap | ProgramSource
ProgramCommand: TypeAlias = str | Sequence[ProgramValue]
ProgramMap: TypeAlias = Mapping[str, object]
ProgramData: TypeAlias = dict[str, object]
ProgramOptionMap: TypeAlias = Mapping[str, ProgramValue]
ProgramSetup: TypeAlias = ProgramValue | Sequence[ProgramValue]
ProgramChannel: TypeAlias = Literal["callable", "mcp"]
ProgramChannelSpec: TypeAlias = ProgramChannel | ConfigMap
ProgramChannels: TypeAlias = ProgramChannelSpec | list[ProgramChannelSpec]

Objects: TypeAlias = Mapping[str, object | Callable[[], object | Awaitable[object]]]

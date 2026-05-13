from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import TypeAlias

from pydantic import BaseModel
from verifiers.clients import Client
from verifiers.types import ClientConfig, Message

Handler: TypeAlias = Callable[..., object]
ConfigMap: TypeAlias = Mapping[str, object]
ConfigData: TypeAlias = dict[str, object]
ConfigFactory: TypeAlias = Callable[[], BaseModel | ConfigMap]
ConfigSource: TypeAlias = BaseModel | ConfigMap | str | ConfigFactory
CallableConfigEntry: TypeAlias = Handler | str | ConfigMap
HandlerList: TypeAlias = Iterable[Handler]

TaskRow: TypeAlias = Mapping[str, object]
TaskRows: TypeAlias = Iterable[TaskRow]
TaskRowsSource: TypeAlias = Callable[[], TaskRows] | TaskRows
TaskSource: TypeAlias = str | TaskRowsSource

PromptMessage: TypeAlias = Message | Mapping[str, object]
PromptInput: TypeAlias = str | Sequence[PromptMessage]
ToolSpec: TypeAlias = Handler | str | ConfigMap
ToolSpecs: TypeAlias = ToolSpec | Sequence[ToolSpec]
ToolsetSpecs: TypeAlias = ToolSpec | Sequence[ToolSpec] | ConfigMap

ModelClient: TypeAlias = Client | ClientConfig

ProgramScalar: TypeAlias = str | int | float | bool | None
ProgramValue: TypeAlias = ProgramScalar | Handler | ConfigMap
ProgramCommand: TypeAlias = str | list[ProgramValue]
ProgramMap: TypeAlias = Mapping[str, object]
ProgramData: TypeAlias = dict[str, object]
ProgramOptionMap: TypeAlias = Mapping[str, ProgramValue]
ProgramSetup: TypeAlias = ProgramValue | list[ProgramValue]
ProgramTools: TypeAlias = str | ConfigMap | list[str | ConfigMap]

ObjectFactory: TypeAlias = Callable[[], object | Awaitable[object]]
ObjectSpec: TypeAlias = object | ObjectFactory
ObjectSpecs: TypeAlias = Mapping[str, ObjectSpec]
Objects: TypeAlias = dict[str, ObjectSpec]

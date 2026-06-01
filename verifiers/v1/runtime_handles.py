from typing import Literal

from pydantic import Field
from verifiers.types import ClientType

from .config import Config
from .types import ConfigData


class RuntimeHandleConfig(Config):
    runtime_id: str = Field(min_length=1)


class ModelRuntimeHandleConfig(RuntimeHandleConfig):
    client_key: str = Field(min_length=1)
    model: str | None = None
    client_type: ClientType | None = None
    sampling_args: ConfigData | None = None


class TrajectoryRuntimeHandleConfig(RuntimeHandleConfig):
    trajectory_id: str = Field(min_length=1)
    mode: Literal["append"] = "append"
    start: int = 0


class SandboxRuntimeStateConfig(Config):
    id: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    key: str = Field(min_length=1)
    lease_key: tuple[str, str]


class SandboxRuntimeHandleConfig(RuntimeHandleConfig):
    id: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    key: str = Field(min_length=1)
    lease_key: tuple[str, str]


class ToolsRuntimeHandleConfig(RuntimeHandleConfig):
    handle_id: str = Field(min_length=1)
    names: list[str] = Field(default_factory=list)


class ResolvedRuntimeHandlesConfig(Config):
    model: ModelRuntimeHandleConfig | None = None
    endpoint: RuntimeHandleConfig | None = None
    trajectory: TrajectoryRuntimeHandleConfig | None = None
    sandbox: SandboxRuntimeHandleConfig | None = None
    tools: ToolsRuntimeHandleConfig | None = None

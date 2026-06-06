from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .config import Config
from .types import JsonData


class VisibilityConfig(Config):
    show: list[str] | None = None
    hide: list[str] | None = None

    @model_validator(mode="after")
    def validate_visibility(self) -> "VisibilityConfig":
        if self.show is not None and self.hide is not None:
            raise ValueError("Visibility accepts show or hide, not both.")
        return self


class MCPServerSpec(Config):
    command: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    resources: JsonData = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_server(self) -> "MCPServerSpec":
        if bool(self.command) == bool(self.url):
            raise ValueError("MCPServerSpec requires exactly one of command or url.")
        return self


class ToolsetConfig(VisibilityConfig):
    name: str
    server: MCPServerSpec
    scope: Literal["rollout", "env"] = "rollout"


class Toolset(ToolsetConfig):
    pass


Toolsets = list[Toolset]

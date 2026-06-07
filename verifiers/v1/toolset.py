from __future__ import annotations

import inspect
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from pydantic import Field, model_validator

from .config import Config, ConfigSource
from .types import JsonData
from .utils.config_utils import (
    coerce_config,
    config_type_from_class,
    import_config_ref,
    registered_config_type,
    register_config_type,
)


Scope: TypeAlias = Literal["rollout", "env"]


class VisibilityConfig(Config):
    show: list[str] | None = None
    hide: list[str] | None = None

    @model_validator(mode="after")
    def validate_visibility(self) -> "VisibilityConfig":
        if self.show is not None and self.hide is not None:
            raise ValueError("Visibility accepts show or hide, not both.")
        return self


class ServerConfig(VisibilityConfig):
    loader: str
    name: str | None = None
    scope: Scope = "rollout"
    env: dict[str, str] = Field(default_factory=dict)
    resources: JsonData = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_loader(self) -> "ServerConfig":
        if not self.loader:
            raise ValueError("ServerConfig.loader must be non-empty.")
        return self

    def server_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "verifiers.v1.toolset_runner",
            self.model_dump_json(),
        ]

    def server_env(self) -> dict[str, str]:
        resolved = dict(self.env)
        pythonpath = os.pathsep.join(path for path in sys.path if path)
        if pythonpath:
            resolved.setdefault("PYTHONPATH", pythonpath)
        return resolved

    def load(self) -> "Toolset":
        return load_toolset(self.loader, self)

    def resolved_name(self) -> str:
        if self.name is not None:
            return self.name
        return self.load().name

    def tool_bindings(self) -> dict[str, "ToolBinding"]:
        return {
            name: ToolBinding(
                args=dict(spec.args),
                sets=dict(spec.sets),
                extends=dict(spec.extends),
            )
            for name, spec in iter_tool_specs(type(self.load())).items()
        }


class ToolsetConfig(ServerConfig):
    pass


ConfigT = TypeVar("ConfigT", bound=ServerConfig)


class Toolset(Generic[ConfigT]):
    config: ConfigT
    name: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        config_type = config_type_from_class(
            cls,
            inherited=False,
            owner_base=Toolset,
            config_base=ServerConfig,
        )
        if config_type is not None:
            register_config_type(cls, config_type)

    def __init__(self, config: ConfigSource = None):
        config_type = registered_config_type(type(self), ServerConfig)
        self.config = cast(ConfigT, coerce_config(config_type, config))
        self.name = self.config.name or toolset_name(type(self))


class ToolBinding(Config):
    args: dict[str, str] = Field(default_factory=dict)
    sets: dict[str, str] = Field(default_factory=dict)
    extends: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str | None
    args: dict[str, str]
    sets: dict[str, str]
    extends: dict[str, str]


ToolFunc = TypeVar("ToolFunc", bound=Callable[..., object])


def tool(
    func: ToolFunc | None = None,
    *,
    args: Mapping[str, str] | None = None,
    sets: Mapping[str, str] | None = None,
    extends: Mapping[str, str] | None = None,
    name: str | None = None,
) -> ToolFunc | Callable[[ToolFunc], ToolFunc]:
    def decorate(item: ToolFunc) -> ToolFunc:
        setattr(
            item,
            "__vf_tool__",
            ToolSpec(
                name=name,
                args=dict(args or {}),
                sets=dict(sets or {}),
                extends=dict(extends or {}),
            ),
        )
        return item

    if func is not None:
        return decorate(func)
    return decorate


Toolsets: TypeAlias = list[ToolsetConfig]


def load_toolset(loader: str, config: ServerConfig) -> Toolset:
    obj = import_config_ref(loader)
    if isinstance(obj, type) and issubclass(obj, Toolset):
        return obj(config=config)
    if callable(obj):
        loaded = obj(config)
        if isinstance(loaded, Toolset):
            return loaded
    raise TypeError(f"Toolset loader {loader!r} must be a Toolset class or loader.")


def iter_tool_specs(toolset: type[Toolset]) -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    for _, member in inspect.getmembers(toolset, predicate=callable):
        spec = getattr(member, "__vf_tool__", None)
        if not isinstance(spec, ToolSpec):
            continue
        tool_name = spec.name or getattr(member, "__name__", "")
        if not isinstance(tool_name, str) or not tool_name:
            raise TypeError("Tool names must be non-empty strings.")
        if tool_name in specs:
            raise ValueError(f"Tool {tool_name!r} is defined twice.")
        specs[tool_name] = spec
    return specs


def toolset_name(toolset: type[Toolset]) -> str:
    name = toolset.__name__
    if name.endswith("Toolset") and len(name) > len("Toolset"):
        name = name[: -len("Toolset")]
    return camel_to_snake(name or "toolset")


def camel_to_snake(value: str) -> str:
    result: list[str] = []
    for index, char in enumerate(value):
        if char.isupper() and index > 0 and not value[index - 1].isupper():
            result.append("_")
        result.append(char.lower())
    return "".join(result).replace("-", "_")


def build_fastmcp(toolset: Toolset):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(toolset.name)
    for method_name, method in inspect.getmembers(toolset, predicate=callable):
        spec = getattr(getattr(type(toolset), method_name, None), "__vf_tool__", None)
        if not isinstance(spec, ToolSpec):
            continue
        tool_name = spec.name or method_name
        mcp.tool(name=tool_name)(method)
    return mcp


def run_toolset(config_json: str) -> None:
    data = json.loads(config_json)
    if not isinstance(data, dict):
        raise TypeError("Toolset runner config must decode to an object.")
    loader = data.get("loader")
    if not isinstance(loader, str) or not loader:
        raise TypeError("Toolset runner config requires a loader.")
    config_cls = config_type_for_loader(loader)
    config = config_cls.model_validate(data)
    toolset = load_toolset(loader, config)
    build_fastmcp(toolset).run(transport="stdio")


def config_type_for_loader(loader: str) -> type[ServerConfig]:
    obj = import_config_ref(loader)
    if isinstance(obj, type) and issubclass(obj, Toolset):
        return registered_config_type(obj, ServerConfig)
    return ServerConfig

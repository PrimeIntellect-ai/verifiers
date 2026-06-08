from __future__ import annotations

import inspect
import importlib.util
import asyncio
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from pydantic import BaseModel, Field, model_validator

from .config import Config, ConfigSource
from .runtime import RuntimeConfig, SubprocessRuntimeConfig
from .types import JsonData
from .utils.config_utils import (
    coerce_config,
    config_type_from_class,
    explicit_config_data,
    import_config_ref,
    registered_config_type,
    register_config_type,
)


Scope: TypeAlias = Literal["rollout", "env"]
ServerPlacement: TypeAlias = Literal["dedicated", "colocated", "remote"]
ConfigT = TypeVar("ConfigT", bound="ServerConfig")


class VisibilityConfig(Config):
    show: list[str] | None = None
    hide: list[str] | None = None

    @model_validator(mode="after")
    def validate_visibility(self) -> "VisibilityConfig":
        if self.show is not None and self.hide is not None:
            raise ValueError("Visibility accepts show or hide, not both.")
        return self


class ServerConfig(VisibilityConfig):
    source: str | None = None
    enabled: bool = True
    scope: Scope = "rollout"
    placement: ServerPlacement = "dedicated"
    runtime: RuntimeConfig | None = Field(default_factory=SubprocessRuntimeConfig)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    resources: JsonData = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_server(self) -> "ServerConfig":
        if self.scope == "env" and self.placement == "colocated":
            raise ValueError("env-scope servers cannot use colocated placement.")
        if self.placement == "remote":
            if not self.url:
                raise ValueError("Remote server configs require url.")
            return self
        if self.url is not None:
            raise ValueError("Only remote server configs may set url.")
        return self

    def default_server_ref(self) -> str:
        config_type = type(self)
        module_name = config_type.__module__
        if module_name.startswith("verifiers.v1."):
            raise ValueError(
                f"{config_type.__name__} cannot infer a toolset implementation from "
                "the framework package."
            )
        package = ServerConfig.config_package(module_name)
        class_name = config_type.__name__
        if module_name.endswith(".config"):
            if class_name.endswith("ToolsetConfig"):
                impl_name = f"{class_name.removesuffix('Config')}"
            else:
                basename = package.rsplit(".", 1)[-1]
                impl_name = f"{ServerConfig.snake_to_pascal(basename)}Toolset"
            return f"{package}.toolset:{impl_name}"
        if class_name.endswith("Config"):
            impl_name = f"{class_name.removesuffix('Config')}"
        else:
            impl_name = f"{class_name}Toolset"
        return f"{module_name}:{impl_name}"

    def implementation_ref(self) -> str:
        if self.placement == "remote":
            raise ValueError("Remote server configs do not have an implementation ref.")
        ref = type(self).resolve_ref(self.default_server_ref(), type(self))
        if not ref:
            raise ValueError("Server implementation ref must be non-empty.")
        return ref

    def load(self) -> "Toolset":
        return Toolset.load_ref(self.implementation_ref(), self)

    @classmethod
    def resolve_config(
        cls,
        name: str,
        value: object,
        *,
        default: "ServerConfig | None",
        base_type: type[ConfigT],
    ) -> ConfigT:
        if isinstance(value, base_type):
            if default is not None:
                cls.validate_default_source(
                    name,
                    source=value.source,
                    default=default,
                    base_type=base_type,
                )
            return value
        if value is not None and not isinstance(value, BaseModel | Mapping):
            raise TypeError("Server config values must be mappings or config objects.")
        data = explicit_config_data(cast(ConfigSource, value))
        source = data.get("source")
        if default is not None:
            cls.validate_default_source(
                name,
                source=source,
                default=default,
                base_type=base_type,
            )
            config_type = type(default)
            merged = default.model_dump(mode="json", exclude_none=True)
            merged.update(data)
            return cast(ConfigT, config_type.model_validate(merged))
        if data.get("enabled") is False and source is None:
            raise ValueError(
                f"Server {name!r} is not declared by the taskset and cannot be "
                "disabled."
            )
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"Server {name!r} is not declared by the taskset; set source to a "
                f"{base_type.__name__} class."
            )
        config_type = cls.source_type(source, base_type)
        return cast(ConfigT, config_type.model_validate(data))

    @classmethod
    def validate_default_source(
        cls,
        name: str,
        *,
        source: object,
        default: "ServerConfig",
        base_type: type[ConfigT],
    ) -> None:
        if source is None:
            return
        if not isinstance(source, str) or not source:
            raise TypeError(f"Server {name!r} source must be a non-empty string.")
        config_type = cls.source_type(source, base_type)
        if config_type is not type(default):
            raise TypeError(
                f"Server {name!r} source must match taskset-defined "
                f"{type(default).__name__}; got {config_type.__name__}."
            )

    @classmethod
    def source_type(cls, source: str, base_type: type[ConfigT]) -> type[ConfigT]:
        obj = import_config_ref(source)
        if isinstance(obj, type) and issubclass(obj, base_type):
            return obj
        raise TypeError(
            f"Server source {source!r} must point to a {base_type.__name__}."
        )

    @staticmethod
    def config_package(module_name: str) -> str:
        if module_name.endswith(".config"):
            return module_name.rsplit(".", 1)[0]
        return module_name.rsplit(".", 1)[0]

    @staticmethod
    def resolve_ref(ref: str, config_type: type["ServerConfig"]) -> str:
        module_name, separator, attr_path = ref.partition(":")
        if not separator:
            raise ValueError(f"Server ref {ref!r} must use 'module:object'.")
        if module_name.startswith("."):
            package = ServerConfig.config_package(config_type.__module__)
            module_name = importlib.util.resolve_name(module_name, package)
        return f"{module_name}:{attr_path}"

    @staticmethod
    def snake_to_pascal(value: str) -> str:
        return "".join(part[:1].upper() + part[1:] for part in value.split("_") if part)


class ToolsetConfig(ServerConfig):
    pass


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
        self.name = type(self).default_name()
        self.resources: dict[str, object] = {}

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def load_resources(self) -> None:
        for method_name, method in inspect.getmembers(self, predicate=callable):
            spec = getattr(
                getattr(type(self), method_name, None), "__vf_resource__", None
            )
            if not isinstance(spec, ResourceSpec):
                continue
            name = spec.name or method_name
            value = method()
            if inspect.isawaitable(value):
                value = asyncio.run(cast(Coroutine[object, object, object], value))
            self.resources[name] = value

    @staticmethod
    def load_ref(server: str, config: ServerConfig) -> "Toolset":
        obj = import_config_ref(server)
        if isinstance(obj, type) and issubclass(obj, Toolset):
            return obj(config=config)
        if callable(obj):
            loader = cast(Callable[[ServerConfig], object], obj)
            loaded = loader(config)
            if isinstance(loaded, Toolset):
                return loaded
        raise TypeError(f"Server {server!r} must be a Toolset class or loader.")

    @classmethod
    def tool_specs(cls) -> dict[str, "ToolSpec"]:
        specs: dict[str, ToolSpec] = {}
        for _, member in inspect.getmembers(cls, predicate=callable):
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

    @classmethod
    def default_name(cls) -> str:
        name = cls.__name__
        if name.endswith("Toolset") and len(name) > len("Toolset"):
            name = name[: -len("Toolset")]
        return cls.name_from_class(name or "toolset")

    @staticmethod
    def name_from_class(value: str) -> str:
        result: list[str] = []
        for index, char in enumerate(value):
            if char.isupper() and index > 0 and not value[index - 1].isupper():
                result.append("_")
            result.append(char.lower())
        return "".join(result).replace("-", "_")


class ToolBinding(Config):
    args: dict[str, str] = Field(default_factory=dict)
    sets: dict[str, str] = Field(default_factory=dict)
    extends: dict[str, str] = Field(default_factory=dict)
    hidden: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str | None
    args: dict[str, str]
    sets: dict[str, str]
    extends: dict[str, str]
    hidden: bool


@dataclass(frozen=True)
class ResourceSpec:
    name: str | None


ToolFunc = TypeVar("ToolFunc", bound=Callable[..., object])


def tool(
    func: ToolFunc | None = None,
    *,
    args: Mapping[str, str] | None = None,
    sets: Mapping[str, str] | None = None,
    extends: Mapping[str, str] | None = None,
    name: str | None = None,
    hidden: bool = False,
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
                hidden=hidden,
            ),
        )
        return item

    if func is not None:
        return decorate(func)
    return decorate


ResourceFunc = TypeVar("ResourceFunc", bound=Callable[..., object])


def resource(
    func: ResourceFunc | None = None,
    *,
    name: str | None = None,
) -> ResourceFunc | Callable[[ResourceFunc], ResourceFunc]:
    def decorate(item: ResourceFunc) -> ResourceFunc:
        setattr(item, "__vf_resource__", ResourceSpec(name=name))
        return item

    if func is not None:
        return decorate(func)
    return decorate


ToolsetConfigs: TypeAlias = dict[str, ToolsetConfig]

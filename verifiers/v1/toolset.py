from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

from pydantic import field_validator, model_validator

from .config import (
    CallableEntry,
    Config,
    JsonMap,
    resolve_config_object,
)
from .sandbox import SandboxConfig, sandbox_config_mapping
from .utils.binding_utils import BindingMap, normalize_binding_map
from .utils.binding_utils import normalize_object_map
from .utils.config_callable_utils import config_callables
from .types import ConfigMap, Handler, Objects, ToolSpec
from .utils.toolset_utils import (
    collect_toolsets as collect_toolsets,
    flatten_toolsets as flatten_toolsets,
    iter_toolsets as iter_toolsets,
    normalize_toolset as normalize_toolset,
    normalize_toolset_collection as normalize_toolset_collection,
    normalize_toolset_result as normalize_toolset_result,
    optional_string as optional_string,
    string_items as string_items,
    tool_item as tool_item,
    tool_items as tool_items,
    tool_name as tool_name,
    toolset_config_mapping as toolset_config_mapping,
)

ToolsetCallableEntry: TypeAlias = CallableEntry | Handler


class MCPToolConfig(Config):
    command: str
    args: list[str] = []
    env: dict[str, str] | None = None
    cwd: str | None = None

    @field_validator("args", mode="before")
    @classmethod
    def validate_args(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value


class ToolsetConfig(Config):
    tools: str | JsonMap | list[str | JsonMap] | None = []
    show: list[str] | None = None
    hide: list[str] | None = None
    bindings: BindingMap = {}
    objects: dict[str, str] = {}
    write: bool = False
    scope: Literal["rollout", "group", "global"] | None = None
    sandbox: SandboxConfig | Literal["program"] | None = None
    stops: list[CallableEntry] = []
    setups: list[CallableEntry] = []
    updates: list[CallableEntry] = []
    cleanups: list[CallableEntry] = []
    teardowns: list[CallableEntry] = []

    @field_validator("show", "hide", mode="before")
    @classmethod
    def validate_visibility_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> BindingMap:
        return normalize_binding_map(value, "toolset.bindings")

    @model_validator(mode="after")
    def validate_visibility(self) -> "ToolsetConfig":
        if self.show is not None and self.hide is not None:
            raise ValueError("Toolset accepts show or hide, not both.")
        return self


@dataclass(frozen=True)
class Toolset:
    # Tool surface.
    tools: "tuple[ToolEntry, ...]" = ()
    show: tuple[str, ...] | None = None
    hide: tuple[str, ...] | None = None
    # Local dependencies and runtime policy.
    bindings: BindingMap = field(default_factory=dict)
    objects: Objects = field(default_factory=dict)
    write: bool = False
    scope: str | None = None
    sandbox: ConfigMap | SandboxConfig | str | None = None
    # Lifecycle collections.
    stops: tuple[Handler, ...] = ()
    setups: tuple[Handler, ...] = ()
    updates: tuple[Handler, ...] = ()
    cleanups: tuple[Handler, ...] = ()
    teardowns: tuple[Handler, ...] = ()
    # Config.
    config: ToolsetConfig | None = None

    def __init__(
        self,
        # Tool surface.
        tools: "ToolEntries | None" = (),
        show: Iterable[str] | None = None,
        hide: Iterable[str] | None = None,
        # Local dependencies and runtime policy.
        bindings: BindingMap | None = None,
        objects: Objects | None = None,
        write: bool | None = None,
        scope: str | None = None,
        sandbox: ConfigMap | SandboxConfig | str | None = None,
        # Lifecycle collections.
        stops: Iterable[ToolsetCallableEntry] = (),
        setups: Iterable[ToolsetCallableEntry] = (),
        updates: Iterable[ToolsetCallableEntry] = (),
        cleanups: Iterable[ToolsetCallableEntry] = (),
        teardowns: Iterable[ToolsetCallableEntry] = (),
        # Config.
        config: ToolsetConfig | None = None,
    ):
        config_map = toolset_config_mapping(config)
        tool_values = tool_items(tools)
        config_bindings: BindingMap = {}
        config_objects: Objects = {}
        if config_map:
            tool_values.extend(tool_items(config_map.get("tools")))
            show = show if show is not None else string_items(config_map.get("show"))
            hide = hide if hide is not None else string_items(config_map.get("hide"))
            config_bindings = normalize_binding_map(
                config_map.get("bindings"), "Toolset bindings"
            )
            config_objects = cast(
                Objects,
                {
                    str(key): resolve_config_object(item)
                    for key, item in normalize_object_map(
                        config_map.get("objects"), "Toolset objects"
                    ).items()
                },
            )
            if "write" in config_map and write is None:
                write_value = config_map["write"]
                if not isinstance(write_value, bool):
                    raise TypeError("Toolset write must be a boolean.")
                write = write_value
            scope = (
                scope if scope is not None else optional_string(config_map.get("scope"))
            )
            sandbox = (
                sandbox
                if sandbox is not None
                else cast(ConfigMap | str | None, config_map.get("sandbox"))
            )
            stops = [*stops, *config_callables(config_map.get("stops"), "stop")]
            setups = [
                *setups,
                *config_callables(config_map.get("setups"), "setup"),
            ]
            updates = [
                *updates,
                *config_callables(config_map.get("updates"), "update"),
            ]
            cleanups = [
                *cleanups,
                *config_callables(config_map.get("cleanups"), "cleanup"),
            ]
            teardowns = [
                *teardowns,
                *config_callables(config_map.get("teardowns"), "teardown"),
            ]
        if show is not None and hide is not None:
            raise ValueError("Toolset accepts show or hide, not both.")
        if write is not None and not isinstance(write, bool):
            raise TypeError("Toolset write must be a boolean.")
        object.__setattr__(self, "tools", tuple(tool_values))
        object.__setattr__(self, "show", tuple(show) if show is not None else None)
        object.__setattr__(self, "hide", tuple(hide) if hide is not None else None)
        object.__setattr__(
            self,
            "bindings",
            {
                **config_bindings,
                **normalize_binding_map(bindings, "Toolset bindings"),
            },
        )
        object.__setattr__(
            self,
            "objects",
            {
                **config_objects,
                **cast(
                    Objects,
                    {
                        str(key): resolve_config_object(item)
                        for key, item in normalize_object_map(
                            objects, "Toolset objects"
                        ).items()
                    },
                ),
            },
        )
        object.__setattr__(self, "write", bool(write))
        if scope is not None and scope not in {"rollout", "group", "global"}:
            raise ValueError("Toolset scope must be 'rollout', 'group', or 'global'.")
        object.__setattr__(self, "scope", scope)
        if isinstance(sandbox, str) and sandbox != "program":
            raise ValueError("Toolset sandbox string must be 'program'.")
        sandbox_value: ConfigMap | str | None
        if isinstance(sandbox, str):
            sandbox_value = sandbox
        else:
            sandbox_value = sandbox_config_mapping(sandbox)
        if isinstance(sandbox_value, Mapping):
            prefer = cast(ConfigMap, sandbox_value).get("prefer")
            if prefer is not None and prefer != "program":
                raise ValueError("Toolset sandbox.prefer must be 'program'.")
        object.__setattr__(self, "sandbox", sandbox_value)
        object.__setattr__(self, "stops", tuple(config_callables(stops, "stop")))
        object.__setattr__(self, "setups", tuple(config_callables(setups, "setup")))
        object.__setattr__(self, "updates", tuple(config_callables(updates, "update")))
        object.__setattr__(
            self, "cleanups", tuple(config_callables(cleanups, "cleanup"))
        )
        object.__setattr__(
            self, "teardowns", tuple(config_callables(teardowns, "teardown"))
        )
        object.__setattr__(self, "config", config)

    def add_tool(self, tool: "ToolEntry") -> None:
        tool_value = tool_item(tool)
        object.__setattr__(self, "tools", (*self.tools, tool_value))


ToolsetItem: TypeAlias = Toolset | ToolSpec
ToolsetCollection: TypeAlias = (
    ToolsetItem | Iterable[ToolsetItem] | dict[str, ToolsetItem | ConfigMap]
)
Toolsets: TypeAlias = ToolsetCollection | None


@dataclass(frozen=True)
class MCPTool:
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None

    def __init__(
        self,
        command: str,
        args: Iterable[str] = (),
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ):
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "env", dict(env) if env is not None else None)
        object.__setattr__(self, "cwd", cwd)


ToolEntry: TypeAlias = Handler | str | ConfigMap | Toolset | MCPTool | MCPToolConfig
ToolEntries: TypeAlias = ToolEntry | Iterable[ToolEntry]

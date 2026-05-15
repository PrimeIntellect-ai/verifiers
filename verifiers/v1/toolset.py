import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from pydantic import BaseModel

from .config import (
    CallableConfigEntry,
    MCPToolConfig,
    SandboxConfig,
    ToolsetConfig,
    config_callables,
    resolve_config_object,
    sandbox_config_mapping,
)
from .utils.binding_utils import BindingMap, normalize_binding_map
from .utils.binding_utils import normalize_object_map
from .types import ConfigMap, Handler, Objects, ToolSpec

ToolsetCallableEntry: TypeAlias = CallableConfigEntry | Handler


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
            config_objects = {
                str(key): resolve_config_object(item)
                for key, item in normalize_object_map(
                    config_map.get("objects"), "Toolset objects"
                ).items()
            }
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
            {**config_objects, **normalize_object_map(objects, "Toolset objects")},
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


ToolsetItem: TypeAlias = Toolset | ToolSpec
ToolsetCollection: TypeAlias = (
    ToolsetItem | Iterable[ToolsetItem] | dict[str, ToolsetItem | ConfigMap]
)


def flatten_toolsets(
    toolsets: "Iterable[ToolEntry]", apply_visibility: bool = False
) -> "list[ToolEntry]":
    flat: list[ToolEntry] = []
    for item in toolsets:
        if isinstance(item, Toolset):
            tools = flatten_toolsets(item.tools, apply_visibility)
            if apply_visibility and item.show is not None:
                show = set(item.show)
                tools = [tool for tool in tools if tool_name(tool) in show]
            if apply_visibility and item.hide is not None:
                hide = set(item.hide)
                tools = [tool for tool in tools if tool_name(tool) not in hide]
            flat.extend(tools)
        else:
            flat.append(item)
    return flat


def iter_toolsets(toolsets: "Iterable[ToolEntry]") -> list[Toolset]:
    groups: list[Toolset] = []
    for item in toolsets:
        if isinstance(item, Toolset):
            groups.append(item)
            groups.extend(iter_toolsets(item.tools))
    return groups


def normalize_toolsets(toolsets: "Iterable[ToolEntry]") -> list[Toolset]:
    return [normalize_toolset(toolset) for toolset in toolsets]


def merge_toolsets(
    values: object,
    config: object,
) -> tuple[list[Toolset], dict[str, Toolset]]:
    value_toolsets, value_named = normalize_toolset_collection(values)
    config_toolsets, config_named = normalize_toolset_collection(config)
    duplicate = set(value_named) & set(config_named)
    if duplicate:
        raise ValueError(f"Toolsets are defined twice: {sorted(duplicate)}.")
    return [*value_toolsets, *config_toolsets], {**value_named, **config_named}


def normalize_toolset_collection(
    value: object,
) -> tuple[list[Toolset], dict[str, Toolset]]:
    if value is None:
        return [], {}
    if isinstance(value, Mapping):
        named: dict[str, Toolset] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Toolset names must be strings.")
            if key in named:
                raise ValueError(f"Toolset {key!r} is defined twice.")
            named[key] = named_toolset_from_config(key, item)
        return list(named.values()), named
    if isinstance(value, str):
        return [normalize_toolset(value)], {}
    if not isinstance(value, Iterable):
        return [normalize_toolset(value)], {}
    return normalize_toolsets(cast(Iterable[ToolEntry], value)), {}


def named_toolset_from_config(name: str, value: object) -> Toolset:
    value = resolve_config_object(value)
    if isinstance(value, Toolset):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        spec = cast(ConfigMap, value)
        if "fn" in spec:
            return toolset_from_factory(name, spec)
        return toolset_from_mapping(spec)
    if callable(value):
        return call_toolset_factory(name, cast(Handler, value), {})
    return normalize_toolset(value)


def toolset_from_factory(name: str, spec: ConfigMap) -> Toolset:
    fn = resolve_config_object(spec.get("fn"))
    if not callable(fn):
        raise TypeError(f"Toolset {name!r} requires callable fn.")
    kwargs = {key: value for key, value in spec.items() if key != "fn"}
    return call_toolset_factory(name, cast(Handler, fn), kwargs)


def call_toolset_factory(name: str, fn: Handler, kwargs: ConfigMap) -> Toolset:
    result = fn(**kwargs)
    if inspect.isawaitable(result):
        raise TypeError(f"Toolset {name!r} fn must be synchronous.")
    toolsets = normalize_toolset_result(result)
    if len(toolsets) != 1:
        raise ValueError(f"Toolset {name!r} fn must return exactly one Toolset.")
    return toolsets[0]


def normalize_toolset_result(value: object) -> list[Toolset]:
    value = resolve_config_object(value)
    if value is None:
        return []
    if isinstance(value, Toolset | Mapping | str):
        return [normalize_toolset(value)]
    if not isinstance(value, Iterable):
        return [normalize_toolset(value)]
    return normalize_toolsets(cast(Iterable[ToolEntry], value))


def normalize_toolset(value: object) -> Toolset:
    value = resolve_config_object(value)
    if isinstance(value, Toolset):
        return value
    if isinstance(value, Mapping):
        return toolset_from_mapping(cast(ConfigMap, value))
    return Toolset(tools=[cast(ToolEntry, value)])


def toolset_from_mapping(spec: ConfigMap) -> Toolset:
    return Toolset(
        tools=cast(ToolEntries | None, spec.get("tools", ())),
        show=cast(Iterable[str] | None, spec.get("show")),
        hide=cast(Iterable[str] | None, spec.get("hide")),
        bindings=cast(BindingMap | None, spec.get("bindings")),
        objects=cast(Objects | None, spec.get("objects")),
        write=cast(bool | None, spec.get("write")),
        scope=cast(str | None, spec.get("scope")),
        sandbox=cast(ConfigMap | SandboxConfig | str | None, spec.get("sandbox")),
        stops=cast(Iterable[ToolsetCallableEntry], spec.get("stops") or ()),
        setups=cast(Iterable[ToolsetCallableEntry], spec.get("setups") or ()),
        updates=cast(Iterable[ToolsetCallableEntry], spec.get("updates") or ()),
        cleanups=cast(Iterable[ToolsetCallableEntry], spec.get("cleanups") or ()),
        teardowns=cast(Iterable[ToolsetCallableEntry], spec.get("teardowns") or ()),
    )


def tool_items(value: object) -> "list[ToolEntry]":
    if value is None:
        return []
    if isinstance(value, str) or isinstance(value, Mapping):
        return [tool_item(value)]
    if not isinstance(value, Iterable):
        return [tool_item(value)]
    return [tool_item(item) for item in value]


def tool_item(value: object) -> "ToolEntry":
    value = resolve_config_object(value)
    if isinstance(value, Toolset | MCPTool):
        return value
    if isinstance(value, MCPToolConfig):
        return MCPTool.from_mapping(value.model_dump(exclude_none=True))
    if isinstance(value, Mapping):
        if "command" in value:
            return MCPTool.from_mapping(cast(ConfigMap, value))
        raise TypeError("Tool mapping specs require command.")
    if not callable(value):
        raise TypeError("Tool entries must be callables, Toolsets, or MCP tool specs.")
    return cast(Handler, value)


def toolset_config_mapping(config: ToolsetConfig | None) -> ConfigMap:
    if config is None:
        return {}
    return ToolsetConfig.from_config(config).model_dump(exclude_none=True)


def string_items(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable):
        raise TypeError("Toolset visibility fields must be strings or lists.")
    return [str(item) for item in value]


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Toolset scope must be a string.")
    return value


def tool_name(tool: object) -> str:
    name = getattr(tool, "__name__", None) or getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Tools require a stable __name__ or name.")
    return name


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

    @classmethod
    def from_mapping(cls, spec: ConfigMap) -> "MCPTool":
        config = MCPToolConfig.from_config(spec)
        return cls(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )


ToolEntry: TypeAlias = Handler | str | ConfigMap | Toolset | MCPTool | MCPToolConfig
ToolEntries: TypeAlias = ToolEntry | Iterable[ToolEntry]

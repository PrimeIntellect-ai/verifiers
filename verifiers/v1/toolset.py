from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast

from .config import config_callables, resolve_config_object, string_mapping


@dataclass(frozen=True)
class Toolset:
    # Tool surface.
    tools: tuple[object, ...] = ()
    show: tuple[str, ...] | None = None
    hide: tuple[str, ...] | None = None
    # Local dependencies and runtime policy.
    bindings: Mapping[str, object] = field(default_factory=dict)
    objects: Mapping[str, object] = field(default_factory=dict)
    write: bool = False
    scope: str | None = None
    sandbox: Mapping[str, object] | str | None = None
    # Lifecycle collections.
    stops: tuple[object, ...] = ()
    setups: tuple[object, ...] = ()
    updates: tuple[object, ...] = ()
    cleanups: tuple[object, ...] = ()
    teardowns: tuple[object, ...] = ()
    # Config.
    config: object | None = None

    def __init__(
        self,
        # Tool surface.
        tools: Iterable[object] = (),
        show: Iterable[str] | None = None,
        hide: Iterable[str] | None = None,
        # Local dependencies and runtime policy.
        bindings: Mapping[str, object] | None = None,
        objects: Mapping[str, object] | None = None,
        write: bool | None = None,
        scope: str | None = None,
        sandbox: Mapping[str, object] | str | None = None,
        # Lifecycle collections.
        stops: Iterable[object] = (),
        setups: Iterable[object] = (),
        updates: Iterable[object] = (),
        cleanups: Iterable[object] = (),
        teardowns: Iterable[object] = (),
        # Config.
        config: object | None = None,
    ):
        config_map = toolset_config_mapping(config)
        if config_map:
            tools = [*tools, *tool_items(config_map.get("tools"))]
            show = show if show is not None else string_items(config_map.get("show"))
            hide = hide if hide is not None else string_items(config_map.get("hide"))
            config_bindings = config_map.get("bindings") or {}
            if not isinstance(config_bindings, Mapping):
                raise TypeError("Toolset bindings must be a mapping.")
            bindings = {
                **string_mapping(cast(Mapping[object, object], config_bindings)),
                **dict(bindings or {}),
            }
            config_objects = config_map.get("objects") or {}
            if not isinstance(config_objects, Mapping):
                raise TypeError("Toolset objects must be a mapping.")
            objects = {
                **{
                    str(key): resolve_config_object(item)
                    for key, item in config_objects.items()
                },
                **dict(objects or {}),
            }
            if "write" in config_map and write is None:
                write_value = config_map["write"]
                if not isinstance(write_value, bool):
                    raise TypeError("Toolset write must be a boolean.")
                write = write_value
            scope = (
                scope if scope is not None else optional_string(config_map.get("scope"))
            )
            config_sandbox = config_map.get("sandbox")
            if config_sandbox is not None and not isinstance(
                config_sandbox, Mapping | str
            ):
                raise TypeError("Toolset sandbox must be a mapping or 'program'.")
            sandbox = (
                sandbox
                if sandbox is not None
                else cast(Mapping[str, object] | str | None, config_sandbox)
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
        object.__setattr__(self, "tools", tuple(tool_item(tool) for tool in tools))
        object.__setattr__(self, "show", tuple(show) if show is not None else None)
        object.__setattr__(self, "hide", tuple(hide) if hide is not None else None)
        object.__setattr__(self, "bindings", dict(bindings or {}))
        object.__setattr__(self, "objects", dict(objects or {}))
        object.__setattr__(self, "write", bool(write))
        if scope is not None and scope not in {"rollout", "group", "global"}:
            raise ValueError("Toolset scope must be 'rollout', 'group', or 'global'.")
        object.__setattr__(self, "scope", scope)
        if isinstance(sandbox, str) and sandbox != "program":
            raise ValueError("Toolset sandbox string must be 'program'.")
        if isinstance(sandbox, Mapping):
            prefer = cast(Mapping[str, object], sandbox).get("prefer")
            if prefer is not None and prefer != "program":
                raise ValueError("Toolset sandbox.prefer must be 'program'.")
        object.__setattr__(self, "sandbox", sandbox)
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


def flatten_toolsets(
    toolsets: Iterable[object], apply_visibility: bool = False
) -> list[object]:
    flat: list[object] = []
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


def iter_toolsets(toolsets: Iterable[object]) -> list[Toolset]:
    groups: list[Toolset] = []
    for item in toolsets:
        if isinstance(item, Toolset):
            groups.append(item)
            groups.extend(iter_toolsets(item.tools))
    return groups


def normalize_toolsets(toolsets: Iterable[object]) -> list[Toolset]:
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
    return normalize_toolsets(value), {}


def named_toolset_from_config(name: str, value: object) -> Toolset:
    value = resolve_config_object(value)
    if isinstance(value, Toolset):
        return value
    if isinstance(value, Mapping):
        spec = cast(Mapping[str, object], value)
        if "fn" in spec:
            return toolset_from_factory(name, spec)
        return Toolset(config=spec)
    if callable(value):
        return call_toolset_factory(name, cast(Callable[..., object], value), {})
    return normalize_toolset(value)


def toolset_from_factory(name: str, spec: Mapping[str, object]) -> Toolset:
    fn = resolve_config_object(spec.get("fn"))
    if not callable(fn):
        raise TypeError(f"Toolset {name!r} requires callable fn.")
    kwargs = {key: value for key, value in spec.items() if key != "fn"}
    return call_toolset_factory(name, cast(Callable[..., object], fn), kwargs)


def call_toolset_factory(
    name: str,
    fn: Callable[..., object],
    kwargs: Mapping[str, object],
) -> Toolset:
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
    return normalize_toolsets(value)


def normalize_toolset(value: object) -> Toolset:
    value = resolve_config_object(value)
    if isinstance(value, Toolset):
        return value
    if isinstance(value, Mapping):
        return toolset_from_mapping(cast(Mapping[str, object], value))
    return Toolset(tools=[value])


def toolset_from_mapping(spec: Mapping[str, object]) -> Toolset:
    return Toolset(config=spec)


def tool_items(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str) or isinstance(value, Mapping):
        return [tool_item(value)]
    if not isinstance(value, Iterable):
        return [tool_item(value)]
    return [tool_item(item) for item in value]


def tool_item(value: object) -> object:
    value = resolve_config_object(value)
    if isinstance(value, Mapping):
        if "command" in value:
            return MCPTool.from_mapping(cast(Mapping[str, object], value))
        raise TypeError("Tool mapping specs require command.")
    return value


def toolset_config_mapping(config: object | None) -> Mapping[str, object]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        return {}
    spec = cast(Mapping[str, object], config)
    unknown_keys = set(spec) - {
        "tools",
        "show",
        "hide",
        "bindings",
        "objects",
        "write",
        "scope",
        "sandbox",
        "stops",
        "setups",
        "updates",
        "cleanups",
        "teardowns",
    }
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Toolset config has unknown keys: {unknown}.")
    return spec


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
    env: Mapping[str, str] | None = None
    cwd: str | None = None

    def __init__(
        self,
        command: str,
        args: Iterable[str] = (),
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ):
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "env", dict(env) if env is not None else None)
        object.__setattr__(self, "cwd", cwd)

    @classmethod
    def from_mapping(cls, spec: Mapping[str, object]) -> MCPTool:
        unknown_keys = set(spec) - {"command", "args", "env", "cwd"}
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"MCPTool config has unknown keys: {unknown}.")
        command = spec.get("command")
        if not isinstance(command, str):
            raise TypeError("MCPTool command must be a string.")
        args = spec.get("args") or ()
        if isinstance(args, str):
            args = [args]
        if not isinstance(args, Iterable):
            raise TypeError("MCPTool args must be a list of strings.")
        env = spec.get("env")
        if env is not None and not isinstance(env, Mapping):
            raise TypeError("MCPTool env must be a mapping.")
        if isinstance(env, Mapping):
            for key, value in env.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise TypeError("MCPTool env keys and values must be strings.")
        cwd = spec.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise TypeError("MCPTool cwd must be a string.")
        return cls(
            command=command,
            args=[str(arg) for arg in args],
            env=cast(Mapping[str, str] | None, env),
            cwd=cwd,
        )

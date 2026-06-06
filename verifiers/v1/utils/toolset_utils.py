import inspect
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel
from verifiers.types import Tool

from ..types import ConfigData, Handler
from .config_utils import coerce_config, resolved_config_data, resolve_config_object

if TYPE_CHECKING:
    from ..toolset import ToolEntry, Toolset


def flatten_toolsets(
    toolsets: Iterable["ToolEntry"], apply_visibility: bool = False
) -> list["ToolEntry"]:
    from ..toolset import Toolset

    flat: list["ToolEntry"] = []
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


def iter_toolsets(toolsets: Iterable["ToolEntry"]) -> list["Toolset"]:
    from ..toolset import Toolset

    groups: list["Toolset"] = []
    for item in toolsets:
        if isinstance(item, Toolset):
            groups.append(item)
            groups.extend(iter_toolsets(item.tools))
    return groups


def normalize_toolsets(toolsets: Iterable["ToolEntry"]) -> list["Toolset"]:
    return [normalize_toolset(toolset) for toolset in toolsets]


def collect_toolsets(
    values: object,
    config: object,
) -> tuple[list["Toolset"], dict[str, "Toolset"]]:
    value_toolsets, value_named = normalize_toolset_collection(values)
    config_toolsets, config_named = normalize_toolset_collection(config)
    duplicate = set(value_named) & set(config_named)
    if duplicate:
        raise ValueError(f"Toolsets are defined twice: {sorted(duplicate)}.")
    return [*value_toolsets, *config_toolsets], {**value_named, **config_named}


def normalize_toolset_collection(
    value: object,
) -> tuple[list["Toolset"], dict[str, "Toolset"]]:
    from ..toolset import ToolEntry

    if value is None:
        return [], {}
    if isinstance(value, dict):
        named: dict[str, Toolset] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Toolset names must be strings.")
            if key in named:
                raise ValueError(f"Toolset {key!r} is defined twice.")
            named[key] = named_toolset(key, item)
        return list(named.values()), named
    if isinstance(value, str):
        return [normalize_toolset(value)], {}
    if not isinstance(value, Iterable):
        return [normalize_toolset(value)], {}
    return normalize_toolsets(cast(Iterable[ToolEntry], value)), {}


def named_toolset(name: str, value: object) -> "Toolset":
    from ..toolset import Toolset

    value = resolve_config_object(value)
    if isinstance(value, Toolset):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        spec = cast(ConfigData, value)
        if "fn" in spec:
            return toolset_from_factory(name, spec)
        return toolset_from_mapping(spec)
    if callable(value):
        return call_toolset_factory(name, cast(Handler, value), {})
    return normalize_toolset(value)


def toolset_from_factory(name: str, spec: ConfigData) -> "Toolset":
    fn = resolve_config_object(spec.get("fn"))
    if not callable(fn):
        raise TypeError(f"Toolset {name!r} requires callable fn.")
    kwargs = {key: value for key, value in spec.items() if key != "fn"}
    return call_toolset_factory(name, cast(Handler, fn), kwargs)


def call_toolset_factory(name: str, fn: Handler, kwargs: ConfigData) -> "Toolset":
    result = fn(**kwargs)
    if inspect.isawaitable(result):
        raise TypeError(f"Toolset {name!r} fn must be synchronous.")
    toolsets = normalize_toolset_result(result)
    if len(toolsets) != 1:
        raise ValueError(f"Toolset {name!r} fn must return exactly one Toolset.")
    return toolsets[0]


def normalize_toolset_result(value: object) -> list["Toolset"]:
    from ..toolset import ToolEntry, Toolset

    value = resolve_config_object(value)
    if value is None:
        return []
    if isinstance(value, Toolset | dict | str):
        return [normalize_toolset(value)]
    if not isinstance(value, Iterable):
        return [normalize_toolset(value)]
    return normalize_toolsets(cast(Iterable[ToolEntry], value))


def normalize_toolset(value: object) -> "Toolset":
    from ..toolset import ToolEntry, Toolset

    value = resolve_config_object(value)
    if isinstance(value, Toolset):
        return value
    if isinstance(value, dict):
        return toolset_from_mapping(cast(ConfigData, value))
    return Toolset(tools=[cast(ToolEntry, value)])


def toolset_from_mapping(spec: ConfigData) -> "Toolset":
    from ..toolset import Toolset, ToolsetConfig

    extra_keys = set(spec) - set(ToolsetConfig.model_fields)
    if extra_keys:
        raise ValueError(f"Unknown toolset config keys: {sorted(extra_keys)}.")
    return Toolset(config=ToolsetConfig.model_validate(spec))


def tool_items(value: object) -> list["ToolEntry"]:
    if value is None:
        return []
    if isinstance(value, str) or isinstance(value, dict):
        return [tool_item(value)]
    if not isinstance(value, Iterable):
        return [tool_item(value)]
    return [tool_item(item) for item in value]


def tool_item(value: object) -> "ToolEntry":
    from ..toolset import MCPTool, MCPToolConfig, Toolset

    value = resolve_config_object(value)
    if isinstance(value, Toolset | MCPTool | Tool):
        return value
    if isinstance(value, MCPToolConfig):
        return MCPTool(
            command=value.command,
            args=value.args,
            env=value.env,
            cwd=value.cwd,
        )
    if isinstance(value, dict):
        if "command" in value:
            config = coerce_config(MCPToolConfig, cast(ConfigData, value))
            return MCPTool(
                command=config.command,
                args=config.args,
                env=config.env,
                cwd=config.cwd,
            )
        raise TypeError("Tool mapping specs require command.")
    if not callable(value):
        raise TypeError(
            "Tool entries must be callables, Tools, Toolsets, or MCP tool specs."
        )
    return cast(Handler, value)


def toolset_config_mapping(config: BaseModel | ConfigData | None) -> ConfigData:
    from ..toolset import ToolsetConfig

    if config is None:
        return {}
    return resolved_config_data(coerce_config(ToolsetConfig, config))


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Toolset scope must be a string.")
    return value


def tool_name(tool: object) -> str:
    if isinstance(tool, Tool):
        return tool.name
    name = getattr(tool, "__name__", None) or getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Tools require a stable __name__ or name.")
    return name

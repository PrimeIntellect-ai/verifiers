from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Toolset:
    tools: tuple[object, ...] = ()
    show: tuple[str, ...] | None = None
    hide: tuple[str, ...] | None = None
    bindings: Mapping[str, object] = field(default_factory=dict)
    objects: Mapping[str, object] = field(default_factory=dict)
    write: bool = False
    sandbox: Mapping[str, object] | None = None
    cleanup: tuple[object, ...] = ()
    teardown: tuple[object, ...] = ()
    config: object | None = None

    def __init__(
        self,
        tools: Iterable[object] = (),
        show: Iterable[str] | None = None,
        hide: Iterable[str] | None = None,
        bindings: Mapping[str, object] | None = None,
        objects: Mapping[str, object] | None = None,
        write: bool = False,
        sandbox: Mapping[str, object] | None = None,
        cleanup: Iterable[object] = (),
        teardown: Iterable[object] = (),
        config: object | None = None,
    ):
        if show is not None and hide is not None:
            raise ValueError("Toolset accepts show or hide, not both.")
        object.__setattr__(self, "tools", tuple(tools))
        object.__setattr__(self, "show", tuple(show) if show is not None else None)
        object.__setattr__(self, "hide", tuple(hide) if hide is not None else None)
        object.__setattr__(self, "bindings", dict(bindings or {}))
        object.__setattr__(self, "objects", dict(objects or {}))
        object.__setattr__(self, "write", write)
        object.__setattr__(self, "sandbox", sandbox)
        object.__setattr__(self, "cleanup", tuple(cleanup))
        object.__setattr__(self, "teardown", tuple(teardown))
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


def tool_name(tool: object) -> str:
    name = getattr(tool, "__name__", None) or getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Tools require a stable __name__ or name.")
    return name

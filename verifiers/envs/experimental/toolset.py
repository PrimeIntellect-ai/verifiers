from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from verifiers.envs.experimental.binding import (
    Binding,
    BindingContext,
    ResourceBinding,
    StateBinding,
    TaskBinding,
)

if TYPE_CHECKING:
    from verifiers.envs.experimental.channels.tools_channel import ToolRegistry


@dataclass(frozen=True)
class CallableTool:
    func: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    injected_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Toolset:
    tools: tuple[Any, ...] = ()
    bindings: dict[str, object] = field(default_factory=dict)
    channels: dict[str, object] = field(default_factory=dict)
    name: str | None = None

    def __init__(
        self,
        tools: Iterable[Any] = (),
        bindings: dict[str, object] | None = None,
        channels: dict[str, object] | None = None,
        name: str | None = None,
    ):
        object.__setattr__(self, "tools", tuple(tools))
        object.__setattr__(self, "bindings", dict(bindings or {}))
        object.__setattr__(self, "channels", dict(channels or {}))
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class ToolInjector:
    name: str
    resolve: Callable[[BindingContext], object]


@dataclass(frozen=True)
class MCPTool:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    description: str = ""


class ToolArgumentError(ValueError):
    pass


def inject_resources(context: BindingContext) -> object:
    return context.resources


def inject_state(context: BindingContext) -> object:
    return context.state


def inject_task(context: BindingContext) -> object:
    return context.task


def inject_client(context: BindingContext) -> object:
    return getattr(context.resources, "client")


def inject_model(context: BindingContext) -> object:
    return getattr(context.resources, "model")


def inject_sampling_args(context: BindingContext) -> object:
    return getattr(context.resources, "sampling_args")


def inject_tools(context: BindingContext) -> object:
    return getattr(context.resources, "tools")


def default_tool_injectors() -> dict[str, ToolInjector]:
    return {
        "resources": ToolInjector("resources", inject_resources),
        "state": ToolInjector("state", inject_state),
        "task": ToolInjector("task", inject_task),
        "client": ToolInjector("client", inject_client),
        "model": ToolInjector("model", inject_model),
        "sampling_args": ToolInjector("sampling_args", inject_sampling_args),
        "tools": ToolInjector("tools", inject_tools),
    }


def resolve_tool_binding(name: str, value: object, context: BindingContext) -> object:
    if is_context_binding(value):
        binding = cast(Binding | StateBinding | TaskBinding | ResourceBinding, value)
        return binding.resolve(context)
    return value


def is_context_binding(value: object) -> bool:
    return isinstance(value, Binding | StateBinding | TaskBinding | ResourceBinding)


def load_tools_source(source: object) -> ToolRegistry | Toolset | Iterable[Any]:
    from verifiers.envs.experimental.channels.tools_channel import ToolRegistry

    if source is None:
        return []
    if isinstance(source, ToolRegistry | Toolset):
        return source
    if callable(source):
        if is_zero_arg_callable(source):
            return load_tools_source(cast(Callable[[], object], source)())
        raise TypeError("tools must be a list, Toolset, or zero-arg loader.")
    if isinstance(source, str | bytes | Mapping):
        raise TypeError("tools must be a list, Toolset, or zero-arg loader.")
    if isinstance(source, Iterable):
        return source
    raise TypeError("tools must be a list, Toolset, or zero-arg loader.")


def is_zero_arg_callable(value: object) -> bool:
    if not callable(value):
        return False
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect.Parameter.empty:
            return False
    return True

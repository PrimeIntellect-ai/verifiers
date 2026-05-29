from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias, TypeVar, cast, final

from pydantic import StrictBool, model_validator
from verifiers.types import Tool

from .artifact import Artifacts, ArtifactsConfig
from .config import (
    CallableEntry,
    Config,
    ConfigSource,
    resolve_config_object,
)
from .sandbox import SandboxConfig
from .utils.binding_utils import BindingSources, BindingsConfig, binding_sources
from .utils.binding_utils import ObjectsConfig
from .utils.config_utils import (
    coerce_config,
    config_type_from_class,
    registered_config_type,
    register_config_type,
)
from .utils.config_callable_utils import config_callables
from .types import Handler, Objects
from .utils.toolset_utils import (
    collect_toolsets as collect_toolsets,
    flatten_toolsets as flatten_toolsets,
    iter_toolsets as iter_toolsets,
    normalize_toolset as normalize_toolset,
    normalize_toolset_collection as normalize_toolset_collection,
    normalize_toolset_result as normalize_toolset_result,
    tool_item as tool_item,
    tool_items as tool_items,
    tool_name as tool_name,
)

if TYPE_CHECKING:
    from .state import State
    from .task import Task

ToolsetCallableEntry: TypeAlias = CallableEntry | Handler


class MCPToolConfig(Config):
    command: str
    args: list[str] = []
    env: dict[str, str] | None = None
    cwd: str | None = None


ToolEntryConfig: TypeAlias = str | MCPToolConfig


class VisibilityConfig(Config):
    show: list[str] | None = None
    hide: list[str] | None = None

    @model_validator(mode="after")
    def validate_visibility(self) -> "VisibilityConfig":
        if self.show is not None and self.hide is not None:
            raise ValueError("Visibility accepts show or hide, not both.")
        for field_name, names in (("show", self.show), ("hide", self.hide)):
            if names is not None and len(names) != len(set(names)):
                raise ValueError(f"Visibility {field_name} contains duplicate names.")
        return self


class ToolsetConfig(VisibilityConfig):
    tools: list[ToolEntryConfig] = []
    handler: str | None = None
    bindings: BindingsConfig = BindingsConfig()
    objects: ObjectsConfig = ObjectsConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    write: StrictBool = False
    scope: Literal["rollout", "group", "global"] | None = None
    sandbox: SandboxConfig | Literal["program"] | None = None
    stops: list[CallableEntry] = []
    setups: list[CallableEntry] = []
    updates: list[CallableEntry] = []
    cleanups: list[CallableEntry] = []
    teardowns: list[CallableEntry] = []


ConfigT = TypeVar("ConfigT", bound=ToolsetConfig)


@dataclass(frozen=True)
class Toolset(Generic[ConfigT]):
    # Tool surface.
    tools: "tuple[ToolEntry, ...]" = ()
    handler: Handler | None = None
    show: tuple[str, ...] | None = None
    hide: tuple[str, ...] | None = None
    # Local dependencies and runtime policy.
    bindings: BindingSources = field(default_factory=dict)
    objects: Objects = field(default_factory=dict)
    artifacts: Artifacts = field(default_factory=dict)
    write: bool = False
    scope: str | None = None
    sandbox: SandboxConfig | Literal["program"] | None = None
    # Lifecycle collections.
    stops: tuple[Handler, ...] = ()
    setups: tuple[Handler, ...] = ()
    updates: tuple[Handler, ...] = ()
    cleanups: tuple[Handler, ...] = ()
    teardowns: tuple[Handler, ...] = ()
    # Config.
    config: ConfigT | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        config_type = config_type_from_class(
            cls,
            inherited=False,
            owner_base=Toolset,
            config_base=ToolsetConfig,
        )
        if config_type is not None:
            register_config_type(cls, config_type)

    @final
    def __init__(
        self,
        # Tool surface.
        tools: "ToolEntries | None" = None,
        handler: ToolsetCallableEntry | None = None,
        show: Iterable[str] | None = None,
        hide: Iterable[str] | None = None,
        # Local dependencies and runtime policy.
        bindings: BindingSources | BindingsConfig | None = None,
        objects: ObjectsConfig | None = None,
        artifacts: ArtifactsConfig | None = None,
        write: bool | None = None,
        scope: str | None = None,
        sandbox: SandboxConfig | Literal["program"] | None = None,
        # Lifecycle collections.
        stops: Iterable[ToolsetCallableEntry] | None = None,
        setups: Iterable[ToolsetCallableEntry] | None = None,
        updates: Iterable[ToolsetCallableEntry] | None = None,
        cleanups: Iterable[ToolsetCallableEntry] | None = None,
        teardowns: Iterable[ToolsetCallableEntry] | None = None,
        # Config.
        config: ConfigSource = None,
    ):
        if config is not None:
            if any(
                value is not None
                for value in (
                    tools,
                    handler,
                    show,
                    hide,
                    bindings,
                    objects,
                    artifacts,
                    write,
                    scope,
                    sandbox,
                    stops,
                    setups,
                    updates,
                    cleanups,
                    teardowns,
                )
            ):
                raise ValueError(
                    "Toolset accepts either config or constructor fields, not both."
                )
            config_type = registered_config_type(type(self), ToolsetConfig)
            config_value = coerce_config(config_type, config)
            tools = config_value.tools
            handler = config_value.handler
            show = config_value.show
            hide = config_value.hide
            bindings = config_value.bindings
            objects = config_value.objects
            artifacts = config_value.artifacts
            write = config_value.write
            scope = config_value.scope
            sandbox = config_value.sandbox
            stops = config_value.stops
            setups = config_value.setups
            updates = config_value.updates
            cleanups = config_value.cleanups
            teardowns = config_value.teardowns
        else:
            config_value = None
        tool_values = tool_items(tools)
        if show is not None and hide is not None:
            raise ValueError("Toolset accepts show or hide, not both.")
        if isinstance(show, str) or isinstance(hide, str):
            raise TypeError("Toolset show/hide must be lists of names.")
        show_names = tuple(show) if show is not None else None
        hide_names = tuple(hide) if hide is not None else None
        if show_names is not None and not all(
            isinstance(name, str) for name in show_names
        ):
            raise TypeError("Toolset show must contain only strings.")
        if hide_names is not None and not all(
            isinstance(name, str) for name in hide_names
        ):
            raise TypeError("Toolset hide must contain only strings.")
        resolved_handler: object = handler
        if handler is not None:
            resolved_handler = resolve_config_object(handler)
            if not callable(resolved_handler):
                raise TypeError("Toolset handler must resolve to a callable.")
        if write is not None and not isinstance(write, bool):
            raise TypeError("Toolset write must be a boolean.")
        object.__setattr__(self, "tools", tuple(tool_values))
        object.__setattr__(self, "handler", cast(Handler | None, resolved_handler))
        object.__setattr__(self, "show", show_names)
        object.__setattr__(self, "hide", hide_names)
        object.__setattr__(
            self,
            "bindings",
            binding_sources(bindings, "toolset.bindings"),
        )
        object.__setattr__(
            self,
            "objects",
            self.load_objects(ObjectsConfig.model_validate(objects or {})),
        )
        object.__setattr__(
            self,
            "artifacts",
            self.load_artifacts(ArtifactsConfig.model_validate(artifacts or {})),
        )
        object.__setattr__(self, "write", bool(write))
        if scope is not None and scope not in {"rollout", "group", "global"}:
            raise ValueError("Toolset scope must be 'rollout', 'group', or 'global'.")
        object.__setattr__(self, "scope", scope)
        if (
            sandbox is not None
            and sandbox != "program"
            and not isinstance(sandbox, SandboxConfig)
        ):
            raise TypeError("Toolset sandbox must be SandboxConfig or 'program'.")
        object.__setattr__(self, "sandbox", sandbox)
        object.__setattr__(self, "stops", tuple(config_callables(stops or (), "stop")))
        object.__setattr__(
            self, "setups", tuple(config_callables(setups or (), "setup"))
        )
        object.__setattr__(
            self, "updates", tuple(config_callables(updates or (), "update"))
        )
        object.__setattr__(
            self, "cleanups", tuple(config_callables(cleanups or (), "cleanup"))
        )
        object.__setattr__(
            self, "teardowns", tuple(config_callables(teardowns or (), "teardown"))
        )
        object.__setattr__(self, "config", config_value)

    def load_objects(self, config: ObjectsConfig) -> Objects:
        return config.objects("toolset.objects")

    def load_artifacts(self, config: ArtifactsConfig) -> Artifacts:
        return config.artifacts("toolset.artifacts")

    async def get_object(self, name: str, task: "Task", state: "State") -> object:
        return await state._runtime().resolve_owner_object(self, name, task, state)


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


ToolEntry: TypeAlias = Handler | str | Tool | Toolset | MCPTool | MCPToolConfig
ToolEntries: TypeAlias = ToolEntry | Iterable[ToolEntry]
ToolsetItem: TypeAlias = Toolset | ToolEntry
ToolsetCollection: TypeAlias = (
    ToolsetItem | Iterable[ToolsetItem] | dict[str, ToolsetItem | ToolsetConfig]
)
Toolsets: TypeAlias = ToolsetCollection | None

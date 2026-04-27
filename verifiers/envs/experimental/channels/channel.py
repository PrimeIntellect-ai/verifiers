from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from verifiers.envs.experimental.task import Task

ChannelConfig = object
ChannelMap = dict[str, ChannelConfig]
CanonicalizeFn = Callable[[ChannelConfig], ChannelConfig]
ResolveFn = Callable[[list[ChannelConfig], "ChannelContext"], "ResourcePatch"]
ExtendFn = Callable[[object, list[ChannelConfig], "ChannelContext"], object]
ResourceType = type[object]
LifecycleHandler = Callable[..., object]
ChannelRefs = str | tuple[str, ...]


@dataclass(frozen=True)
class ChannelContext:
    """Shared context available while resolving channel configs."""

    objects: Mapping[str, object] = field(default_factory=dict)
    owners: tuple[object, ...] = ()
    phase: str = "env"

    def get_object(self, ref: object) -> object:
        if not isinstance(ref, str):
            return ref
        if ref in self.objects:
            return self.objects[ref]
        raise KeyError(f"No channel object named {ref!r} is registered.")


@dataclass(frozen=True)
class LifecycleHooks:
    stop: tuple[LifecycleHandler, ...] = ()
    render: tuple[LifecycleHandler, ...] = ()
    render_group: tuple[LifecycleHandler, ...] = ()
    cleanup: tuple[LifecycleHandler, ...] = ()
    cleanup_group: tuple[LifecycleHandler, ...] = ()
    teardown: tuple[LifecycleHandler, ...] = ()

    def merged(self, other: "LifecycleHooks") -> "LifecycleHooks":
        return LifecycleHooks(
            stop=(*self.stop, *other.stop),
            render=(*self.render, *other.render),
            render_group=(*self.render_group, *other.render_group),
            cleanup=(*self.cleanup, *other.cleanup),
            cleanup_group=(*self.cleanup_group, *other.cleanup_group),
            teardown=(*self.teardown, *other.teardown),
        )

    def get(self, kind: str) -> tuple[LifecycleHandler, ...]:
        if kind == "stop":
            return self.stop
        if kind == "render":
            return self.render
        if kind == "render_group":
            return self.render_group
        if kind == "cleanup":
            return self.cleanup
        if kind == "cleanup_group":
            return self.cleanup_group
        if kind == "teardown":
            return self.teardown
        raise ValueError(f"Unknown lifecycle hook kind: {kind}")


@dataclass(frozen=True)
class ResourcePatch:
    objects: Mapping[str, object] = field(default_factory=dict)
    hooks: LifecycleHooks = field(default_factory=LifecycleHooks)
    contributions: Mapping[str, tuple[ChannelConfig, ...]] = field(default_factory=dict)

    def merged(self, incoming: "ResourcePatch") -> "ResourcePatch":
        objects = dict(self.objects)
        for name, value in incoming.objects.items():
            if name not in objects:
                objects[name] = value
                continue
            objects[name] = merge_resource_value(name, objects[name], value)
        return ResourcePatch(
            objects=objects,
            hooks=self.hooks.merged(incoming.hooks),
            contributions=merge_channel_contributions(
                self.contributions,
                incoming.contributions,
            ),
        )


@dataclass(frozen=True)
class Channel:
    """Definition for one config-shaped channel."""

    name: str
    outputs: Mapping[str, ResourceType] = field(default_factory=dict)
    extends: ChannelRefs = ()
    always_resolve: bool = False
    canonicalize_fn: CanonicalizeFn | None = None
    resolve_fn: ResolveFn | None = None
    extend_fn: ExtendFn | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "extends", normalize_channel_refs(self.extends))

    def canonicalize(self, config: ChannelConfig) -> ChannelConfig:
        if self.canonicalize_fn is None:
            return config
        return self.canonicalize_fn(config)

    def resolve(
        self, configs: list[ChannelConfig], context: ChannelContext
    ) -> ResourcePatch:
        canonical = [
            self.canonicalize(config) for config in configs if not self.is_empty(config)
        ]
        if self.resolve_fn is not None:
            return self.resolve_fn(canonical, context)
        if len(canonical) > 1:
            raise ValueError(
                f"Channel {self.name!r} received multiple contributions but does "
                "not define how to combine them."
            )
        if not canonical:
            return ResourcePatch()
        output_name = first_output_name(self) or self.name
        return ResourcePatch(objects={output_name: canonical[0]})

    def is_empty(self, config: ChannelConfig) -> bool:
        return config is None

    def extend(
        self,
        current: object,
        configs: list[ChannelConfig],
        context: ChannelContext,
    ) -> object:
        if self.extend_fn is None:
            raise ValueError(f"Channel {self.name!r} cannot be extended.")
        return self.extend_fn(current, configs, context)


def resolve_channels(
    *channel_maps: Mapping[str, ChannelConfig],
    channels: Mapping[str, Channel],
    context: ChannelContext,
) -> ResourcePatch:
    patch = ResourcePatch()
    configs_by_name = collect_channel_configs(*channel_maps)
    pending = resolution_order(configs_by_name, channels)
    resolved_channels: set[str] = set()
    while pending:
        progressed = False
        for name in pending[:]:
            channel = channels[name]
            configs = configs_by_name.get(name, [])
            channel_context = ChannelContext(
                objects={**context.objects, **patch.objects},
                owners=context.owners,
                phase=context.phase,
            )
            incoming = channel.resolve(configs, channel_context)
            extensions = enqueue_channel_contributions(
                incoming.contributions,
                source=channel,
                configs_by_name=configs_by_name,
                channels=channels,
                pending=pending,
                resolved_channels=resolved_channels,
                patch=patch,
                context=ChannelContext(
                    objects={**context.objects, **patch.objects, **incoming.objects},
                    owners=context.owners,
                    phase=context.phase,
                ),
            )
            if extensions:
                patch = ResourcePatch(
                    objects={**patch.objects, **extensions},
                    hooks=patch.hooks,
                    contributions=patch.contributions,
                )
            patch = patch.merged(incoming)
            pending.remove(name)
            resolved_channels.add(name)
            progressed = True
        if not progressed:
            missing = ", ".join(f"{name}: {channels[name].extends}" for name in pending)
            raise ValueError(f"Could not resolve channel dependencies: {missing}")
    return patch


def resolution_order(
    configs_by_name: Mapping[str, list[ChannelConfig]],
    channels: Mapping[str, Channel],
) -> list[str]:
    names: set[str] = set()
    for name, channel in channels.items():
        if name in configs_by_name or channel.always_resolve:
            names.add(name)
    for name in configs_by_name:
        if name not in channels:
            raise ValueError(
                f"Channel {name!r} is not registered. Add it to channel_definitions()."
            )
    names.update(configs_by_name)
    expand_channel_dependencies(names, channels)
    return ordered_channel_names(names, channels)


def expand_channel_dependencies(
    names: set[str], channels: Mapping[str, Channel]
) -> None:
    while True:
        expanded = False
        for name in tuple(names):
            channel = channels[name]
            for target in channel.extends:
                if target not in channels:
                    raise ValueError(
                        f"Channel {name!r} contributes to unknown channel {target!r}."
                    )
                if target not in names:
                    names.add(target)
                    expanded = True
        if not expanded:
            return


def ordered_channel_names(
    names: set[str], channels: Mapping[str, Channel]
) -> list[str]:
    ordered: list[str] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if name in permanent:
            return
        if name in temporary:
            cycle = " -> ".join((*stack, name))
            raise ValueError(f"Circular channel dependency: {cycle}")
        temporary.add(name)
        for dependency in channels_contributing_to(name, names, channels):
            visit(dependency, (*stack, name))
        temporary.remove(name)
        permanent.add(name)
        ordered.append(name)

    for name in channels:
        if name in names:
            visit(name, ())
    return ordered


def channels_contributing_to(
    target: str, names: set[str], channels: Mapping[str, Channel]
) -> list[str]:
    return [
        name for name in channels if name in names and target in channels[name].extends
    ]


def normalize_channel_refs(value: ChannelRefs) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def resolve_resource_objects(
    *owners: object,
    phase: str,
    task: Task | None = None,
) -> ResourcePatch:
    definitions = channel_definitions(*owners)
    return resolve_channels(
        *[raw_channels(owner, task) for owner in owners],
        channels=definitions,
        context=channel_context(*owners, phase=phase),
    )


def collect_channel_configs(
    *channel_maps: Mapping[str, ChannelConfig],
) -> dict[str, list[ChannelConfig]]:
    grouped: dict[str, list[ChannelConfig]] = {}
    for channel_map in channel_maps:
        for name, config in channel_map.items():
            grouped.setdefault(name, []).append(config)
    return grouped


def enqueue_channel_contributions(
    contributions: Mapping[str, tuple[ChannelConfig, ...]],
    *,
    source: Channel,
    configs_by_name: dict[str, list[ChannelConfig]],
    channels: Mapping[str, Channel],
    pending: list[str],
    resolved_channels: set[str],
    patch: ResourcePatch,
    context: ChannelContext,
) -> dict[str, object]:
    extensions: dict[str, object] = {}
    for name, configs in contributions.items():
        if name not in channels:
            raise ValueError(
                f"Channel {name!r} is not registered. Add it to channel_definitions()."
            )
        if name not in source.extends:
            raise ValueError(
                f"Channel {source.name!r} contributed to {name!r} without declaring "
                f"extends={name!r}."
            )
        if name in resolved_channels:
            output_name, value = extend_resolved_channel(
                name, configs, channels[name], patch, context
            )
            extensions[output_name] = value
            continue
        configs_by_name.setdefault(name, []).extend(configs)
        if name not in pending:
            pending.append(name)
    return extensions


def extend_resolved_channel(
    name: str,
    configs: tuple[ChannelConfig, ...],
    channel: Channel,
    patch: ResourcePatch,
    context: ChannelContext,
) -> tuple[str, object]:
    output_name = first_output_name(channel)
    if output_name is None:
        raise ValueError(f"Extendable channel {name!r} must declare an output.")
    if output_name not in patch.objects:
        raise ValueError(f"Resolved resource {output_name!r} is not available.")
    current = patch.objects[output_name]
    extended = channel.extend(current, list(configs), context)
    return output_name, extended


def first_output_name(channel: Channel) -> str | None:
    return next(iter(channel.outputs), None)


def raw_channels(owner: object, task: Task | None = None) -> Mapping[str, object]:
    channels_fn = getattr(owner, "channels", None)
    if not callable(channels_fn):
        return {}
    return dict(channels_fn(task) or {})


def raw_channel_objects(owner: object) -> dict[str, object]:
    objects_fn = getattr(owner, "channel_objects", None)
    if not callable(objects_fn):
        return {}
    return dict(objects_fn() or {})


def raw_channel_definitions(owner: object) -> dict[str, Channel]:
    definitions_fn = getattr(owner, "channel_definitions", None)
    if not callable(definitions_fn):
        return {}
    return dict(definitions_fn() or {})


def channel_definitions(*owners: object) -> dict[str, Channel]:
    from verifiers.envs.experimental.channels import DEFAULT_CHANNELS

    definitions = dict(DEFAULT_CHANNELS)
    owner_definitions: dict[str, Channel] = {}
    for owner in owners:
        for name, channel in raw_channel_definitions(owner).items():
            existing = owner_definitions.get(name)
            if existing is not None and existing != channel:
                raise ValueError(f"Conflicting channel definitions for {name!r}.")
            owner_definitions[name] = channel
            definitions[name] = channel
    return definitions


def channel_resource_types(channels: Mapping[str, Channel]) -> dict[str, ResourceType]:
    resource_types: dict[str, ResourceType] = {}
    for channel in channels.values():
        for name, resource_type in channel.outputs.items():
            existing = resource_types.get(name)
            if existing is not None and existing != resource_type:
                raise ValueError(f"Conflicting resource type for {name!r}.")
            resource_types[name] = resource_type
    return resource_types


def channel_context(*owners: object, phase: str = "env") -> ChannelContext:
    objects: dict[str, object] = {}
    for owner in owners:
        for name, value in raw_channel_objects(owner).items():
            existing = objects.get(name)
            if existing is not None and existing != value:
                raise ValueError(
                    f"Conflicting channel objects named {name!r}: "
                    f"{existing!r} vs {value!r}"
                )
            objects[name] = value
    return ChannelContext(objects=objects, owners=owners, phase=phase)


def lifecycle_handlers(handlers: object) -> list[LifecycleHandler]:
    if handlers is None:
        return []
    if isinstance(handlers, list | tuple | set):
        return [lifecycle_handler(handler) for handler in handlers]
    return [lifecycle_handler(handlers)]


def lifecycle_handler(handler: object) -> LifecycleHandler:
    if not callable(handler):
        raise TypeError("Lifecycle channel entries must be callable.")
    return cast(LifecycleHandler, handler)


def merge_resource_value(name: str, existing: object, incoming: object) -> object:
    if existing == incoming:
        return existing
    if isinstance(existing, list) and isinstance(incoming, list):
        return [*existing, *incoming]
    raise ValueError(
        f"Channel resolution produced conflicting values for resource object {name!r}."
    )


def merge_channel_contributions(
    existing: Mapping[str, tuple[ChannelConfig, ...]],
    incoming: Mapping[str, tuple[ChannelConfig, ...]],
) -> dict[str, tuple[ChannelConfig, ...]]:
    merged = dict(existing)
    for name, configs in incoming.items():
        merged[name] = (*merged.get(name, ()), *configs)
    return merged


def is_empty_config(config: ChannelConfig) -> bool:
    return config is None


def single_config(name: str, configs: list[ChannelConfig]) -> ChannelConfig | None:
    if len(configs) > 1:
        raise ValueError(f"Channel {name!r} received multiple contributions.")
    return configs[0] if configs else None


def as_list(config: ChannelConfig) -> list[object]:
    if is_empty_config(config):
        return []
    if isinstance(config, list | tuple | set):
        return list(config)
    return [config]


def require_mapping(name: str, config: ChannelConfig) -> Mapping[str, object]:
    if not isinstance(config, Mapping):
        raise TypeError(f"Channel {name!r} expected a mapping config.")
    return cast(Mapping[str, object], config)

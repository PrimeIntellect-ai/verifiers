from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.envs.experimental.task import Task

ChannelConfig = object
ChannelMap = Mapping[str, ChannelConfig]
CanonicalizeFn = Callable[[ChannelConfig], ChannelConfig]
ResolvedObjects = dict[str, object]
ResolveFn = Callable[[list[ChannelConfig], "ChannelContext"], ResolvedObjects]


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
class Channel:
    """Definition for one config-shaped channel."""

    name: str
    outputs: tuple[str, ...] = ()
    canonicalize_fn: CanonicalizeFn | None = None
    resolve_fn: ResolveFn | None = None

    def canonicalize(self, config: ChannelConfig) -> ChannelConfig:
        if self.canonicalize_fn is None:
            return config
        return self.canonicalize_fn(config)

    def resolve(
        self, configs: list[ChannelConfig], context: ChannelContext
    ) -> ResolvedObjects:
        canonical = [
            self.canonicalize(config)
            for config in configs
            if not is_empty_config(config)
        ]
        if self.resolve_fn is not None:
            return self.resolve_fn(canonical, context)
        if len(canonical) > 1:
            raise ValueError(
                f"Channel {self.name!r} received multiple contributions but does "
                "not define how to combine them."
            )
        if not canonical:
            return {}
        output_name = self.outputs[0] if self.outputs else self.name
        return {output_name: canonical[0]}


@dataclass(frozen=True)
class ResourceResolution:
    objects: dict[str, object]
    stop_conditions: list[Callable[..., object]] = field(default_factory=list)
    cleanup_handlers: list[Callable[..., object]] = field(default_factory=list)
    teardown_handlers: list[Callable[..., object]] = field(default_factory=list)


def resolve_channels(
    *channel_maps: Mapping[str, ChannelConfig],
    channels: Mapping[str, Channel],
    context: ChannelContext,
) -> ResolvedObjects:
    objects: dict[str, object] = {}
    for name, configs in collect_channel_configs(*channel_maps).items():
        channel = channels.get(name, Channel(name=name, outputs=(name,)))
        for object_name, value in channel.resolve(configs, context).items():
            objects[object_name] = merge_resource_value(
                object_name,
                objects.get(object_name),
                value,
            )
    return objects


def resolve_resource_objects(
    *owners: object,
    phase: str,
    task: Task | None = None,
    normalize: bool = True,
) -> ResourceResolution:
    objects = resolve_channels(
        *[raw_channels(owner, task) for owner in owners],
        channels=channel_definitions(*owners),
        context=channel_context(*owners, phase=phase),
    )
    if normalize:
        normalize_resource_objects(objects)
    return ResourceResolution(
        objects=objects,
        stop_conditions=pop_handlers(objects, "stop_conditions"),
        cleanup_handlers=pop_handlers(objects, "cleanup_handlers"),
        teardown_handlers=pop_handlers(objects, "teardown_handlers"),
    )


def collect_channel_configs(
    *channel_maps: Mapping[str, ChannelConfig],
) -> dict[str, list[ChannelConfig]]:
    grouped: dict[str, list[ChannelConfig]] = {}
    for channel_map in channel_maps:
        for name, config in channel_map.items():
            grouped.setdefault(name, []).append(config)
    return grouped


def raw_channels(owner: object, task: Task | None = None) -> Mapping[str, object]:
    channels_fn = getattr(owner, "channels", None)
    if not callable(channels_fn):
        return {}
    return channels_fn(task) or {}


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


def normalize_resource_objects(objects: dict[str, object]) -> None:
    from verifiers.envs.experimental.channels import NoOpRubric, ToolRegistry

    system_prompt = objects.get("system_prompt")
    if system_prompt is None:
        objects["system_prompt"] = ""
    elif not isinstance(system_prompt, str):
        raise TypeError("The system_prompt channel must resolve to a string.")
    if not isinstance(objects.get("tools"), ToolRegistry):
        objects["tools"] = ToolRegistry()
    if objects.get("rubric") is None:
        objects["rubric"] = NoOpRubric()
    objects.setdefault("skills", [])
    objects.setdefault("sandbox_request", None)
    objects.setdefault("sandbox_scoring", False)
    objects.setdefault("user", None)
    objects.setdefault("upload_dirs", {})


def pop_handlers(objects: dict[str, object], name: str) -> list[Callable[..., object]]:
    return lifecycle_handlers(objects.pop(name, None))


def lifecycle_handlers(handlers: object) -> list[Callable[..., object]]:
    if handlers is None:
        return []
    if isinstance(handlers, list | tuple | set):
        return list(handlers)
    return [handlers]


def merge_resource_value(name: str, existing: object, incoming: object) -> object:
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    if existing == "":
        return incoming
    if incoming == "":
        return existing
    if existing == incoming:
        return existing
    if isinstance(existing, list) and isinstance(incoming, list):
        return [*existing, *incoming]
    if (
        name == "sandbox_scoring"
        and isinstance(existing, bool | int)
        and isinstance(incoming, bool | int)
    ):
        return bool(existing or incoming)
    if (
        name == "upload_dirs"
        and isinstance(existing, Mapping)
        and isinstance(incoming, Mapping)
    ):
        return {**existing, **incoming}
    from verifiers.envs.experimental.channels import ToolRegistry

    if isinstance(existing, ToolRegistry) and isinstance(incoming, ToolRegistry):
        return existing.merged(incoming)
    raise ValueError(
        f"Channel resolution produced conflicting values for resource object {name!r}."
    )


def is_empty_config(config: ChannelConfig) -> bool:
    return config is None or config == "" or config == [] or config == {}


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
    return config

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import cast

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    LifecycleHooks,
    ResourcePatch,
    as_list,
    lifecycle_handlers,
)


def resolve_render(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    rollout_handlers: list[object] = []
    group_handlers: list[object] = []
    for config in configs:
        targets = staged_handlers(config, "render")
        rollout_handlers.extend(targets["rollout"])
        group_handlers.extend(targets["group"])
    return ResourcePatch(
        hooks=LifecycleHooks(
            render=tuple(lifecycle_handlers(rollout_handlers)),
            render_group=tuple(lifecycle_handlers(group_handlers)),
        )
    )


render_channel = Channel(
    name="render",
    resolve_fn=resolve_render,
)


def staged_handlers(config: ChannelConfig, kind: str) -> dict[str, list[object]]:
    targets: dict[str, list[object]] = {"rollout": [], "group": []}
    if isinstance(config, Mapping):
        mapping = cast(Mapping[str, object], config)
        if "fn" in mapping:
            stage = str(mapping.get("stage") or "rollout")
            if stage not in targets:
                raise ValueError(f"Unknown {kind} stage: {stage!r}")
            validate_handler(mapping["fn"], kind, stage)
            targets[stage].append(mapping["fn"])
            return targets
        unknown = set(mapping) - {"rollout", "group"}
        if unknown:
            raise ValueError(f"Unknown {kind} stage(s): {sorted(unknown)}")
        for stage in targets:
            handlers = as_list(mapping.get(stage))
            for handler in handlers:
                validate_handler(handler, kind, stage)
            targets[stage].extend(handlers)
        return targets
    for handler in as_list(config):
        stage = str(getattr(handler, f"{kind}_stage", "rollout"))
        if stage not in targets:
            raise ValueError(f"Unknown {kind} stage: {stage!r}")
        validate_handler(handler, kind, stage)
        targets[stage].append(handler)
    return targets


def validate_handler(handler: object, kind: str, stage: str) -> None:
    if not callable(handler):
        raise TypeError(f"{kind} channel entries must be callable.")
    parameters = inspect.signature(handler).parameters
    names = set(parameters)
    has_group_arg = bool(names & {"tasks", "states"})
    has_rollout_arg = bool(names & {"task", "state"})
    if stage == "group":
        if not has_group_arg:
            raise ValueError(
                f"Group {kind} handler {handler_name(handler)!r} must accept "
                "'tasks' or 'states'."
            )
        if has_rollout_arg:
            raise ValueError(
                f"Group {kind} handler {handler_name(handler)!r} must not accept "
                "singular task/state args."
            )
    elif has_group_arg:
        raise ValueError(
            f"Rollout {kind} handler {handler_name(handler)!r} must not accept "
            "plural tasks/states args. Set stage='group' explicitly for group "
            "handlers."
        )


def handler_name(handler: object) -> str:
    return str(getattr(handler, "__name__", handler.__class__.__name__))

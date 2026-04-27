from __future__ import annotations

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
from verifiers.envs.experimental.channels.render_channel import (
    staged_handlers,
    validate_handler,
)


def resolve_cleanup(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    rollout_handlers: list[object] = []
    group_handlers: list[object] = []
    for config in configs:
        targets = cleanup_targets(config)
        rollout_handlers.extend(targets["rollout"])
        group_handlers.extend(targets["group"])
    return ResourcePatch(
        hooks=LifecycleHooks(
            cleanup=tuple(lifecycle_handlers(rollout_handlers)),
            cleanup_group=tuple(lifecycle_handlers(group_handlers)),
        ),
    )


cleanup_channel = Channel(
    name="cleanup",
    resolve_fn=resolve_cleanup,
)


def cleanup_targets(config: ChannelConfig) -> dict[str, list[object]]:
    if isinstance(config, Mapping) and ("harness" in config or "rubric" in config):
        mapping = cast(Mapping[str, object], config)
        targets: dict[str, list[object]] = {"rollout": [], "group": []}
        unknown = set(mapping) - {"harness", "rubric"}
        if unknown:
            raise ValueError(f"Unknown cleanup target(s): {sorted(unknown)}")
        for handler in as_list(mapping.get("harness")):
            validate_handler(handler, "cleanup", "rollout")
            targets["rollout"].append(handler)
        for handler in as_list(mapping.get("rubric")):
            validate_handler(handler, "cleanup", "rollout")
            targets["rollout"].append(handler)
        return targets
    return staged_handlers(config, "cleanup")

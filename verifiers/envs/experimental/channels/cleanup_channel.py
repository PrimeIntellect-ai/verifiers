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


def resolve_cleanup(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    harness_handlers: list[object] = []
    rubric_handlers: list[object] = []
    for config in configs:
        targets = cleanup_targets(config)
        harness_handlers.extend(targets["harness"])
        rubric_handlers.extend(targets["rubric"])
    contributions: dict[str, tuple[ChannelConfig, ...]] = {}
    if rubric_handlers:
        contributions["rubric"] = ({"cleanup": rubric_handlers},)
    return ResourcePatch(
        hooks=LifecycleHooks(cleanup=tuple(lifecycle_handlers(harness_handlers))),
        contributions=contributions,
    )


cleanup_channel = Channel(
    name="cleanup",
    extends="rubric",
    resolve_fn=resolve_cleanup,
)


def cleanup_targets(config: ChannelConfig) -> dict[str, list[object]]:
    targets: dict[str, list[object]] = {"harness": [], "rubric": []}
    if isinstance(config, Mapping):
        mapping = cast(Mapping[str, object], config)
        unknown = set(mapping) - {"harness", "rubric"}
        if unknown:
            raise ValueError(f"Unknown cleanup target(s): {sorted(unknown)}")
        for target in targets:
            targets[target].extend(as_list(mapping.get(target)))
        return targets
    targets["harness"].extend(as_list(config))
    return targets

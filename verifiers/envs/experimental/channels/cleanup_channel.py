from __future__ import annotations

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
    handlers: list[object] = []
    for config in configs:
        handlers.extend(as_list(config))
    return ResourcePatch(
        hooks=LifecycleHooks(cleanup=tuple(lifecycle_handlers(handlers)))
    )


cleanup_channel = Channel(
    name="cleanup",
    resolve_fn=resolve_cleanup,
)

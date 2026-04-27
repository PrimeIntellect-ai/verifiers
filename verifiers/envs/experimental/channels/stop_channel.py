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


def resolve_stop(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    handlers: list[object] = []
    for config in configs:
        handlers.extend(as_list(config))
    return ResourcePatch(hooks=LifecycleHooks(stop=tuple(lifecycle_handlers(handlers))))


stop_channel = Channel(
    name="stop",
    resolve_fn=resolve_stop,
)

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


def resolve_teardown(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    handlers: list[object] = []
    for config in configs:
        handlers.extend(as_list(config))
    return ResourcePatch(
        hooks=LifecycleHooks(teardown=tuple(lifecycle_handlers(handlers)))
    )


teardown_channel = Channel(
    name="teardown",
    resolve_fn=resolve_teardown,
)

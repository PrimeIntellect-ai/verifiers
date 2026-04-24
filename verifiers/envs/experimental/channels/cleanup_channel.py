from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    as_list,
)


def resolve_cleanup(
    configs: list[ChannelConfig], context: ChannelContext
) -> dict[str, object]:
    handlers: list[object] = []
    for config in configs:
        handlers.extend(as_list(config))
    return {"cleanup_handlers": handlers}


cleanup_channel = Channel(
    name="cleanup",
    outputs=("cleanup_handlers",),
    resolve_fn=resolve_cleanup,
)

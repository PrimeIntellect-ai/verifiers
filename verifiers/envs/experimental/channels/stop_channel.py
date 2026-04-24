from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    as_list,
)


def resolve_stop(
    configs: list[ChannelConfig], context: ChannelContext
) -> dict[str, object]:
    handlers: list[object] = []
    for config in configs:
        handlers.extend(as_list(config))
    return {"stop_conditions": handlers}


stop_channel = Channel(
    name="stop",
    outputs=("stop_conditions",),
    resolve_fn=resolve_stop,
)

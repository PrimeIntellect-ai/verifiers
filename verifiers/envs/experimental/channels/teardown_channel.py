from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    as_list,
)


def resolve_teardown(
    configs: list[ChannelConfig], context: ChannelContext
) -> dict[str, object]:
    handlers: list[object] = []
    for config in configs:
        handlers.extend(as_list(config))
    return {"teardown_handlers": handlers}


teardown_channel = Channel(
    name="teardown",
    outputs=("teardown_handlers",),
    resolve_fn=resolve_teardown,
)

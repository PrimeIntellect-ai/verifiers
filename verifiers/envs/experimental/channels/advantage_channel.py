from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    ResourcePatch,
)
from verifiers.envs.experimental.scoring import signals_from_configs


def resolve_advantage(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    return ResourcePatch(
        objects={"advantages": signals_from_configs(configs, context, "advantage")}
    )


advantage_channel = Channel(
    name="advantage",
    outputs={"advantages": list},
    resolve_fn=resolve_advantage,
)

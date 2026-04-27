from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    ResourcePatch,
)
from verifiers.envs.experimental.scoring import signals_from_configs


def resolve_rewards(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    return ResourcePatch(
        objects={"rewards": signals_from_configs(configs, context, "reward")}
    )


rewards_channel = Channel(
    name="rewards",
    outputs={"rewards": list},
    resolve_fn=resolve_rewards,
)

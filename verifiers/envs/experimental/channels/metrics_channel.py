from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    ResourcePatch,
)
from verifiers.envs.experimental.scoring import signals_from_configs


def resolve_metrics(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    return ResourcePatch(
        objects={"metrics": signals_from_configs(configs, context, "metric")}
    )


metrics_channel = Channel(
    name="metrics",
    outputs={"metrics": list},
    resolve_fn=resolve_metrics,
)

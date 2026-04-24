from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    ResourcePatch,
    single_config,
)


def resolve_system_prompt(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    config = single_config("system_prompt", configs)
    if config is None:
        return ResourcePatch(objects={"system_prompt": ""})
    if not isinstance(config, str):
        raise TypeError("The system_prompt channel must resolve to a string.")
    return ResourcePatch(objects={"system_prompt": config})


system_prompt_channel = Channel(
    name="system_prompt",
    outputs=("system_prompt",),
    output_types={"system_prompt": str},
    always_resolve=True,
    resolve_fn=resolve_system_prompt,
)

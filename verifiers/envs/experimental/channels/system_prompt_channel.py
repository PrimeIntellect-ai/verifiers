from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    single_config,
)


def resolve_system_prompt(
    configs: list[ChannelConfig], context: ChannelContext
) -> dict[str, object]:
    config = single_config("system_prompt", configs)
    if config is None:
        return {}
    if not isinstance(config, str):
        raise TypeError("The system_prompt channel must resolve to a string.")
    return {"system_prompt": config}


system_prompt_channel = Channel(
    name="system_prompt",
    outputs=("system_prompt",),
    resolve_fn=resolve_system_prompt,
)

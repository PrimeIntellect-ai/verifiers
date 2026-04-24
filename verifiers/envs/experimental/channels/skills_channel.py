from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    ResourcePatch,
    as_list,
)


def resolve_skills(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    skills: list[object] = []
    for config in configs:
        skills.extend(as_list(config))
    return ResourcePatch(objects={"skills": skills})


skills_channel = Channel(
    name="skills",
    outputs={"skills": list},
    always_resolve=True,
    resolve_fn=resolve_skills,
)

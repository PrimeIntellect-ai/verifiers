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
    contributions: dict[str, tuple[object, ...]] = {}
    if len(skills) == 1:
        contributions["sandbox"] = ({"uploads": {"skills": skills[0]}},)
    elif len(skills) > 1:
        raise ValueError("The skills channel expects one skills directory.")
    return ResourcePatch(objects={"skills": skills}, contributions=contributions)


skills_channel = Channel(
    name="skills",
    outputs={"skills": list},
    extends="sandbox",
    resolve_fn=resolve_skills,
)

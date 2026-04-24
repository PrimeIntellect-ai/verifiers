from __future__ import annotations

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    as_list,
)


def resolve_skills(
    configs: list[ChannelConfig], context: ChannelContext
) -> dict[str, object]:
    skills: list[object] = []
    for config in configs:
        skills.extend(as_list(config))
    return {"skills": skills}


skills_channel = Channel(name="skills", outputs=("skills",), resolve_fn=resolve_skills)

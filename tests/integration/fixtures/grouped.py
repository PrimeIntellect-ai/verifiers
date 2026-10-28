"""grouped: the echo task, played by an env whose `finalize()` copies the episode's
rollout group onto each trace's `info`, so a test can see what the env saw while the
episode ran."""

from echo import EchoTaskset

import verifiers.v1 as vf


class GroupedEnv(vf.SingleAgentEnv):
    async def finalize(self, task, episode):
        for trace in episode.traces:
            trace.info["group"] = episode.group.id if episode.group else None


class GroupedTaskset(EchoTaskset):
    pass


__all__ = ["GroupedEnv", "GroupedTaskset"]

"""The same Harbor task under its native single-agent environment."""

from harbor_swarm.taskset import HarborSwarmConfig, HarborSwarmTaskset

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborEnv, HarborTask


class HarborBaselineTaskset(vf.Taskset[HarborTask, HarborSwarmConfig]):
    def load(self):
        return [task.harbor() for task in HarborSwarmTaskset(self.config).load()]


__all__ = ["HarborBaselineTaskset", "HarborEnv"]

"""The same Harbor task under its native single-agent environment."""

from harbor_swarm.env import HarborEvidenceEnv
from harbor_swarm.taskset import HarborSwarmConfig, HarborSwarmTaskset

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborTask


class HarborBaselineTaskset(vf.Taskset[HarborTask, HarborSwarmConfig]):
    def load(self):
        return [task.harbor() for task in HarborSwarmTaskset(self.config).load()]


__all__ = ["HarborBaselineTaskset", "HarborEvidenceEnv"]

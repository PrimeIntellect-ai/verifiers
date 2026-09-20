"""World-backed solve with Harbor's native isolated verification lifecycle."""

from pydantic import Field

import verifiers.v1 as vf
from harbor_swarm.repository import archive
from harbor_swarm.taskset import HarborSwarmTask
from verifiers.v1.envs.swarm import SwarmEnv, SwarmEnvConfig
from verifiers.v1.tasksets.harbor.env import HarborEnv, HarborEnvConfig
from verifiers.v1.tasksets.harbor.taskset import HarborTask, verifier_box_data
from verifiers.v1.utils.compile import resolve_runtime_config


class HarborSwarmEnvConfig(SwarmEnvConfig, HarborEnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    coordinator: vf.AgentConfig = vf.AgentConfig()
    participants: dict[str, int] = Field(
        default_factory=lambda: {"solver": 4, "coordinator": 1}
    )
    max_concurrent_agents: int | None = 5


class HarborSwarmEnv(SwarmEnv, HarborEnv, vf.Env[HarborSwarmEnvConfig]):
    config: HarborSwarmEnvConfig

    def verifier_config(self, task):
        base = self.config.verifier.runtime or self.config.solver.runtime
        return resolve_runtime_config(base, HarborTask(verifier_box_data(task.data)))

    async def run(self, task, agents):
        if not isinstance(task, HarborSwarmTask):
            raise TypeError("harbor-swarm requires HarborSwarmTask")
        self.verifier_config(task)
        task.seed_runtime = resolve_runtime_config(
            self.config.solver.runtime, task.harbor()
        )
        if isinstance(task.seed_runtime, vf.SubprocessConfig):
            raise TypeError("Repository tasks require isolated container runtimes")
        await super().run(task, agents)

    async def finalize(self, task, episode):
        if not isinstance(task, HarborSwarmTask):
            raise TypeError("harbor-swarm requires HarborSwarmTask")
        if not episode.traces or any(not trace.ok for trace in episode.traces):
            return
        solution = episode.traces[0]
        task.validate_files(task.snapshot["files"])
        solution.state.artifacts = archive(task.data.workspace, task.snapshot["files"])
        grader = HarborTask(verifier_box_data(task.data))
        scores, graded = await self.grade(
            self.verifier_config(task),
            grader,
            solution,
            scoring_timeout_covers_attempt=True,
        )
        # Retain the exact shared input and grading evidence on the first trace.
        graded.info["repository_snapshot"] = task.snapshot
        episode.traces[0] = graded
        items = scores.items() if isinstance(scores, dict) else [("solved", scores)]
        for trace in episode.traces:
            for name, value in items:
                trace.record_reward(name, value)

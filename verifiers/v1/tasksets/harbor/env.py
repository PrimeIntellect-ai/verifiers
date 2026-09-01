"""The harbor taskset's own env: the single solver seat, plus separate-verifier
grading for tasks that declare ``[verifier].environment_mode = "separate"``.

The default env for harbor runs (the taskset package exports it). A shared-verifier
task runs exactly as under the single-agent env: one `agent` trace, graded in the
box it worked in. A separate-verifier task is graded by `finalize` instead: the
solver's declared artifacts travel (collected by its task `finalize` while its box
is alive), a fresh box is provisioned from the task's verifier declaration,
`tests/` is staged there, and the verifier's rewards land on the solver's trace.
No second agent is involved — the verifier is the task's own `tests/test.sh`.
"""

import copy

import verifiers.v1 as vf
from verifiers.v1.envs.isolated_verifier import (
    IsolatedVerifierEnv,
    IsolatedVerifierEnvConfig,
)
from verifiers.v1.runtimes import Runtime, RuntimeConfig
from verifiers.v1.tasksets.harbor.taskset import (
    HarborTask,
    verifier_box_data,
)
from verifiers.v1.utils.compile import resolve_runtime_config


class HarborEnvConfig(IsolatedVerifierEnvConfig):
    """The Harbor solver plus its optional independent verifier runtime."""

    verifier_runtime: RuntimeConfig | None = None
    """Where a separate-verifier task grades. None derives the grading box from
    the solver runtime; the task's verifier declaration supplies its own image,
    workdir, resources, environment, and network policy."""


class HarborEnv(IsolatedVerifierEnv, vf.Env[HarborEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, HarborTask):
            raise TypeError(
                f"the harbor env runs harbor tasks; got {type(task).__name__}"
            )
        if task.data.verifier is None:
            await agents.agent.run(task)
            return
        # Resolve the verifier's box before the solve, so an impossible pairing
        # (e.g. a restricted Prime verifier without vm=true) costs nothing
        # rather than a full agent run.
        self.verifier_config(task)
        solver_task = copy.copy(task)
        solver_task.verifier_elsewhere = True
        await agents.agent.run(solver_task)

    def verifier_config(self, task: HarborTask) -> RuntimeConfig:
        base = (
            self.config.verifier_runtime
            if self.config.verifier_runtime is not None
            else self.config.agent.runtime
        )
        return resolve_runtime_config(base, HarborTask(verifier_box_data(task.data)))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """Grade a separate-verifier task in its own box, onto the solver's trace.

        Provision a fresh box from the task's verifier declaration, restore the
        solver's collected artifacts, stage `tests/`, run the verifier, and record
        its rewards (and any extra reward.json keys as metrics) on the solver's
        trace. Setup, restoration, staging, and scoring failures retry per
        `verifier_retries`; the last one fails the episode."""
        if not isinstance(task, HarborTask) or task.data.verifier is None:
            return
        solution = episode.traces[0]
        if not solution.ok:
            return
        grader = HarborTask(verifier_box_data(task.data))
        scores, solution = await self.grade(
            self.verifier_config(task), grader, solution
        )
        items = scores.items() if isinstance(scores, dict) else [("solved", scores)]
        for name, value in items:
            solution.record_reward(name, value)
        episode.traces[0] = solution

    async def verify(
        self, task: vf.Task, solution: vf.Trace, runtime: Runtime
    ) -> float | dict[str, float]:
        assert isinstance(task, HarborTask)
        return await task.run_verifier(runtime, solution)

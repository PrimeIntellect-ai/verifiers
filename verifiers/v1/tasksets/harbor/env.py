"""The harbor taskset's own env: the single solver seat, plus separate-verifier
grading for tasks that declare ``[verifier].environment_mode = "separate"``.

The default env for harbor runs (the taskset package exports it). A shared-verifier
task runs exactly as under the single-agent env: one `agent` trace, graded in the
box it worked in. A separate-verifier task is graded by `finalize` instead: the
solver's declared artifacts travel (collected after its task `finalize` while its
box is alive), a fresh box is provisioned from the task's verifier declaration,
`tests/` is staged there, and the verifier's rewards land on the solver's trace.
No second agent is involved — the verifier is the task's own `tests/test.sh`,
staged by the task's own class, so taskset subclasses customize grading through
their task hooks rather than a custom env.
"""

import asyncio
from contextlib import nullcontext
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.agent import resolve_rollout_timeouts
from verifiers.v1.envs.isolated_verifier import (
    IsolatedVerifierEnv,
    IsolatedVerifierEnvConfig,
)
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.runtimes import (
    DockerConfig,
    ModalConfig,
    PrimeConfig,
    Runtime,
    RuntimeConfig,
)
from verifiers.v1.tasksets.harbor.compose import compose_services
from verifiers.v1.tasksets.harbor.taskset import (
    HarborTask,
    verifier_box_data,
)
from verifiers.v1.utils.compile import resolve_runtime_config


class HarborEnvConfig(IsolatedVerifierEnvConfig):
    """The Harbor solver plus its optional independent verifier runtime."""

    trust_compose: bool = False
    """Allow local Compose tasks to use host files and Docker privileges. Only enable
    for trusted task packages; Compose definitions are executable infrastructure."""


class HarborEnv(IsolatedVerifierEnv, vf.Env[HarborEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, HarborTask):
            raise TypeError(
                f"the harbor env runs harbor tasks; got {type(task).__name__}"
            )
        separate = task.data.verifier is not None
        if separate:
            self.verifier_config(task)
        context = None
        if (Path(task.data.task_dir) / "environment/docker-compose.yaml").is_file():
            config = resolve_runtime_config(agents.agent.runtime_config, task)
            if not isinstance(config, (DockerConfig, PrimeConfig, ModalConfig)):
                raise TypeError("Harbor Compose requires Docker, Prime VM or Modal VM")
            timeouts = resolve_rollout_timeouts(agents.agent.timeout, task)
            context = compose_services(
                config,
                task,
                trust_compose=self.config.trust_compose,
                setup_timeout=timeouts.setup,
            )
        async with context or nullcontext(({}, None)) as (runtimes, stop_main):
            trace = await agents.agent.run(
                task.defer_scoring() if separate else task,
                runtime=runtimes.get("main"),
                collect_artifacts=separate,
                on_trace=(lambda trace: trace.state.services.update(runtimes))
                if runtimes
                else None,
            )
            try:
                services = {
                    *(artifact.service for artifact in task.data.artifacts),
                    *(hook.service for hook in task.data.collect),
                } - {"main"}
                if stop_main is not None and separate and trace.ok and services:
                    # Harness cleanup has finished; freeze main before collecting sidecars.
                    async with (
                        boundary(TaskError, "collecting Compose sidecars"),
                        asyncio.timeout(timeouts.finalize),
                    ):
                        await stop_main()
                        await task.finalize(trace, runtimes["main"], services=services)
            finally:
                trace.state.services.clear()

    def verifier_config(self, task: HarborTask) -> RuntimeConfig:
        base = (
            self.config.verifier.runtime
            if self.config.verifier.runtime is not None
            else self.config.agent.runtime
        )
        return resolve_runtime_config(base, HarborTask(verifier_box_data(task.data)))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """Grade a separate-verifier task in its own box, onto the solver's trace.

        Provision a fresh box from the task's verifier declaration, restore the
        solver's collected artifacts, stage `tests/`, run the verifier, and record
        its rewards (and any extra reward.json keys as metrics) on the solver's
        trace. The grader is the task's own class, so a `HarborTask` subclass's
        `setup` and `stage_verifier` run in the verifier box too. Setup,
        restoration, staging, and scoring failures retry per `verifier.retries`;
        the last one fails the episode."""
        if not isinstance(task, HarborTask) or task.data.verifier is None:
            return
        solution = episode.traces[0]
        if not solution.ok:
            return
        runtime = solution.agent.runtime
        if task.data.verifier.fresh_copy and runtime is not None and runtime.borrowed:
            # Compose resolves images and workdirs at startup, including local builds.
            task = type(task)(
                task.data.model_copy(
                    update=runtime.model_dump(include={"image", "workdir"})
                ),
                task.config,
            )
        grader = type(task)(verifier_box_data(task.data), task.config)
        scores, solution = await self.grade(
            self.verifier_config(task),
            grader,
            solution,
            scoring_timeout_covers_attempt=True,
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

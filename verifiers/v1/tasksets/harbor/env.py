"""Harbor service orchestration with common task verification."""

import asyncio
import time
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.agent import resolve_rollout_timeouts
from verifiers.v1.configs.scoring import resolve_scoring
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.grading import grade_task, verifier_runtime
from verifiers.v1.tasksets.harbor.compose import compose_services
from verifiers.v1.tasksets.harbor.taskset import (
    HarborTask,
)
from verifiers.v1.utils.compile import resolve_runtime_config


class HarborEnvConfig(vf.EnvConfig):
    """The Harbor solver and trusted Compose configuration."""

    agent: vf.AgentConfig = vf.AgentConfig()

    trust_compose: bool = False
    """Allow local Compose tasks to use host files and Docker privileges. Only enable
    for trusted task packages; Compose definitions are executable infrastructure."""


class HarborEnv(vf.Env[HarborEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, HarborTask):
            raise TypeError(
                f"the harbor env runs harbor tasks; got {type(task).__name__}"
            )
        if not (Path(task.data.task_dir) / "environment/docker-compose.yaml").is_file():
            await agents.agent.run(task)
            return
        scoring = resolve_scoring(task.scoring, self.config.agent.scoring)
        separate = scoring.mode == "isolated"
        original = task
        if separate:
            task = task.defer_scoring()
        timeouts = resolve_rollout_timeouts(agents.agent.timeout, task)
        async with compose_services(
            resolve_runtime_config(agents.agent.runtime_config, task),
            task,
            trust_compose=self.config.trust_compose,
            setup_timeout=timeouts.setup,
        ) as (services, stop_main):
            # Ordered like the hooks, which Harbor runs as authored.
            declared = dict.fromkeys(
                entry.service for entry in (*task.data.collect, *task.data.artifacts)
            )
            if missing := declared.keys() - services.keys():
                raise ValueError(f"Unknown Compose services: {sorted(missing)}")
            if separate:
                grader = original.grading_task(services["main"], scoring=scoring)
                config = verifier_runtime(
                    grader,
                    services["main"],
                    agents.agent.runtime_config,
                    scoring,
                )
            trace = await agents.agent.run(
                task, runtime=services["main"], collect_artifacts=separate
            )
            sidecars = {name: services[name] for name in declared if name != "main"}
            if separate and trace.ok and sidecars:
                # As in Harbor, main stops after its own collection so leftover agent
                # processes cannot interfere with sidecar evidence.
                async with (
                    boundary(TaskError, "collecting Compose sidecars"),
                    asyncio.timeout(timeouts.finalize),
                ):
                    await stop_main()
                    await task.finalize(trace, services["main"], sidecars)

        # Release the solver project before provisioning any fresh verifier.
        if separate and trace.ok:
            try:
                await grade_task(
                    grader,
                    trace,
                    config,
                    timeouts,
                    scoring,
                    scoring_elapsed=trace.info.get("harness_scoring_seconds", 0.0),
                )
            except TimeoutError:
                trace.record_timeout("scoring")
            except Exception as error:
                trace.record_error(error)
                raise
            finally:
                trace.timing.scoring.end = time.time()
                trace.notify()

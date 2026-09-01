"""Deterministic task verification in a fresh runtime.

One solver agent runs the task. Its task scoring is deferred, declared artifacts
are collected after normal task finalization, and the solver runtime is destroyed.
The task is then set up with a fresh controller in a fresh runtime, its artifacts
are restored, and its ordinary metrics and rewards run there onto the solver's trace.
"""

import asyncio
import copy
import logging
from contextlib import AsyncExitStack
from pathlib import PurePosixPath
from typing import Any

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.agent import resolve_rollout_timeouts
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.runtimes import Runtime, RuntimeConfig, provision_runtime
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.decorators import invoke
from verifiers.v1.utils.retries import backoff

logger = logging.getLogger(__name__)


class IsolatedVerifierEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    """The one solver seat."""
    verifier_runtime: RuntimeConfig | None = None
    """Independent verifier placement and policy. None provisions a fresh runtime
    equivalent to the solver's resolved task runtime."""
    verifier_env: dict[str, str] | None = None
    """Process environment for verifier setup and scoring. None uses the task's
    normal runtime environment."""
    verifier_retries: int = Field(2, ge=0)
    """Extra fresh-runtime attempts after verifier setup or scoring failures."""


class IsolatedVerifierEnv(vf.Env[IsolatedVerifierEnvConfig]):
    """Run one solver, then its deterministic task scoring in a fresh runtime."""

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if task.config.judges:
            raise ValueError(
                "isolated-verifier runs deterministic task metrics and rewards; "
                "model-backed task judges are not supported"
            )
        self.verifier_config(task)  # Refuse an impossible verifier before solving.
        await agents.agent.run(task.graded_elsewhere())

    def verifier_config(self, task: vf.Task) -> RuntimeConfig:
        base = self.config.verifier_runtime or self.config.agent.runtime
        config = resolve_runtime_config(base, task)
        if (
            self.config.verifier_runtime is not None
            and "image" in base.model_fields_set
        ):
            config = config.model_copy(update={"image": base.image})
        if isinstance(config, vf.SubprocessConfig):
            raise TypeError(
                "isolated-verifier requires a container runtime so artifacts can be "
                "restored safely; configure the agent or verifier runtime as docker, "
                "prime, or modal"
            )
        relative = [
            artifact.source
            for artifact in task.data.artifacts
            if not PurePosixPath(artifact.source).is_absolute()
        ]
        solver = resolve_runtime_config(self.config.agent.runtime, task)
        solver_workdir = PurePosixPath(getattr(solver, "workdir", "") or "/")
        verifier_workdir = PurePosixPath(config.workdir)
        if relative and solver_workdir != verifier_workdir:
            raise ValueError(
                "isolated-verifier cannot transfer relative artifacts "
                f"{relative!r} between solver workdir {str(solver_workdir)!r} and "
                f"verifier workdir {str(verifier_workdir)!r}; use matching workdirs "
                "or absolute artifact paths"
            )
        return config

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        solution = episode.traces[0]
        if solution.ok:
            graded = await self.grade(self.verifier_config(task), task, solution)
            episode.traces[0] = graded[1]

    async def stage_verifier(
        self, task: vf.Task, solution: vf.Trace, runtime: Runtime
    ) -> None:
        artifacts = dict(solution.state.artifacts)
        async with boundary(TaskError, "verifier task setup"):
            await invoke(task.setup, {"trace": solution, "runtime": runtime})
        await vf.restore(runtime, artifacts)
        async with boundary(TaskError, "verifier staging"):
            await invoke(task.stage_verifier, {"trace": solution, "runtime": runtime})

    async def verify(self, task: vf.Task, solution: vf.Trace, runtime: Runtime) -> Any:
        await task.score(solution, runtime)

    async def grade(
        self, config: RuntimeConfig, task: vf.Task, solution: vf.Trace
    ) -> tuple[Any, vf.Trace]:
        timeouts = resolve_rollout_timeouts(self.config.agent.timeout, task)
        last: Exception | None = None
        for attempt in range(self.config.verifier_retries + 1):
            if attempt:
                delay = backoff(attempt - 1)
                logger.warning(
                    "isolated verifier attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    self.config.verifier_retries + 1,
                    last,
                    delay,
                )
                await asyncio.sleep(delay)
            try:
                # Teardown is outside the stage deadlines: a completed score must
                # survive a slow cleanup of the verifier runtime.
                async with AsyncExitStack() as boxes:
                    async with asyncio.timeout(timeouts.setup):
                        # Failed setup or scoring must not alter the next attempt.
                        # Only the successful controller and trace leave this scope.
                        verifier_task = copy.deepcopy(task)
                        verifier_solution = copy.deepcopy(solution)
                        runtime = await boxes.enter_async_context(
                            provision_runtime(
                                config,
                                env=(
                                    verifier_task.runtime_env()
                                    if self.config.verifier_env is None
                                    else self.config.verifier_env
                                ),
                            )
                        )
                        await runtime.prepare_setup()
                        await self.stage_verifier(
                            verifier_task, verifier_solution, runtime
                        )
                        await runtime.prepare_execution([])
                    async with asyncio.timeout(timeouts.scoring):
                        result = await self.verify(
                            verifier_task, verifier_solution, runtime
                        )
                    return result, verifier_solution
            except Exception as error:  # noqa: BLE001 - retry the whole fresh box
                last = error
        assert last is not None
        raise last

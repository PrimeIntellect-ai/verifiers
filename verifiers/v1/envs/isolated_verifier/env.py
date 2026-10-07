"""Deterministic task verification in a fresh runtime.

One solver agent runs the task. Its task scoring is deferred, declared artifacts
are collected after normal task finalization, and the solver runtime is destroyed.
The task is then set up with a fresh controller in a fresh runtime, its artifacts
are restored, and its ordinary metrics and rewards run there onto the solver's trace.
"""

import logging
from pathlib import PurePosixPath
from typing import Any

from pydantic import Field, SerializationInfo, field_serializer

import verifiers.v1 as vf
from verifiers.v1.agent import resolve_rollout_timeouts
from verifiers.v1.runtimes import Runtime, RuntimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.verify import grade_in_fresh_runtime, score_task, stage_task

logger = logging.getLogger(__name__)


class VerifierConfig(vf.BaseConfig):
    runtime: RuntimeConfig | None = None
    """Independent verifier placement and policy. None provisions a fresh runtime
    equivalent to the solver's resolved task runtime."""
    env: dict[str, str] | None = None
    """Process environment for verifier setup and scoring. None uses the task's
    normal runtime environment."""
    retries: int = Field(2, ge=0)
    """Extra fresh-runtime attempts after setup, restoration, staging, or scoring
    failures."""

    @field_serializer("runtime")
    def serialize_runtime(
        self, runtime: RuntimeConfig | None, info: SerializationInfo
    ) -> dict | None:
        """Keep an omitted image omitted across a resolved-config round trip."""
        if runtime is None:
            return None
        values = runtime.model_dump(mode=info.mode)
        if "image" not in runtime.model_fields_set:
            values.pop("image", None)
        return values


class IsolatedVerifierEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    """The one solver seat."""
    verifier: VerifierConfig = VerifierConfig()
    """The fresh verifier's runtime, process environment, and retry policy."""


class IsolatedVerifierEnv(vf.Env[IsolatedVerifierEnvConfig]):
    """Run one solver, then its deterministic task scoring in a fresh runtime."""

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if task.config.judges:
            raise ValueError(
                "isolated-verifier runs deterministic task metrics and rewards; "
                "model-backed task judges are not supported"
            )
        self.verifier_config(task)  # Refuse an impossible verifier before solving.
        await agents.agent.run(task.defer_scoring(), collect_artifacts=True)

    def verifier_config(self, task: vf.Task) -> RuntimeConfig:
        base = self.config.verifier.runtime or self.config.agent.runtime
        config = resolve_runtime_config(base, task)
        image_spec = type(base).model_fields.get("image")
        if (
            self.config.verifier.runtime is not None
            and image_spec is not None
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
        solver_workdir = PurePosixPath(getattr(solver, "workdir", "/") or "/app")
        verifier_workdir = PurePosixPath(config.workdir or "/app")
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
        await stage_task(task, solution, runtime)

    async def verify(self, task: vf.Task, solution: vf.Trace, runtime: Runtime) -> Any:
        await score_task(task, solution, runtime)

    async def grade(
        self,
        config: RuntimeConfig,
        task: vf.Task,
        solution: vf.Trace,
        *,
        scoring_timeout_covers_attempt: bool = False,
    ) -> tuple[Any, vf.Trace]:
        return await grade_in_fresh_runtime(
            config,
            task,
            solution,
            timeouts=resolve_rollout_timeouts(self.config.agent.timeout, task),
            retries=self.config.verifier.retries,
            env=self.config.verifier.env,
            stage=self.stage_verifier,
            verify=self.verify,
            scoring_timeout_covers_attempt=scoring_timeout_covers_attempt,
        )

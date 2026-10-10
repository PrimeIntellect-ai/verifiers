"""Task scoring in a fresh runtime, independent of environment control flow."""

import asyncio
import copy
import logging
from contextlib import AsyncExitStack
from pathlib import PurePosixPath

from verifiers.v1.configs.verifier import VerifierConfig
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.runtimes import (
    PrimeConfig,
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
)
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.utils.artifacts import restore
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.decorators import invoke
from verifiers.v1.utils.retries import backoff

logger = logging.getLogger(__name__)


def verifier_runtime(
    task: Task, solver: Runtime, base: RuntimeConfig, policy: VerifierConfig
) -> RuntimeConfig:
    source = policy.runtime or base
    if isinstance(source, PrimeConfig):
        if policy.runtime is not None and source.checkpoint:
            raise TaskError("isolated verification requires an image, not a checkpoint")
        source = source.model_copy(update={"checkpoint": None})
    # A task without its own image/workdir inherits the actual workspace's values,
    # including a Compose image or split execution configuration.
    task = task.with_data(
        image=task.data.image or getattr(solver.config, "image", None),
        workdir=task.data.workdir or getattr(solver.config, "workdir", None),
    )
    config = resolve_runtime_config(source, task)
    if policy.runtime is not None and "image" in policy.runtime.model_fields_set:
        config = config.model_copy(update={"image": policy.runtime.image})
    if isinstance(config, SubprocessConfig):
        raise TaskError("isolated verification requires an isolated filesystem")
    relative = [
        a.source
        for a in task.data.artifacts
        if not PurePosixPath(a.source).is_absolute()
    ]
    solver_dir = PurePosixPath(getattr(solver.config, "workdir", "/") or "/app")
    grader_dir = PurePosixPath(getattr(config, "workdir", "/") or "/app")
    if relative and solver_dir != grader_dir:
        raise TaskError(
            f"cannot restore relative artifacts {relative!r} from {solver_dir} into {grader_dir}; use matching workdirs or absolute paths"
        )
    return config


async def stage_verifier(task: Task, trace: Trace, runtime: Runtime) -> None:
    artifacts = dict(trace.state.artifacts)
    async with boundary(TaskError, "verifier task setup"):
        await invoke(task.setup, {"trace": trace, "runtime": runtime})
    await restore(runtime, artifacts)
    async with boundary(TaskError, "verifier staging"):
        await invoke(task.stage_verifier, {"trace": trace, "runtime": runtime})


async def grade_task(
    task: Task,
    trace: Trace,
    config: RuntimeConfig,
    timeouts,
    policy: VerifierConfig,
    deployment,
) -> None:
    for attempt in range(policy.retries + 1):
        if attempt:
            await asyncio.sleep(backoff(attempt - 1))
        try:
            async with AsyncExitStack() as stack:
                grader, result = copy.deepcopy(task), copy.deepcopy(trace)
                async with asyncio.timeout(timeouts.setup):
                    target = await stack.enter_async_context(
                        deployment.verifier(
                            config,
                            env=grader.runtime_env()
                            if policy.env is None
                            else policy.env,
                        )
                    )
                    await target.prepare_setup()
                    await deployment.run(
                        stage_verifier(grader, result, target), target=target
                    )
                    await target.prepare_execution([])
                try:
                    async with asyncio.timeout(timeouts.scoring):
                        await deployment.run(
                            grader.score(result, target), target=target
                        )
                finally:
                    # Model-backed judges are billed even when an attempt fails.
                    # Preserve their accounting without committing partial rewards/state.
                    trace.extra_usage = result.extra_usage
                    if "judge_calls" in result.info:
                        trace.info["judge_calls"] = result.info["judge_calls"]
                # Cleanup failure must not discard an authoritative completed score.
                trace.state, trace.rewards, trace.metrics = (
                    result.state,
                    result.rewards,
                    result.metrics,
                )
                trace.info = result.info
                try:
                    await stack.aclose()
                except Exception:
                    logger.warning("verifier teardown failed", exc_info=True)
                return
        except Exception:
            if attempt == policy.retries:
                raise
            logger.warning(
                "verifier attempt %d failed; retrying", attempt + 1, exc_info=True
            )

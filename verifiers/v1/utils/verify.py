"""Grade a finished solution in a fresh runtime that receives only its artifacts."""

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any

from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.rollout import RolloutTimeouts
from verifiers.v1.runtimes import Runtime, RuntimeConfig, provision_runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.utils.artifacts import restore
from verifiers.v1.utils.decorators import invoke
from verifiers.v1.utils.retries import backoff

logger = logging.getLogger(__name__)

Stage = Callable[[Task, Trace, Runtime], Awaitable[None]]
Verify = Callable[[Task, Trace, Runtime], Awaitable[Any]]

GRADER_RETRIES = 2
"""Extra fresh-runtime attempts for a task's `grader()`."""


async def stage_task(task: Task, solution: Trace, runtime: Runtime) -> None:
    """Set the task up in the fresh box, restore the solution's artifacts, and stage
    the task's trusted verifier inputs."""
    async with boundary(TaskError, "verifier task setup"):
        await invoke(task.setup, {"trace": solution, "runtime": runtime})
    await restore(runtime, dict(solution.state.artifacts))
    async with boundary(TaskError, "verifier staging"):
        await invoke(task.stage_verifier, {"trace": solution, "runtime": runtime})


async def score_task(task: Task, solution: Trace, runtime: Runtime) -> None:
    await task.score(solution, runtime)


async def grade_in_fresh_runtime(
    config: RuntimeConfig,
    task: Task,
    solution: Trace,
    *,
    timeouts: RolloutTimeouts,
    retries: int,
    env: dict[str, str] | None = None,
    stage: Stage = stage_task,
    verify: Verify = score_task,
    scoring_timeout_covers_attempt: bool = False,
) -> tuple[Any, Trace]:
    """Provision a fresh runtime from `config`, stage `task` there with the
    solution's artifacts, and verify. Each failed attempt retries in a new runtime;
    the last failure raises. Returns the verifier's result and the graded copy of
    `solution`."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            delay = backoff(attempt - 1)
            logger.warning(
                "isolated verifier attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                retries + 1,
                last,
                delay,
            )
            await asyncio.sleep(delay)
        try:
            # Teardown is outside the stage deadlines: a completed score must
            # survive a slow cleanup of the verifier runtime.
            async with (
                AsyncExitStack() as boxes,
                asyncio.timeout(
                    timeouts.scoring if scoring_timeout_covers_attempt else None
                ),
            ):
                async with asyncio.timeout(timeouts.setup):
                    # Failed setup or scoring must not alter the next attempt.
                    # Only the successful controller and trace leave this scope.
                    verifier_task = copy.deepcopy(task)
                    verifier_solution = copy.deepcopy(solution)
                    runtime = await boxes.enter_async_context(
                        provision_runtime(
                            config,
                            env=verifier_task.runtime_env() if env is None else env,
                        )
                    )
                    await runtime.prepare_setup()
                    await stage(verifier_task, verifier_solution, runtime)
                    await runtime.prepare_execution([])
                async with asyncio.timeout(timeouts.scoring):
                    result = await verify(verifier_task, verifier_solution, runtime)
                return result, verifier_solution
        except Exception as error:  # noqa: BLE001 - retry the whole fresh box
            last = error
    assert last is not None
    raise last

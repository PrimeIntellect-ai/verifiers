"""Task scoring in a fresh runtime, independent of environment control flow."""

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, contextmanager
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.runtimes import (
    NetworkPolicyConfig,
    PrimeConfig,
    Runtime,
    RuntimeConfig,
    backend_cls,
)
from verifiers.v1.task import Task
from verifiers.v1.trace import Error, Trace
from verifiers.v1.utils.artifacts import restore
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.decorators import invoke
from verifiers.v1.utils.env import resolve_env
from verifiers.v1.utils.retries import RetryState, backoff

if TYPE_CHECKING:
    from verifiers.v1.runtimes.deployment import Deployment

logger = logging.getLogger(__name__)
_SCORING_INFO_KEYS = ("judge_calls", "scoring_diagnostics")
"""Accounting and diagnostics survive failed scoring attempts, unlike partial scores."""


@contextmanager
def scoring_transaction(trace: Trace):
    """Discard partial shared scores when a required scoring callback fails."""
    saved = copy.deepcopy((trace.state, trace.rewards, trace.metrics, trace.info))
    try:
        yield
    except BaseException:
        retained = {
            key: trace.info[key] for key in _SCORING_INFO_KEYS if key in trace.info
        }
        trace.state, trace.rewards, trace.metrics, trace.info = saved
        trace.info.update(retained)
        raise


def verifier_runtime(
    task: Task, solver_runtime: Runtime, base: RuntimeConfig, policy: ScoringConfig
) -> RuntimeConfig:
    source = policy.runtime or base
    if isinstance(source, PrimeConfig):
        if policy.runtime is not None and source.checkpoint:
            raise TaskError("isolated verification requires an image, not a checkpoint")
        source = source.model_copy(update={"checkpoint": None})
    # A task without its own image inherits the actual workspace's image and workdir,
    # including a Compose image or split execution configuration. One with its own image
    # and no workdir runs in that image's WORKDIR, as a container of it starts — unless
    # it restores relative artifacts, which land where the solver's workdir put them.
    relative_artifacts = any(
        not PurePosixPath(a.source).is_absolute() for a in task.data.artifacts
    )
    inherit_workdir = task.data.image is None or relative_artifacts
    task = task.with_data(
        image=task.data.image or getattr(solver_runtime.config, "image", None),
        workdir=task.data.workdir
        or (
            getattr(solver_runtime.config, "workdir", None) if inherit_workdir else None
        ),
    )
    if policy.network is not None:
        task = task.with_data(network=policy.network)
    config = resolve_runtime_config(source, task)
    if policy.runtime is not None and "image" in policy.runtime.model_fields_set:
        config = config.model_copy(update={"image": policy.runtime.image})
    if not backend_cls(config).SUPPORTS_TASK_IMAGES:
        raise TaskError(
            f"isolated verification requires an isolated filesystem; the "
            f"{config.type} runtime cannot run images"
        )
    relative = [
        a.source
        for a in task.data.artifacts
        if not PurePosixPath(a.source).is_absolute()
    ]
    solver_dir = PurePosixPath(getattr(solver_runtime.config, "workdir", "/") or "/app")
    grader_dir = PurePosixPath(getattr(config, "workdir", "/") or "/app")
    if relative and solver_dir != grader_dir:
        raise TaskError(
            f"cannot restore relative artifacts {relative!r} from {solver_dir} into {grader_dir}; use matching workdirs or absolute paths"
        )
    return config


def scoring_network(
    base: RuntimeConfig, network: NetworkPolicyConfig
) -> NetworkPolicyConfig:
    """`network` as the task's policy while scoring, combined with the runtime's own
    rules the way `TaskData.network` is."""
    if isinstance(base, NetworkPolicyConfig):
        return base.with_task_network_policy(network)
    if network.network_restricted:
        raise TaskError(
            f"scoring requires a network policy, but the {base.type} runtime does not "
            "support one"
        )
    return network


def bind_scoring_env(runtime: Runtime, policy: ScoringConfig) -> Runtime:
    """Bind scoring variables without changing the solver's runtime or backend."""
    return runtime.with_env({**runtime.env, **resolve_env(policy.env or {})})


async def stage_verifier(task: Task, trace: Trace, runtime: Runtime) -> None:
    artifacts = dict(trace.state.artifacts)
    async with boundary(TaskError, "verifier task setup"):
        await invoke(task.setup, {"trace": trace, "runtime": runtime})
    await restore(runtime, artifacts, protected=task.RESTORE_PROTECTED)
    async with boundary(TaskError, "verifier staging"):
        await invoke(task.stage_verifier, {"trace": trace, "runtime": runtime})


async def grade_task(
    task: Task,
    trace: Trace,
    config: RuntimeConfig,
    timeouts,
    policy: ScoringConfig,
    deployment,
    *,
    scoring_elapsed: float = 0.0,
    on_score=None,
) -> None:
    await deployment.prepare_verifier()
    # Provisioning, scoring, and retry backoff share the remaining scoring budget.
    # Setup may have an additional, stricter limit; retries do not reset scoring.
    deadline = (
        asyncio.get_running_loop().time() + max(0.0, timeouts.scoring - scoring_elapsed)
        if timeouts.scoring is not None
        else None
    )
    retry = RetryState(policy.retries)
    for attempt in range(retry.max_retries + 1):
        if attempt:
            async with asyncio.timeout_at(deadline):
                await asyncio.sleep(backoff(attempt - 1))
        try:
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("scoring budget exhausted")
            async with AsyncExitStack() as stack:
                grader, result = copy.deepcopy(task), copy.deepcopy(trace)
                async with (
                    asyncio.timeout_at(deadline),
                    asyncio.timeout(timeouts.setup),
                    boundary(TaskError, "verifier setup"),
                ):
                    runtime = await stack.enter_async_context(
                        deployment.verifier(
                            config,
                            env=grader.runtime_env(),
                        )
                    )
                    runtime = bind_scoring_env(runtime, policy)
                    await runtime.backend.prepare_setup()
                    await stage_verifier(grader, result, runtime)
                    await runtime.backend.prepare_execution([])
                try:
                    async with (
                        asyncio.timeout_at(deadline),
                        boundary(TaskError, "verifier scoring"),
                    ):
                        await grader.score(result, runtime, deployment)
                        if on_score is not None:
                            await on_score(result, runtime)
                finally:
                    # Preserve failed attempts' evidence and accounting without
                    # committing partial rewards/state.
                    trace.extra_usage = result.extra_usage
                    trace.info.update(
                        {
                            key: result.info[key]
                            for key in _SCORING_INFO_KEYS
                            if key in result.info
                        }
                    )
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
        except Exception as error:
            recorded = Error(
                type=type(error).__name__,
                message=str(error),
                status_code=getattr(error, "status_code", None),
            )
            if attempt == retry.max_retries or retry.next_error([recorded]) is None:
                raise
            logger.warning(
                "verifier attempt %d failed; retrying", attempt + 1, exc_info=True
            )


async def grade_submission(
    task: Task,
    trace: Trace,
    runtime: Runtime,
    *,
    scoring: ScoringConfig | None = None,
    deployment: "Deployment | None" = None,
    on_score: Callable[[Trace, Runtime], Awaitable[None]] | None = None,
) -> None:
    """Grade captured artifacts in a fresh verifier, as a rollout's close does.

    The caller finalizes the submission and collects artifacts into the trace first.
    Uses the task's grader, scoring policy, and timeouts, including retry behavior.
    Without `deployment`, `runtime` is borrowed and stays alive. With the `Deployment`
    that owns `runtime` (a model-free run of a composed task), grading stops its
    workspace, collects sidecar evidence (`Task.finalize_services`), and releases its
    resources before the verifier starts; the caller still closes it.

    `on_score(trace, runtime)` runs after task scoring in the verifier, as `Agent.run`'s
    does: inside its timeout and retry boundary, and only an attempt it finishes
    commits the scores it left on the trace.

    As in a rollout, the verifier derives from the agent's runtime config on the trace
    (`trace.agent.config.runtime`), not from the solver's resolved one, so the solver
    task's requirements, such as its network policy, don't carry over.
    """
    from verifiers.v1.runtimes.deployment import Deployment

    policy = scoring if scoring is not None else task.scoring
    if policy.mode != "isolated":
        raise ValueError("grade_submission requires isolated scoring")
    grader = task.grading_task(runtime, scoring=policy)
    borrowed = deployment is None
    if borrowed:
        deployment = Deployment(
            runtime.backend, borrowed_runtime=runtime, env=dict(runtime.env)
        )
        await deployment.start()
    try:
        if not borrowed:
            await deployment.stop_execution()
            await task.finalize_services(
                trace,
                {name: deployment.target(name) for name in deployment.target_backends},
            )
        config = verifier_runtime(grader, runtime, trace.agent.config.runtime, policy)
        await grade_task(
            grader,
            trace,
            config,
            task.data.timeout,
            policy,
            deployment,
            on_score=on_score,
        )
    except TimeoutError:
        trace.record_timeout("scoring")
        raise
    except Exception as error:
        trace.record_error(error)
        raise
    finally:
        if borrowed:
            await deployment.close()

"""Per-run resolution: what a run may cost, where it runs, and whether the pairing works.

These are the rules that turn "this harness, this taskset, this task" into one runnable
placement — stage timeouts, the task-resolved runtime config, pairing validation, and
the remote-lifetime cap. They live below both consumers: `Agent` (the program surface)
resolves through them per run, and `Environment` (the eval surface) applies the same
rules through its internal Agent — so a run behaves identically however it was reached.
"""

import logging

from pydantic_config import BaseConfig

from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import (
    RuntimeConfig,
    SubprocessConfig,
    runtime_is_local,
)
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset

logger = logging.getLogger(__name__)


class TimeoutConfig(BaseConfig):
    """Framework-enforced wall-clock timeouts per rollout stage, in seconds (None = no
    limit). Each bounds one stage of `Rollout.run`: the taskset's `setup` hook, the harness
    run, the taskset's `finalize` hook, then scoring."""

    setup: float | None = None
    """Max wall-clock for the taskset's `setup` hook (per-task runtime prep)."""
    rollout: float | None = None
    """Max wall-clock for the rollout (the harness run)."""
    finalize: float | None = None
    """Max wall-clock for the taskset's `finalize` hook (post-run work, before scoring)."""
    scoring: float | None = None
    """Max wall-clock for scoring — verify + rewards/metrics."""


def resolve_runtime_config(
    base: RuntimeConfig, task: Task, warned: set[tuple[str, str]] | None = None
) -> RuntimeConfig:
    """Resolve a task's runtime config from a `base`: inject the task's `image` (a task with
    an image must run in a container — refuse subprocess), and apply its `workdir` and
    requested `resources` to the fields the runtime supports. Precedence is cli/toml > task >
    default; a resource the runtime doesn't support warns once (deduped via `warned`). Shared
    by `Agent` (rollouts) and the `validate` entrypoint."""
    config = base
    updates: dict = {}
    if task.image is not None:
        if isinstance(config, SubprocessConfig):
            raise ValueError(
                f"task {task.idx!r} requires image {task.image!r}, but the subprocess "
                "runtime has no container; use the docker or prime runtime"
            )
        updates["image"] = task.image
    workdir_spec = type(config).model_fields.get("workdir")
    if (
        task.workdir is not None
        and workdir_spec is not None
        and getattr(config, "workdir") == workdir_spec.default
    ):
        updates["workdir"] = task.workdir
    for field, value in task.resources.model_dump(exclude_none=True).items():
        spec = type(config).model_fields.get(field)
        if spec is None:
            key = (config.type, field)
            if warned is not None and key not in warned:
                warned.add(key)
                logger.warning(
                    "runtime %r doesn't support resource %r; ignoring it",
                    config.type,
                    field,
                )
        elif (
            getattr(config, field) == spec.default
        ):  # still the default → task may set it
            updates[field] = value
        # else: cli/toml changed it from the default → it wins over the task
    return config.model_copy(update=updates) if updates else config


def validate_pairing(
    harness: Harness, taskset: Taskset, runtime_config: RuntimeConfig
) -> None:
    """Refuse a harness/taskset/runtime combination that cannot work: a taskset whose
    tools (MCP) or user simulator the harness doesn't drive would run to completion with
    the model never seeing them, and a `NEEDS_CONTAINER` taskset's world hooks would run
    on the host under the subprocess runtime. Applied by `Environment` (once, at init)
    and by every `Agent` rollout (against the resolved runtime)."""
    if not harness.SUPPORTS_MCP and type(taskset).tools is not Taskset.tools:
        raise ValueError(
            f"Harness {harness.config.id!r} does not support MCP tools, but taskset "
            f"{taskset.config.id!r} exposes tool servers (MCP). Run it with a harness "
            f"that supports MCP (e.g. the default harness), or use a taskset without tools."
        )
    if not harness.SUPPORTS_USER_SIM and type(taskset).user is not Taskset.user:
        raise ValueError(
            f"Harness {harness.config.id!r} does not drive a user simulator, but taskset "
            f"{taskset.config.id!r} defines one (Taskset.user). Run it with a harness that "
            f"supports user simulation (e.g. the default harness), or use a taskset without one."
        )
    if taskset.NEEDS_CONTAINER and isinstance(runtime_config, SubprocessConfig):
        raise ValueError(
            f"Taskset {taskset.config.id!r} needs a container runtime "
            "(NEEDS_CONTAINER), but this run resolves to the subprocess runtime; "
            "use a docker or prime runtime."
        )


def cap_remote_harness_timeout(
    harness_timeout: float | None, runtime_config: RuntimeConfig, task: Task
) -> float | None:
    """Remote sandboxes have a maximum lifetime of 24 hours: cap the harness timeout
    there (with a warning) so a long run times out cleanly in the framework instead of
    the provider killing the box mid-run."""
    if (
        harness_timeout is not None
        and harness_timeout > 24 * 60 * 60
        and not runtime_is_local(runtime_config)
    ):
        logger.warning(
            "task %r resolves to a %.1f-hour harness timeout, but %s sandboxes have a "
            "maximum lifetime of 24 hours; capping it at 24 hours",
            task.idx,
            harness_timeout / (60 * 60),
            runtime_config.type,
        )
        return 24 * 60 * 60
    return harness_timeout

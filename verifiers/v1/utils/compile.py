"""Task x agent fit: what refuses before any work happens. `Agent.run` applies
these per run, on the task the agent actually receives; `SingleAgentEnv` applies
them at construction where the pairing is statically decidable."""

import logging
from collections.abc import Collection

from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig, runtime_is_local
from verifiers.v1.task import Task

logger = logging.getLogger(__name__)


def resolve_runtime_config(
    base: RuntimeConfig, task: Task, warned: set[tuple[str, str]] | None = None
) -> RuntimeConfig:
    """Resolve a task's runtime config from `base`: inject the task's `image` (an
    image needs a container — refuse subprocess), apply its `workdir` and requested
    `resources` where the runtime supports them. Precedence: cli/toml > task >
    default; an unsupported resource warns once (deduped via `warned`)."""
    config = base
    updates: dict = {}
    if task.data.image is not None:
        if isinstance(config, SubprocessConfig):
            raise ValueError(
                f"task {task.data.idx!r} requires image {task.data.image!r}, but the subprocess "
                "runtime has no container; use the docker or prime runtime"
            )
        updates["image"] = task.data.image
    workdir_spec = type(config).model_fields.get("workdir")
    if (
        task.data.workdir is not None
        and workdir_spec is not None
        and getattr(config, "workdir") == workdir_spec.default
    ):
        updates["workdir"] = task.data.workdir
    for resource, value in task.data.resources.model_dump(exclude_none=True).items():
        spec = type(config).model_fields.get(resource)
        if spec is None:
            key = (config.type, resource)
            if warned is not None and key not in warned:
                warned.add(key)
                logger.warning(
                    "runtime %r doesn't support resource %r; ignoring it",
                    config.type,
                    resource,
                )
        elif (
            getattr(config, resource) == spec.default
        ):  # still the default → task may set it
            updates[resource] = value
        # else: cli/toml changed it from the default → it wins over the task
    return config.model_copy(update=updates) if updates else config


def validate_pairing(
    harness: Harness,
    task_cls: type[Task],
    runtime_config: RuntimeConfig,
    *,
    shared_tools: Collection = (),
) -> None:
    """Reject an impossible harness/task/runtime combination before any work happens.
    Every check reads class-level facts, so a failure holds for every row the task
    class can carry. For `shared_tools` only emptiness matters — declarations and
    live servers alike mean MCP is in play. (Hosting a user is interaction-scoped,
    not task-scoped — `Agent.interaction` checks the harness can resume an exchange.)"""
    if not harness.SUPPORTS_MCP and (task_cls.tools or shared_tools):
        raise ValueError(
            f"Harness {harness.config.id!r} does not support MCP tools, but "
            f"{task_cls.__name__} exposes tool servers (MCP). Run it with a harness that "
            f"supports MCP (e.g. --env.agent.harness.id bash), or use tasks without tools."
        )
    if task_cls.NEEDS_CONTAINER and isinstance(runtime_config, SubprocessConfig):
        raise ValueError(
            f"{task_cls.__name__} needs a container runtime (NEEDS_CONTAINER), but "
            "this run resolves to the subprocess runtime; use "
            "--env.<agent>.harness.runtime.type docker or prime."
        )


def cap_remote_harness_timeout(
    harness_timeout: float | None, runtime_config: RuntimeConfig, task: Task
) -> float | None:
    """Remote sandboxes live at most 24 hours: cap the harness timeout there (with a
    warning) so a long run times out cleanly instead of the provider killing the box
    mid-run."""
    if (
        harness_timeout is not None
        and harness_timeout > 24 * 60 * 60
        and not runtime_is_local(runtime_config)
    ):
        logger.warning(
            "task %r resolves to a %.1f-hour harness timeout, but %s sandboxes have a "
            "maximum lifetime of 24 hours; capping it at 24 hours",
            task.data.idx,
            harness_timeout / (60 * 60),
            runtime_config.type,
        )
        return 24 * 60 * 60
    return harness_timeout

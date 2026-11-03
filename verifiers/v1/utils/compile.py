"""Task x agent fit: what refuses before any work happens. `Agent.run` applies
these per run, on the task the agent actually receives; `SingleAgentEnv` applies
them at construction where the pairing is statically decidable."""

import logging
from collections.abc import Collection

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import (
    RuntimeConfig,
    backend_cls,
)
from verifiers.v1.task import Task
from verifiers.v1.utils.artifacts import validate_artifact_mounts

logger = logging.getLogger(__name__)


def resolve_runtime_config(
    base: RuntimeConfig, task: Task, warned: set[tuple[str, str]] | None = None
) -> RuntimeConfig:
    """Resolve a task's runtime config from `base`: inject the task's `image` (an
    image needs a container — refuse subprocess), apply its network policy, `workdir`,
    and requested `resources` where the runtime supports them. Precedence: cli/toml >
    task > default except that restrictions compose; an unsupported resource warns once
    (deduped via `warned`)."""
    config = base
    updates: dict = {}
    if task.data.image is not None:
        if not backend_cls(config).SUPPORTS_TASK_IMAGES:
            raise ValueError(
                f"task {task.data.idx!r} requires image {task.data.image!r}, but the "
                f"{config.type} runtime cannot run images; use a supported runtime"
            )
        updates["image"] = task.data.image
    workdir_spec = type(config).model_fields.get("workdir")
    if (
        task.data.workdir is not None
        and workdir_spec is not None
        and config.workdir == workdir_spec.default
    ):
        updates["workdir"] = task.data.workdir
    if (
        task.data.agent_user is not None
        and not backend_cls(config).SUPPORTS_USER_SWITCHING
    ):
        raise ValueError(
            f"task {task.data.idx!r} runs its agent as {task.data.agent_user!r}, but "
            f"the {config.type} runtime cannot run as another user; use a supported "
            "runtime"
        )
    if task.data.network.network_restricted:
        if not isinstance(config, NetworkPolicyConfig):
            raise ValueError(
                f"task {task.data.idx!r} requires a network policy, but the "
                f"{config.type} runtime does not enforce network policies; use a "
                "supported runtime"
            )
        config = config.with_task_network_policy(task.data.network)
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
    config = config.model_copy(update=updates) if updates else config
    validate_artifact_mounts(config, (a.source for a in task.data.artifacts))
    return config


def validate_pairing(
    harness: Harness,
    task_cls: type[Task],
    runtime_config: RuntimeConfig,
    *,
    tools: Collection = (),
) -> None:
    """Reject an impossible harness/task/runtime combination before any work happens.
    A failure holds for every row the task class can carry. For `tools` only
    emptiness matters — task-declared and shared servers alike mean MCP is in play.
    (Hosting a user is interaction-scoped, not task-scoped — `Agent.interaction`
    checks the harness can resume an exchange.)"""
    if not harness.SUPPORTS_MCP and tools:
        raise ValueError(
            f"Harness {harness.config.id!r} does not support MCP tools, but the run "
            f"serves some ({task_cls.__name__}'s or the taskset's shared servers). Run "
            f"it with a harness that supports MCP (e.g. --env.agent.harness.id bash), "
            f"or use tasks without tools."
        )
    if not harness.SUPPORTS_SKILLS and harness.config.skills:
        raise ValueError(
            f"Harness {harness.config.id!r} has no native skill support, but "
            "`skills` is set. Run them with a harness whose program discovers "
            "skills (e.g. --env.agent.harness.id claude-code)."
        )
    container = backend_cls(runtime_config).SUPPORTS_TASK_IMAGES
    if harness.NEEDS_CONTAINER and not container:
        raise ValueError(
            f"Harness {harness.config.id!r} needs a container runtime "
            f"(NEEDS_CONTAINER), but this run resolves to the {runtime_config.type} "
            "runtime; use a supported runtime."
        )
    if task_cls.NEEDS_CONTAINER and not container:
        raise ValueError(
            f"{task_cls.__name__} needs a container runtime (NEEDS_CONTAINER), but "
            f"this run resolves to the {runtime_config.type} runtime; use a "
            "supported runtime."
        )

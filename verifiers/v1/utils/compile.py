"""Task x agent fit: what refuses before any work happens. `Agent.run` applies
these per run, on the task the agent actually receives; `SingleAgentEnv` applies
them at construction where the pairing is statically decidable."""

from collections.abc import Collection

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import RuntimeConfig
from verifiers.v1.task import Task
from verifiers.v1.utils.artifacts import validate_artifact_mounts


def resolve_runtime_config(base: RuntimeConfig, task: Task) -> RuntimeConfig:
    """Resolve a task's runtime config from `base`: inject the task's `image`, apply its
    network policy, `workdir`, and requested `resources`. Precedence: cli/toml > task >
    default except that restrictions compose."""
    config = base
    fields = type(config).model_fields
    updates: dict = {}
    if task.data.image is not None:
        updates["image"] = task.data.image
    if task.data.workdir is not None and config.workdir == fields["workdir"].default:
        updates["workdir"] = task.data.workdir
    if task.data.network.network_restricted:
        if not isinstance(config, NetworkPolicyConfig):
            raise ValueError(
                f"task {task.data.idx!r} requires a network policy, but the "
                f"{config.type} runtime does not enforce network policies; use a "
                "supported runtime"
            )
        config = config.with_task_network_policy(task.data.network)
    for resource, value in task.data.resources.model_dump(exclude_none=True).items():
        # Still the default → the task may set it; a cli/toml value wins over the task.
        if getattr(config, resource) == fields[resource].default:
            updates[resource] = value
    config = config.model_copy(update=updates) if updates else config
    validate_artifact_mounts(config, (a.source for a in task.data.artifacts))
    return config


def validate_pairing(
    harness: Harness,
    task_cls: type[Task],
    runtime_config: RuntimeConfig,
    *,
    tools: Collection = (),
    split: bool = False,
) -> None:
    """Reject an impossible harness/task/runtime combination before any work happens.
    A failure holds for every row the task class can carry. For `tools` only
    emptiness matters — task-declared and shared servers alike mean MCP is in play.
    (Hosting a user is interaction-scoped, not task-scoped — `Agent.interaction`
    checks the harness can resume an exchange.)"""
    if split and tools and not harness.SUPPORTS_SPLIT_MCP:
        raise ValueError(
            f"Harness {harness.config.id!r} does not support MCP tools in split deployment"
        )
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

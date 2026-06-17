"""The immutable rollout input.

A `Task` is a frozen pydantic model. Environments subclass it to add typed,
task-specific fields (the reference answer, ground truths, follow-ups, ...) that
then flow — fully typed — through the rollout and into scoring. This replaces v1's
`dict`-subclass `Task` with its hand-rolled `freeze()` and serializability probing.
"""

from typing import TypeVar

from pydantic import ConfigDict

from verifiers.v1.types import Messages, StrictBaseModel


class TaskResources(StrictBaseModel):
    """Runtime resources a task requests (all optional), in Modal's units. Applied to the
    runtime config where the field exists; a field the runtime doesn't support is warned
    about and ignored. Precedence: cli/toml > task > the runtime default (`None` here =
    use the runtime/provider default)."""

    model_config = ConfigDict(frozen=True)

    cpu: float | None = None
    """CPU cores."""
    memory: float | None = None
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "A100:2" (type[:count])."""
    disk: float | None = None
    """Disk in GB (enforced by prime; advisory on docker/modal)."""


class TaskTimeout(StrictBaseModel):
    """Per-task wall-clock timeout overrides (seconds, all optional), one per rollout stage. Each
    merges with the eval's `timeout` (`TimeoutConfig`): cli/toml > this > default (no limit).
    Frozen, like `TaskResources`."""

    model_config = ConfigDict(frozen=True)

    setup: float | None = None
    """The taskset's `setup` hook."""
    harness: float | None = None
    """The harness run."""
    finalize: float | None = None
    """The taskset's `finalize` hook."""
    scoring: float | None = None
    """Verify + rewards/metrics."""


class Task(StrictBaseModel):
    """A single problem to solve. Subclass to add typed task-specific fields."""

    model_config = ConfigDict(frozen=True)

    idx: int
    """Stable integer index of this example within its taskset."""
    name: str | None = None
    """Optional human-readable task name/label (for display/filtering)."""
    description: str | None = None
    """Optional human-readable task description."""
    instruction: str | Messages | None
    """The user message shown to the model (the task's question/framing). Usually a `str`; a
    `Messages` list seeds a full initial conversation (e.g. a user message carrying images) and
    is only accepted by harnesses that set `SUPPORTS_MESSAGE_INSTRUCTION`. Required — set it
    explicitly to `None` to mean the task carries no prompt: the taskset's user simulator
    (`Taskset.user`) then opens the conversation, its first `respond` supplying the initial user
    turn before the model is ever called."""
    system_prompt: str | None = None
    """Optional system prompt. Harnesses that set `APPENDS_SYSTEM_PROMPT` emit it as a real
    system message (or their own mechanism); others prepend it to `instruction` (with a
    warning). See `Harness.resolve_prompt`."""
    image: str | None = None
    """Container image this task needs (e.g. its harbor environment). When set, the
    runtime must be a container (docker/prime): the Environment injects it into the
    runtime config and refuses the subprocess runtime, which has no container."""
    workdir: str | None = None
    """Working directory the harness and scoring run in — the Environment injects it into
    the runtime config's `workdir` (where the runtime supports one). For a containerized
    task whose image puts the working tree at a non-default path (e.g. a SWE row's
    `/workspace/<repo>`)."""
    timeout: TaskTimeout = TaskTimeout()
    """Optional per-task timeout overrides, one per rollout stage (merge with the eval's `timeout`)."""
    resources: TaskResources = TaskResources()
    """Optional runtime resources this task requests (applied where supported)."""


class WireTask(Task):
    """A `Task` that accepts (and preserves) taskset-specific extra fields. Lets a `Trace`
    be typed on the wire — `Trace[WireTask]` — without importing the taskset, since the real
    `Task` subclass's extra fields land in `model_extra` instead of being rejected. A caller
    that imports the taskset upgrades to the real type via `task_type(taskset_id)`."""

    model_config = ConfigDict(extra="allow")


TaskT = TypeVar("TaskT", bound=Task)

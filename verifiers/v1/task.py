"""The immutable rollout input.

A `Task` is a frozen pydantic model. Environments subclass it to add typed,
task-specific fields (the reference answer, ground truths, follow-ups, ...) that
then flow — fully typed — through the rollout and into scoring. This replaces v1's
`dict`-subclass `Task` with its hand-rolled `freeze()` and serializability probing.
"""

from typing import TypeVar

from pydantic import ConfigDict

from verifiers.v1.types import StrictBaseModel


class Resources(StrictBaseModel):
    """Runtime resources a task requests (all optional). Applied to the runtime
    config where the field exists (e.g. `cpu_cores` on prime); a field the runtime
    doesn't support is warned about and ignored. Precedence: cli/toml > task > the
    runtime default (`None` here = use the runtime/provider default)."""

    model_config = ConfigDict(frozen=True)

    cpu_cores: float | None = None
    memory_gb: float | None = None
    gpu_count: int | None = None
    disk_gb: float | None = None


class Task(StrictBaseModel):
    """A single problem to solve. Subclass to add typed task-specific fields."""

    model_config = ConfigDict(frozen=True)

    idx: int
    """Stable integer index of this example within its taskset."""
    name: str | None = None
    """Optional human-readable task name/label (for display/filtering)."""
    description: str | None = None
    """Optional human-readable task description."""
    instruction: str
    """The single user message shown to the model. All framing/instructions are
    baked in here at load time — there is no separate system prompt."""
    image: str | None = None
    """Container image this task needs (e.g. its harbor environment). When set, the
    runtime must be a container (docker/prime): the Environment injects it into the
    runtime config and refuses the subprocess runtime, which has no container."""
    harness_timeout: float | None = None
    """Optional per-task harness timeout (seconds). Merges with the eval's
    `harness_timeout`: cli/toml > this > default (no limit)."""
    scoring_timeout: float | None = None
    """Optional per-task scoring timeout (seconds). Merges with the eval's
    `scoring_timeout`: cli/toml > this > default (no limit)."""
    resources: Resources = Resources()
    """Optional runtime resources this task requests (applied where supported)."""


class WireTask(Task):
    """A `Task` that accepts (and preserves) taskset-specific extra fields. Lets a `Trace`
    be typed on the wire — `Trace[WireTask]` — without importing the taskset, since the real
    `Task` subclass's extra fields land in `model_extra` instead of being rejected. A caller
    that imports the taskset upgrades to the real type via `task_type(taskset_id)`."""

    model_config = ConfigDict(extra="allow")


TaskT = TypeVar("TaskT", bound=Task)

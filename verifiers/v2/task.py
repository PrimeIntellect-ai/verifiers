"""The immutable rollout input.

A `Task` is a frozen pydantic model. Environments subclass it to add typed,
task-specific fields (the reference answer, ground truths, follow-ups, ...) that
then flow — fully typed — through the rollout and into scoring. This replaces v1's
`dict`-subclass `Task` with its hand-rolled `freeze()` and serializability probing.
"""

from typing import TypeVar

from pydantic import ConfigDict

from verifiers.v2.types import StrictBaseModel


class Task(StrictBaseModel):
    """A single problem to solve. Subclass to add typed task-specific fields."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Stable identifier for this example."""
    name: str | None = None
    """Optional human-readable task name (for display/filtering)."""
    description: str | None = None
    """Optional human-readable task description."""
    instruction: str
    """The single user message shown to the model. All framing/instructions are
    baked in here at load time — there is no separate system prompt."""


TaskT = TypeVar("TaskT", bound=Task)

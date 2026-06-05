"""The transcript: the full record of one rollout.

A `Transcript[TaskT]` carries the typed task plus everything produced during a
rollout (conversation, per-turn responses, reward, metrics, timing, error). It is
the canonical full data dump — written to disk (`results.jsonl`) and consumed by
the platform (visualization) and prime-rl (training). Environments subclass it to
add typed scratch/result fields. The rollout mutates it directly; this replaces
v1's 600-line `dict`-subclass `State` and its dual "contract version" machinery.
"""

import time
import traceback
from typing import Generic, TypeVar

from pydantic import Field, computed_field

from verifiers.nano.task import TaskT
from verifiers.nano.types import Messages, Response, StrictBaseModel


class TimeSpan(StrictBaseModel):
    """A start/end wall-clock span with a computed duration in seconds."""

    start: float = 0.0
    end: float = 0.0

    @computed_field
    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start) if self.end else 0.0


class Timing(StrictBaseModel):
    """Wall-clock timing for the phases of a rollout."""

    start: float = Field(default_factory=time.time)
    generation: TimeSpan = Field(default_factory=TimeSpan)
    scoring: TimeSpan = Field(default_factory=TimeSpan)


class Error(StrictBaseModel):
    """A captured error, recorded on the transcript instead of crashing the rollout."""

    type: str
    message: str
    traceback: str


class TurnTokens(StrictBaseModel):
    """Token ids + sampling logprobs for one turn, for training (None when absent)."""

    prompt_ids: list[int] = Field(default_factory=list)
    completion_ids: list[int] = Field(default_factory=list)
    completion_logprobs: list[float] = Field(default_factory=list)


class Turn(StrictBaseModel):
    """One model turn: the prompt sent, the response, and optional token encoding."""

    prompt: Messages
    response: Response
    tokens: TurnTokens | None = None


class Transcript(StrictBaseModel, Generic[TaskT]):
    """The full record of one rollout. Subclass to add typed fields."""

    task: TaskT
    """The (immutable) task being solved — fully typed, flows into scoring."""
    messages: Messages = Field(default_factory=list)
    """The running conversation: the task's user prompt plus everything appended."""
    trajectory: list[Turn] = Field(default_factory=list)

    rewards: dict[str, float] = Field(default_factory=dict)
    """Per-`@reward`-function contributions, with each function's weight applied."""
    metrics: dict[str, float] = Field(default_factory=dict)
    """Per-`@metric`-function values (unweighted; not summed into the reward)."""

    is_completed: bool = False
    is_truncated: bool = False
    stop_condition: str | None = None
    error: Error | None = None
    timing: Timing = Field(default_factory=Timing)

    @computed_field
    @property
    def reward(self) -> float:
        return sum(self.rewards.values())

    @property
    def has_error(self) -> bool:
        return self.error is not None

    def stop(self, condition: str = "done") -> None:
        self.is_completed = True
        if self.stop_condition is None:
            self.stop_condition = condition

    def capture_error(self, error: Exception) -> None:
        """Record a caught error (with traceback) and stop the rollout."""
        self.error = Error(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )
        self.stop("error")


TranscriptT = TypeVar("TranscriptT", bound=Transcript)  # type: ignore[type-arg]

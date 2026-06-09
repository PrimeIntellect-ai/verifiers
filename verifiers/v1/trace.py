"""The trace: the full record of one rollout.

A `Trace[TaskT]` carries the typed task plus everything produced during a
rollout (conversation, per-turn responses, reward, metrics, timing, error). It is
the canonical full data dump — written to disk (`results.jsonl`) and consumed by
the platform (visualization) and prime-rl (training). Environments subclass it to
add typed scratch/result fields. The rollout mutates it directly; this replaces
v1's 600-line `dict`-subclass `State` and its dual "contract version" machinery.
"""

import logging
import time
import traceback
import uuid
from collections.abc import Mapping
from typing import Generic, TypeVar

from pydantic import Field, PrivateAttr, computed_field

from verifiers.v1 import branching
from verifiers.v1.task import TaskT
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    Response,
    StrictBaseModel,
    ToolMessage,
    TurnTokens,
)

logger = logging.getLogger(__name__)


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
    """A captured error, recorded on the trace instead of crashing the rollout."""

    type: str
    message: str
    traceback: str | None = (
        None  # synthetic errors (cancels, empty trajectory) have none
    )


class Turn(StrictBaseModel):
    """One model turn: the prompt sent, the response, and optional token encoding."""

    prompt: Messages
    response: Response
    tokens: TurnTokens | None = None

    @property
    def num_prompt_tokens(self) -> int:
        if self.tokens and self.tokens.prompt_ids:
            return len(self.tokens.prompt_ids)
        return self.response.usage.prompt_tokens if self.response.usage else 0

    @property
    def num_completion_tokens(self) -> int:
        if self.tokens and self.tokens.completion_ids:
            return len(self.tokens.completion_ids)
        return self.response.usage.completion_tokens if self.response.usage else 0


class Branch(StrictBaseModel):
    """A linear run of turns whose context grew without being rewritten. A trajectory
    that compacts splits into several branches (see `branching`); a linear one is a
    single branch. `messages` is the branch's full conversation — its last turn's
    prompt plus that turn's response (the last prompt already holds the branch's
    earlier turns)."""

    index: int
    turns: list[Turn]

    @computed_field
    @property
    def num_turns(self) -> int:
        """Model turns in this branch."""
        return len(self.turns)

    @property
    def prompt_len(self) -> int:
        """Input context size: this branch's final-turn prompt (the full last context)."""
        return self.turns[-1].num_prompt_tokens if self.turns else 0

    @property
    def completion_len(self) -> int:
        """All assistant-generated tokens across this branch's turns."""
        return sum(turn.num_completion_tokens for turn in self.turns)

    @property
    def total_tokens(self) -> int:
        """This branch's final-turn sequence length (prompt + completion)."""
        last = self.turns[-1] if self.turns else None
        return last.num_prompt_tokens + last.num_completion_tokens if last else 0

    @property
    def messages(self) -> Messages:
        last = self.turns[-1]
        return [*last.prompt, last.response.message]


class Trace(StrictBaseModel, Generic[TaskT]):
    """The full record of one rollout. Subclass to add typed fields."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique id for this rollout, auto-generated per trace."""
    task: TaskT
    """The (immutable) task being solved — fully typed, flows into scoring."""
    trajectory: list[Turn] = Field(default_factory=list)
    """Every model turn in order — the ground truth."""

    rewards: dict[str, float] = Field(default_factory=dict)
    """Per-`@reward`-function contributions, with each function's weight applied."""
    metrics: dict[str, float] = Field(default_factory=dict)
    """Per-`@metric`-function values (unweighted; not summed into the reward)."""

    is_completed: bool = False
    stop_condition: str | None = None
    error: Error | None = None
    timing: Timing = Field(default_factory=Timing)

    _branch_cache: dict[int, list[list[int]]] = PrivateAttr(default_factory=dict)
    """Branch segmentation (index-groups, no turn copies) cached by trajectory length —
    the trajectory only grows, so its length is a sufficient invalidation key."""

    @computed_field
    @property
    def reward(self) -> float:
        return sum(self.rewards.values())

    @property
    def has_error(self) -> bool:
        return self.error is not None

    @property
    def last_turn(self) -> Turn | None:
        """The final model turn, or None for an empty trajectory."""
        return self.trajectory[-1] if self.trajectory else None

    @property
    def prompt_len(self) -> int:
        """Total input context summed over branches (each branch's final-turn prompt) —
        the trajectory yields one training sample per branch, so totals aggregate them."""
        return sum(branch.prompt_len for branch in self.branches)

    @property
    def completion_len(self) -> int:
        """Total assistant-generated (completion) tokens summed over branches — every token
        the model produced (reasoning included), so it can exceed the final context size."""
        return sum(branch.completion_len for branch in self.branches)

    @property
    def total_tokens(self) -> int:
        """Total sequence length summed over branches (each branch's final-turn prompt +
        completion) — used for token batching."""
        return sum(branch.total_tokens for branch in self.branches)

    @property
    def has_response(self) -> bool:
        """Whether the final assistant turn produced non-empty content."""
        last = self.last_turn
        return bool(last and last.response.message.content)

    def _branch_groups(self) -> list[list[int]]:
        """Branch index-groups for the current trajectory, recomputed only when a turn
        is added (the trajectory is append-only, so its length is a sufficient key).
        Caches the groups (ints), never copies of the turns."""
        n = len(self.trajectory)
        if n not in self._branch_cache:
            self._branch_cache = {n: branching.segment(self.trajectory)}
        return self._branch_cache[n]

    @computed_field
    @property
    def branches(self) -> list[Branch]:
        """The trajectory segmented into linear branches (see `branching`): one branch
        when linear, several under compaction or subagents. The structured view of what
        the harness saw, replacing a flat message list. Branches hold turn references, not
        copies."""
        return [
            Branch(index=i, turns=[self.trajectory[j] for j in group])
            for i, group in enumerate(self._branch_groups())
        ]

    @computed_field
    @property
    def num_branches(self) -> int:
        """How many branches the trajectory has (1 = linear; >1 = compaction/subagents)."""
        return len(self._branch_groups())

    @computed_field
    @property
    def num_turns(self) -> int:
        """Total model turns across the whole trajectory (all branches); per-branch
        counts are on each `Branch.num_turns`."""
        return len(self.trajectory)

    @computed_field
    @property
    def is_truncated(self) -> bool:
        """Whether the rollout was cut off by a budget/limit rather than ending on its
        own terms: the framework halted it (`max_turns` / `harness_timeout`), or the
        final turn hit the token cap (`finish_reason == "length"`)."""
        if self.stop_condition in ("max_turns", "harness_timeout"):
            return True
        last = self.last_turn
        return bool(last and last.response.finish_reason == "length")

    @property
    def assistant_messages(self) -> list[AssistantMessage]:
        """Every model response, in order — one per turn, branch-independent."""
        return [turn.response.message for turn in self.trajectory]

    @property
    def tool_messages(self) -> list[ToolMessage]:
        """The tool results in the latest full context — the last turn's prompt. (For a
        linear rollout that's every tool result; computed straight off the trajectory,
        like `assistant_messages`, with no branch reconstruction.)"""
        last = self.trajectory[-1].prompt if self.trajectory else []
        return [m for m in last if isinstance(m, ToolMessage)]

    def record_metric(self, name: str, value: float) -> None:
        """Record a single `@metric` result under `name`. Warns if it overrides an
        existing metric (a name collision, e.g. an harness and a task metric sharing
        a name) — last writer wins, but loudly."""
        if name in self.metrics:
            logger.warning(
                "metric %r overridden: %s -> %s", name, self.metrics[name], value
            )
        self.metrics[name] = float(value)

    def record_metrics(self, values: "Mapping[str, float]") -> None:
        """Merge a family of `@metric` results (so one method can report several,
        e.g. an harness's depth/calls/tokens). Each key warns on override as above."""
        for name, value in values.items():
            self.record_metric(name, value)

    def record_reward(self, name: str, value: float, weight: float = 1.0) -> None:
        """Record a `@reward`/`@group_reward` contribution under `name` (weight
        applied; summed into `reward`). Warns on override — a per-rollout reward and
        a group reward sharing a name would otherwise silently clobber."""
        contribution = float(value) * float(weight)
        if name in self.rewards:
            logger.warning(
                "reward %r overridden: %s -> %s", name, self.rewards[name], contribution
            )
        self.rewards[name] = contribution

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

    def to_wire(self) -> dict:
        """Dump for the wire, dropping the derived (computed) fields at every level —
        the top-level ones (reward, branches, num_branches, num_turns) and the per-span
        timing durations. A strict `Trace` can't round-trip them as input, and the
        consumer recomputes them, so we avoid re-running branching + duplicating the
        trajectory on every reply. The full `model_dump` (with derived fields) is what
        gets written to disk."""
        exclude: dict = {field: True for field in type(self).model_computed_fields}
        exclude["timing"] = {
            "generation": {"duration": True},
            "scoring": {"duration": True},
        }
        return self.model_dump(mode="json", exclude=exclude)


TraceT = TypeVar("TraceT", bound=Trace)  # type: ignore[type-arg]

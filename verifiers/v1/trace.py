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

from verifiers.v1 import graph
from verifiers.v1.graph import MessageNode
from verifiers.v1.task import TaskT
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    StrictBaseModel,
    ToolMessage,
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


class Branch(StrictBaseModel):
    """A linear run of messages whose context grew without being rewritten — a root→leaf
    path in the message graph. A conversation that compacts (or runs subagents) splits into
    several branches; a linear one is a single branch. `messages` is the full conversation;
    one training sample is built per branch."""

    index: int
    nodes: list[MessageNode]

    @computed_field
    @property
    def num_turns(self) -> int:
        """Model turns (assistant messages) in this branch."""
        return sum(1 for n in self.nodes if isinstance(n.message, AssistantMessage))

    @property
    def messages(self) -> Messages:
        """The branch's full conversation, in order."""
        return [n.message for n in self.nodes]

    @property
    def completion_len(self) -> int:
        """All assistant-generated (model-sampled) tokens across this branch."""
        return sum(sum(n.mask) for n in self.nodes)

    @property
    def total_tokens(self) -> int:
        """This branch's full sequence length (final-turn prompt + every completion)."""
        return sum(len(n.token_ids) for n in self.nodes)

    @property
    def prompt_len(self) -> int:
        """Input context size: the final-turn prompt = full sequence minus the last completion."""
        last_completion = next(
            (sum(n.mask) for n in reversed(self.nodes) if any(n.mask)), 0
        )
        return self.total_tokens - last_completion


class Trace(StrictBaseModel, Generic[TaskT]):
    """The full record of one rollout. Subclass to add typed fields."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique id for this rollout, auto-generated per trace."""
    task: TaskT
    """The (immutable) task being solved — fully typed, flows into scoring."""
    nodes: list[MessageNode] = Field(default_factory=list)
    """The message graph — one node per distinct message, linked to its predecessor (see
    `graph`). The ground truth; `trajectory` and `branches` are views over it. Stores each
    message once, so size is linear (not quadratic) in turns."""

    rewards: dict[str, float] = Field(default_factory=dict)
    """Per-`@reward`-function contributions, with each function's weight applied."""
    metrics: dict[str, float] = Field(default_factory=dict)
    """Per-`@metric`-function values (unweighted; not summed into the reward)."""

    is_completed: bool = False
    stop_condition: str | None = None
    errors: list[Error] = Field(default_factory=list)
    """Every error captured across attempts, oldest first (more than one only when the
    rollout was retried). `error` exposes the most recent."""
    timing: Timing = Field(default_factory=Timing)

    _head_index: dict = PrivateAttr(default_factory=dict)
    """`(parent, msg_hash) -> node_id` for the graph builder (`graph.add_turn`); rebuilt
    lazily from `nodes` after deserialization."""

    @computed_field
    @property
    def reward(self) -> float:
        return sum(self.rewards.values())

    @computed_field
    @property
    def error(self) -> Error | None:
        """The most recent captured error (the rest are earlier retry attempts)."""
        return self.errors[-1] if self.errors else None

    @property
    def has_error(self) -> bool:
        return bool(self.errors)

    def _last_assistant(self) -> "MessageNode | None":
        """The most recent assistant node, or None for a trace with no responses."""
        return next(
            (
                n
                for n in reversed(self.nodes)
                if isinstance(n.message, AssistantMessage)
            ),
            None,
        )

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
        """Whether the most recent assistant message produced non-empty content."""
        last = self._last_assistant()
        return bool(last and last.message.content)

    @property
    def branches(self) -> list[Branch]:
        """The conversation segmented into linear branches — a view over the graph: each
        leaf's root→leaf path is a branch (one when linear, several under compaction or
        subagents). Branching falls out of the walk; see `graph.branches_from_nodes`."""
        return graph.branches_from_nodes(self)

    @property
    def num_branches(self) -> int:
        """How many branches (1 = linear; >1 = compaction/subagents)."""
        return len(graph.leaves(self))

    @property
    def num_turns(self) -> int:
        """Total model turns (assistant nodes) across all branches."""
        return sum(1 for n in self.nodes if isinstance(n.message, AssistantMessage))

    @computed_field
    @property
    def is_truncated(self) -> bool:
        """Whether the rollout was cut off by a budget/limit rather than ending on its
        own terms: the framework halted it (`max_turns`, a token budget, or
        `harness_timeout`), or the final turn hit the token cap (`finish_reason ==
        "length"`)."""
        if self.stop_condition in (
            "max_turns",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
            "harness_timeout",
        ):
            return True
        last = self._last_assistant()
        return bool(last and last.finish_reason == "length")

    @property
    def assistant_messages(self) -> list[AssistantMessage]:
        """Every model response, in order — one per turn, branch-independent."""
        return [
            n.message for n in self.nodes if isinstance(n.message, AssistantMessage)
        ]

    @property
    def tool_messages(self) -> list[ToolMessage]:
        """The tool results in the latest full context — the main (last) branch's
        conversation. For a linear rollout that's every tool result."""
        branches = self.branches
        messages = branches[-1].messages if branches else []
        return [m for m in messages if isinstance(m, ToolMessage)]

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
        self.errors.append(
            Error(
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
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

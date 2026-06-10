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

from pydantic import Field, PrivateAttr, computed_field, model_validator

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


def _lcp_len(a: list, b: list) -> int:
    """Length of the longest common prefix of two sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _delta_encode_turns(turns: list[dict]) -> None:
    """Compress `prompt_ids` and cumulative multimodal `mm_items` to per-turn deltas, in
    place. Consecutive turns in a branch share a growing prefix — the prompt restates the
    whole conversation each turn — so a turn stores only the tail beyond the running
    sequence and records the kept-prefix length in `pk` (prompt) / `mk` (per-modality mm).
    Avoids the quadratic blow-up of re-sending the full prompt every turn.
    `_delta_decode_turns` inverts this. The kept length is the actual longest common
    prefix, so forks (no shared prefix) just store the full tail — always lossless."""
    run_ids: list[int] = []
    run_hashes: dict[str, list[str]] = {}
    for turn in turns:
        tok = turn.get("tokens")
        if not isinstance(tok, dict):
            run_ids, run_hashes = [], {}
            continue
        pid = tok.get("prompt_ids") or []
        keep = _lcp_len(run_ids, pid)
        tok["prompt_ids"] = pid[keep:]
        tok["pk"] = keep
        run_ids = pid + (tok.get("completion_ids") or [])

        mm = tok.get("multi_modal_data")
        if isinstance(mm, dict) and mm.get("mm_items"):
            mk: dict[str, int] = {}
            hashes_map = mm.get("mm_hashes") or {}
            for modality, items in list(mm["mm_items"].items()):
                hashes = hashes_map.get(modality) or []
                mkeep = _lcp_len(run_hashes.get(modality, []), hashes) if hashes else 0
                mm["mm_items"][modality] = items[mkeep:]
                if modality in hashes_map:
                    hashes_map[modality] = hashes[mkeep:]
                mk[modality] = mkeep
                run_hashes[modality] = hashes
            tok["mk"] = mk


def _delta_decode_turns(turns: list[dict]) -> None:
    """Inverse of `_delta_encode_turns`: rebuild full `prompt_ids` / cumulative `mm_items`
    from the per-turn deltas and drop the `pk`/`mk` markers. A no-op on inputs without
    markers (full, uncompressed dicts), so it's safe to run on any trace input."""
    run_ids: list[int] = []
    run_items: dict[str, list] = {}
    run_hashes: dict[str, list] = {}
    for turn in turns:
        if not isinstance(turn, dict):
            run_ids, run_items, run_hashes = [], {}, {}
            continue
        tok = turn.get("tokens")
        if not isinstance(tok, dict):
            run_ids, run_items, run_hashes = [], {}, {}
            continue
        if "pk" in tok:
            keep = tok.pop("pk")
            tok["prompt_ids"] = run_ids[:keep] + (tok.get("prompt_ids") or [])
            run_ids = tok["prompt_ids"] + (tok.get("completion_ids") or [])
        mk = tok.pop("mk", None)
        mm = tok.get("multi_modal_data")
        if mk is not None and isinstance(mm, dict):
            hashes_map = mm.get("mm_hashes")
            for modality, mkeep in mk.items():
                items = run_items.get(modality, [])[:mkeep] + (
                    mm["mm_items"].get(modality) or []
                )
                mm["mm_items"][modality] = items
                run_items[modality] = items
                if isinstance(hashes_map, dict) and modality in hashes_map:
                    h = run_hashes.get(modality, [])[:mkeep] + hashes_map[modality]
                    hashes_map[modality] = h
                    run_hashes[modality] = h


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
    errors: list[Error] = Field(default_factory=list)
    """Every error captured across attempts, oldest first (more than one only when the
    rollout was retried). `error` exposes the most recent."""
    timing: Timing = Field(default_factory=Timing)

    _branch_cache: dict[int, list[list[int]]] = PrivateAttr(default_factory=dict)
    """Branch segmentation (index-groups, no turn copies) cached by trajectory length —
    the trajectory only grows, so its length is a sufficient invalidation key."""

    @model_validator(mode="before")
    @classmethod
    def _expand_wire_delta(cls, data):
        """Restore delta-compressed `to_wire` payloads (see `_delta_encode_turns`) before
        strict field validation; a no-op on normal/full inputs."""
        if isinstance(data, dict) and isinstance(data.get("trajectory"), list):
            _delta_decode_turns(data["trajectory"])
        return data

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
        gets written to disk.

        Within a branch, consecutive turns share a growing prefix, so `prompt_ids` (and
        cumulative multimodal `mm_items`) are delta-encoded against the running sequence —
        `_expand_wire_delta` restores them on the way back in. This is what keeps the wire
        from growing quadratically with turn count."""
        exclude: dict = {field: True for field in type(self).model_computed_fields}
        exclude["timing"] = {
            "generation": {"duration": True},
            "scoring": {"duration": True},
        }
        dump = self.model_dump(mode="json", exclude=exclude)
        _delta_encode_turns(dump.get("trajectory") or [])
        return dump


TraceT = TypeVar("TraceT", bound=Trace)  # type: ignore[type-arg]

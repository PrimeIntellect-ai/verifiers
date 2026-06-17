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
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import Field, PrivateAttr, computed_field
from renderers.base import MultiModalData

from verifiers.v1 import graph
from verifiers.v1.graph import MessageNode
from verifiers.v1.state import State, StateT, WireState
from verifiers.v1.task import TaskT, WireTask
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
    """Wall-clock timing for the phases of a rollout: provisioning the runtime + serving
    (`setup`), the harness driving the conversation (`generation`), post-run work before
    scoring (`finalize`), then scoring."""

    start: float = Field(default_factory=time.time)
    setup: TimeSpan = Field(default_factory=TimeSpan)
    generation: TimeSpan = Field(default_factory=TimeSpan)
    finalize: TimeSpan = Field(default_factory=TimeSpan)
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
        """Model turns (sampled responses) in this branch — prompt-supplied assistant
        messages don't count."""
        return sum(1 for n in self.nodes if n.sampled)

    @property
    def messages(self) -> Messages:
        """The branch's full conversation, in order."""
        return [n.message for n in self.nodes]

    @property
    def token_ids(self) -> list[int]:
        """The branch's full token sequence — every node's tokens concatenated in order
        (final-turn prompt + every completion). The training sample's input ids."""
        return [t for node in self.nodes for t in node.token_ids]

    @property
    def sampled_mask(self) -> list[bool]:
        """Per-token trainable flag aligned to `token_ids`: True for the model-sampled
        (completion) tokens, False for prompt/template scaffold."""
        return [m for node in self.nodes for m in node.mask]

    @property
    def logprobs(self) -> list[float]:
        """Per-token sampling logprobs aligned to `token_ids` — the node logprobs spread onto
        their sampled positions, 0.0 on every non-sampled token."""
        out: list[float] = []
        for node in self.nodes:
            li = 0
            for sampled in node.mask:
                if sampled:
                    out.append(node.logprobs[li] if li < len(node.logprobs) else 0.0)
                    li += 1
                else:
                    out.append(0.0)
        return out

    @property
    def multi_modal_data(self) -> MultiModalData | None:
        """The branch's multimodal sidecar — every node's images concatenated in path (token)
        order. None when the branch has no images. Drives the training `mm_kwargs` (the renderer
        items per modality); the per-token `mm_token_type_ids` come from the token ids, so no
        placeholder offsets are carried. Never persisted (node mm is transient)."""
        merged = MultiModalData()
        found = False
        for node in self.nodes:
            mmd = node.multi_modal_data
            if mmd is None or mmd.is_empty():
                continue
            found = True
            for modality, items in mmd.mm_items.items():
                merged.mm_items.setdefault(modality, []).extend(items)
            for modality, hashes in mmd.mm_hashes.items():
                merged.mm_hashes.setdefault(modality, []).extend(hashes)
        return merged if found else None

    @property
    def routed_experts(self) -> np.ndarray | None:
        """The branch's MoE expert-routing array — every node's expert ids concatenated in path
        (token) order, uint8 `[len(token_ids), layers, top_k]` aligned 1:1 with `token_ids`.
        All-or-nothing: returns None unless every token-bearing node carries routing and the
        concatenation matches the branch length (partial routing can't be safely aligned, so the
        trainer skips replay). None when the rollout ran without `enable_return_routed_experts`."""
        nodes = [n for n in self.nodes if n.token_ids]
        if not nodes or any(n.routed_experts is None for n in nodes):
            return None
        merged = np.concatenate([n.routed_experts for n in nodes], axis=0)
        total = sum(len(n.token_ids) for n in nodes)
        return merged if merged.shape[0] == total else None

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

    @property
    def num_prompt_tokens(self) -> int:
        """Final-turn input tokens from provider-reported usage — a fallback for display when
        the endpoint returns no token ids (so `prompt_len` is 0); 0 if no usage was reported."""
        last = next(
            (n.usage for n in reversed(self.nodes) if n.usage is not None), None
        )
        return last.prompt_tokens if last else 0

    @property
    def num_completion_tokens(self) -> int:
        """All completion tokens across the branch from provider-reported usage — a fallback for
        display when the endpoint returns no token ids; 0 if no usage was reported."""
        return sum(n.usage.completion_tokens for n in self.nodes if n.usage is not None)


class Trace(StrictBaseModel, Generic[TaskT, StateT]):
    """The full record of one rollout. Subclass to add typed fields."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique id for this rollout, auto-generated per trace."""
    task: TaskT
    """The (immutable) task being solved — fully typed, flows into scoring."""
    nodes: list[MessageNode] = Field(default_factory=list)
    """The message graph — one node per distinct message, linked to its predecessor (see
    `graph`). The ground truth; `branches` is a view over it. Stores each message once, so
    size is linear (not quadratic) in turns."""

    rewards: dict[str, float] = Field(default_factory=dict)
    """Per-`@reward`-function contributions, with each function's weight applied."""
    metrics: dict[str, float] = Field(default_factory=dict)
    """Per-`@metric`-function values (unweighted; not summed into the reward)."""
    info: dict[str, Any] = Field(default_factory=dict)
    """Free-form, JSON-serializable scratch space for taskset-specific metadata that is neither
    a reward nor a metric — anything an author wants to scrape off the live runtime and persist
    with the trace (captured logs, command output, container/runtime state, artifact paths).
    Populate it from the runtime in `finalize` (or a `@reward`/`@metric`) by assigning into the
    dict (`trace.info["build_log"] = ...`); it round-trips through the wire to `results.jsonl`.
    Use `metrics` for numbers that aggregate, this for everything else. Values must be
    JSON-serializable — a non-serializable value fails the trace dump rather than being dropped."""
    state: StateT = Field(default_factory=State, exclude=True)
    """Transient per-rollout runtime state (see `verifiers.v1.state.State`): shared with the tool/user
    servers as `self.state` (synced over the interception server) and read+written by scoring. Runtime
    scratch (counters, game progress, end-of-trajectory flags) — excluded from every dump (`model_dump`
    + `to_wire`), unlike `info` which persists. Type it via `Taskset[Task, Config, MyState]`; defaults
    to the base `State`."""

    is_completed: bool = False
    stop_condition: str | None = None
    errors: list[Error] = Field(default_factory=list)
    """Every error captured across attempts, oldest first (more than one only when the
    rollout was retried). `error` exposes the most recent."""
    timing: Timing = Field(default_factory=Timing)

    _head_index: dict = PrivateAttr(default_factory=dict)
    """`(parent, msg_hash) -> node_id` for the graph builder (`graph.prepare_turn` / `commit`);
    rebuilt lazily from `nodes` after deserialization."""

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
        """The most recent sampled (model-produced) node, or None for a trace with no
        responses."""
        return next((n for n in reversed(self.nodes) if n.sampled), None)

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
        subagents). Branching falls out of walking each leaf's parents back to its root."""
        branches: list[Branch] = []
        for i, leaf in enumerate(graph.leaves(self)):
            path: list[int] = []
            nid: int | None = leaf
            while nid is not None:
                path.append(nid)
                nid = self.nodes[nid].parent
            path.reverse()
            branches.append(Branch(index=i, nodes=[self.nodes[n] for n in path]))
        return branches

    @property
    def num_branches(self) -> int:
        """How many branches (1 = linear; >1 = compaction/subagents)."""
        return len(graph.leaves(self))

    @property
    def num_turns(self) -> int:
        """Total model turns (sampled responses) across all branches — prompt-supplied
        assistant messages don't count."""
        return sum(1 for n in self.nodes if n.sampled)

    @computed_field
    @property
    def is_truncated(self) -> bool:
        """Whether the rollout was cut off by a budget/limit rather than ending on its
        own terms: the framework halted it (`max_turns`, a token budget, or
        `harness_timeout`), the prompt outgrew the model's context window
        (`context_length`), or the final turn hit the token cap (`finish_reason ==
        "length"`)."""
        if self.stop_condition in (
            "max_turns",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
            "context_length",
            "harness_timeout",
        ):
            return True
        last = self._last_assistant()
        return bool(last and last.finish_reason == "length")

    @property
    def assistant_messages(self) -> list[AssistantMessage]:
        """Every model response, in order — one per turn, branch-independent. Excludes
        prompt-supplied assistant messages (`sampled` is the provenance signal)."""
        return [
            n.message
            for n in self.nodes
            if n.sampled and isinstance(n.message, AssistantMessage)
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
        gets written to disk.

        Dumped in `mode="python"` (not `"json"`) so per-node numpy arrays survive as raw bytes
        in their `__nd__` dicts — the env-server packs the result with a numpy-aware msgpack
        encoder so the arrays ride the `bin` wire untouched. `mode="json"` would coerce the
        bytes to str."""
        exclude: dict = {field: True for field in type(self).model_computed_fields}
        # Drop each timing span's computed `duration` — derived per `TimeSpan` field of
        # `Timing` (setup/generation/scoring, and any future span) so none leaks to the wire.
        exclude["timing"] = {
            name: {f: True for f in TimeSpan.model_computed_fields}
            for name, info in Timing.model_fields.items()
            if info.annotation is TimeSpan
        }
        return self.model_dump(mode="python", exclude=exclude)

    @classmethod
    def from_wire(cls, data: dict) -> "Trace":
        """Load a trace dumped by `to_wire` (or the full `model_dump` on disk) back into a model —
        the inverse of `to_wire`. Drops the derived (computed) fields a strict `Trace` rejects as
        input: the top-level ones (`reward`, `error`, `is_truncated`) and each timing span's
        `duration`. Everything else round-trips, branches included (each node keeps its `parent`).

        Use the permissive `WireTask` / `WireState` to load a dump without importing the originating
        taskset: `Trace[WireTask, WireState].from_wire(d)`, i.e. `WireTrace.from_wire(d)`."""
        data = dict(data)
        for field in cls.model_computed_fields:
            data.pop(field, None)
        timing = data.get("timing")
        if isinstance(timing, dict):
            data["timing"] = {
                name: (
                    {
                        k: v
                        for k, v in span.items()
                        if k not in TimeSpan.model_computed_fields
                    }
                    if isinstance(span, dict)
                    else span
                )
                for name, span in timing.items()
            }
        return cls.model_validate(data)


TraceT = TypeVar("TraceT", bound=Trace)  # type: ignore[type-arg]

WireTrace = Trace[WireTask, WireState]
"""A `Trace` typed for loading a dump without the originating taskset — permissive task + state
(extras preserved on `task`, ignored-but-fine on `state`, which is never dumped). Pair with
`from_wire`: `WireTrace.from_wire(json.loads(line))`."""

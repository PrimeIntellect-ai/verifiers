from __future__ import annotations

import logging
import time
import traceback
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
from pydantic import Field, PrivateAttr
from renderers.base import MultiModalData

if TYPE_CHECKING:
    from verifiers.v1.judge import JudgeResponse

from verifiers.v1 import graph
from verifiers.v1.errors import ProviderError
from verifiers.v1.graph import MessageNode
from verifiers.v1.runtimes import RuntimeInfo
from verifiers.v1.state import State, StateT
from verifiers.v1.task import DataT, WireTaskData
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    StrictBaseModel,
    Tool,
    ToolMessage,
    Usage,
    content_text,
)

logger = logging.getLogger(__name__)


class TimeSpan(StrictBaseModel):
    """Wall-clock timestamps with a derived, non-serialized duration in seconds."""

    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start) if self.end else 0.0


class Timing(StrictBaseModel):
    start: float = Field(default_factory=time.time)
    setup: TimeSpan = Field(default_factory=TimeSpan)
    generation: TimeSpan = Field(default_factory=TimeSpan)
    finalize: TimeSpan = Field(default_factory=TimeSpan)
    scoring: TimeSpan = Field(default_factory=TimeSpan)


class Error(StrictBaseModel):
    type: str
    message: str
    traceback: str | None = None


class Branch(StrictBaseModel):
    """A root-to-leaf graph path; each branch becomes one training sample."""

    index: int
    nodes: list[MessageNode]

    @property
    def num_turns(self) -> int:
        """Model-sampled turns; prompt-supplied assistant messages do not count."""
        return sum(1 for n in self.nodes if n.sampled)

    @property
    def messages(self) -> Messages:
        return [n.message for n in self.nodes]

    @property
    def token_ids(self) -> list[int]:
        """Training input IDs formed by concatenating node token spans."""
        tokens: list[int] = []
        # Extend node spans in bulk to avoid per-token Python work.
        for node in self.nodes:
            tokens.extend(node.token_ids)
        return tokens

    @property
    def sampled_mask(self) -> list[bool]:
        """Per-token trainable flag aligned to `token_ids`: True for the model-sampled
        (completion) tokens, False for prompt/template scaffold."""
        mask: list[bool] = []
        for node in self.nodes:
            mask.extend(node.mask)
        return mask

    @property
    def logprobs(self) -> list[float]:
        """Per-token sampling logprobs aligned to `token_ids` — the node logprobs spread onto
        their sampled positions, 0.0 on every non-sampled token."""
        out: list[float] = []
        for node in self.nodes:
            mask = node.mask
            sampled = sum(mask) if node.logprobs else 0
            # Bulk-fill the canonical unsampled-prefix/sampled-suffix layout.
            if not sampled or all(mask[-sampled:]):
                out += [0.0] * (len(mask) - sampled) + node.logprobs[:sampled]
                out += [0.0] * max(0, sampled - len(node.logprobs))
                continue
            li = 0
            for sampled in mask:
                if sampled:
                    out.append(node.logprobs[li] if li < len(node.logprobs) else 0.0)
                    li += 1
                else:
                    out.append(0.0)
        return out

    @property
    def multi_modal_data(self) -> MultiModalData | None:
        """Node image data concatenated in token order for training; never persisted."""
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
        """uint8 `[tokens, layers, top_k]` routing; partial data returns None."""
        nodes = [n for n in self.nodes if n.token_ids]
        if not nodes or any(n.routed_experts is None for n in nodes):
            return None
        merged = np.concatenate([n.routed_experts for n in nodes], axis=0)
        total = sum(len(n.token_ids) for n in nodes)
        return merged if merged.shape[0] == total else None

    @property
    def num_total_tokens(self) -> int:
        return sum(len(n.token_ids) for n in self.nodes)

    @property
    def usage(self) -> Usage | None:
        return Usage.aggregate(n.usage for n in self.nodes if n.usage is not None)

    @property
    def num_input_tokens(self) -> int:
        """Final-turn prompt size, falling back to provider usage without token IDs."""
        last_completion = next(
            (sum(n.mask) for n in reversed(self.nodes) if any(n.mask)), 0
        )
        token_len = self.num_total_tokens - last_completion
        if token_len:
            return token_len
        last = next(
            (n.usage for n in reversed(self.nodes) if n.usage is not None), None
        )
        return last.input_tokens if last else 0

    @property
    def num_output_tokens(self) -> int:
        """Sampled tokens, falling back to provider usage without token IDs."""
        token_len = sum(sum(n.mask) for n in self.nodes)
        if token_len:
            return token_len
        usage = self.usage
        return usage.completion_tokens if usage else 0


_NODE_DUMP_EXCLUDE: dict = {
    "nodes": {"__all__": {"multi_modal_data", "routed_experts"}}
}
"""Raw tensor fields kept on the msgpack wire but excluded from JSON records."""


class TraceTask(StrictBaseModel, Generic[DataT]):
    """The task as recorded on the trace: the row (`data`, the wire half — fully typed,
    flows into scoring) plus the Task class name that produced the rollout (`type`) —
    provenance, so a bare trace is self-describing: a `from_trace` implementer or an
    offline re-scorer can tell which behavior class made it without the run's config
    (replay warns when it disagrees with the taskset's declared type). Only data and a
    name ride the wire — behavior still re-attaches by construction."""

    type: str
    """The Task class name (`type(task).__name__`), resolution stays anchored to the
    taskset id like everything else."""
    data: DataT
    """The (immutable) row being solved."""


class Trace(StrictBaseModel, Generic[DataT, StateT]):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique id for this rollout, auto-generated per trace."""
    task: TraceTask[DataT]
    """The task being solved: its class name (`task.type`) + its row (`task.data`)."""
    runtime: RuntimeInfo | None = None
    """The runtime's full config plus its provisioned resource ID."""
    nodes: list[MessageNode] = Field(default_factory=list)
    """The message graph; branches are derived views and storage stays linear in turns."""
    tools: list[Tool] | None = None
    """The tools advertised to the model, recorded when an intercepted turn commits (last
    committed turn wins) — never from a refused/failed request the model never saw. The full
    advertised list (not just tools called), so tool-use SFT can re-render the exact prompt;
    a trace-level snapshot: mid-rollout changes collapse to the last set the model saw."""

    rewards: dict[str, float] = Field(default_factory=dict)
    """Weighted contributions from task rewards, group rewards, and judges."""
    metrics: dict[str, float] = Field(default_factory=dict)
    """Unweighted metrics from tasks, harnesses, and judges."""
    info: dict[str, Any] = Field(default_factory=dict)
    """Persistent JSON scratch space for task metadata that is not a reward or metric."""
    state: StateT = Field(default_factory=State, exclude=True)
    """Transient state shared with servers and scoring; excluded from every dump."""

    extra_usage: list[Usage] = Field(default_factory=list)
    """Usage from judges and other calls outside the agent's message graph."""

    is_completed: bool = False
    stop_condition: str | None = None
    errors: list[Error] = Field(default_factory=list)
    """Every error captured across attempts, oldest first (more than one only when the
    rollout was retried). `error` exposes the most recent."""
    timing: Timing = Field(default_factory=Timing)

    _head_index: dict = PrivateAttr(default_factory=dict)
    """`(parent, msg_hash) -> node_id` for the graph builder (`graph.prepare_turn` / `commit`);
    rebuilt lazily from `nodes` after deserialization."""
    _num_turns_cache: tuple[int, int] = PrivateAttr(default=(0, 0))
    """`(counted nodes, sampled turns)` for the append-only message graph."""

    @property
    def reward(self) -> float:
        return sum(self.rewards.values())

    @property
    def error(self) -> Error | None:
        return self.errors[-1] if self.errors else None

    @property
    def has_error(self) -> bool:
        return bool(self.errors)

    def _last_assistant(self) -> MessageNode | None:
        """Most recent model-produced node, ignoring prompt-supplied assistant messages."""
        return next((n for n in reversed(self.nodes) if n.sampled), None)

    @property
    def num_input_tokens(self) -> int:
        """Final-turn prompt sizes summed across training branches."""
        return sum(branch.num_input_tokens for branch in self.branches)

    @property
    def num_output_tokens(self) -> int:
        """Model-sampled tokens summed across training branches."""
        return sum(branch.num_output_tokens for branch in self.branches)

    @property
    def num_total_tokens(self) -> int:
        """Sequence lengths summed across training branches for token batching."""
        return sum(branch.num_total_tokens for branch in self.branches)

    @property
    def usage(self) -> Usage | None:
        """Provider-reported usage summed once per actual model call in this rollout."""
        return Usage.aggregate(n.usage for n in self.nodes if n.usage is not None)

    @property
    def has_response(self) -> bool:
        last = self._last_assistant()
        return bool(last and last.message.content)

    @property
    def branches(self) -> list[Branch]:
        """One root-to-leaf path per graph leaf."""
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
        return len(graph.leaves(self))

    @property
    def num_turns(self) -> int:
        """Total model turns (sampled responses) across all branches — prompt-supplied
        assistant messages don't count."""
        counted_nodes, num_turns = self._num_turns_cache
        node_count = len(self.nodes)
        # Graph commits append nodes, so count only the unseen suffix between reads.
        # If a caller shrinks the list, rebuild the count from the remaining nodes.
        if node_count == counted_nodes:
            return num_turns
        if node_count < counted_nodes:
            counted_nodes, num_turns = 0, 0
        while counted_nodes < node_count:
            num_turns += self.nodes[counted_nodes].sampled
            counted_nodes += 1
        self._num_turns_cache = (node_count, num_turns)
        return num_turns

    @property
    def is_truncated(self) -> bool:
        """True for framework limits or a length-finished final response."""
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
    def last_reply(self) -> str:
        msgs = self.assistant_messages
        return (msgs[-1].content or "").strip() if msgs else ""

    @property
    def transcript(self) -> str:
        """Final-branch text and tool calls for judges; images and reasoning are omitted."""
        branches = self.branches
        blocks: list[str] = []
        for message in branches[-1].messages if branches else []:
            lines = [f"[{message.role}]"]
            if isinstance(message, AssistantMessage):
                if message.content:
                    lines.append(message.content)
                lines.extend(
                    f"[tool_call {call.name}({call.arguments})]"
                    for call in message.tool_calls or []
                )
            else:
                if isinstance(message, ToolMessage) and message.name:
                    lines[0] = f"[{message.role} {message.name}]"
                if text := content_text(message.content):
                    lines.append(text)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @property
    def tool_messages(self) -> list[ToolMessage]:
        """The tool results in the latest full context — the main (last) branch's
        conversation. For a linear rollout that's every tool result."""
        branches = self.branches
        messages = branches[-1].messages if branches else []
        return [m for m in messages if isinstance(m, ToolMessage)]

    def record_metric(self, name: str, value: float) -> None:
        if name in self.metrics:
            logger.warning(
                "metric %r overridden: %s -> %s", name, self.metrics[name], value
            )
        self.metrics[name] = float(value)

    def record_metrics(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self.record_metric(name, value)

    def record_judge(self, response: JudgeResponse) -> None:
        self.info.setdefault("judge", []).append(response.model_dump())
        if response.usage is not None:
            self.extra_usage.append(response.usage)

    def record_reward(self, name: str, value: float, weight: float = 1.0) -> None:
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
        self.errors.append(
            Error(
                type=type(error).__name__,
                message=str(error),
                # Provider errors already carry the actionable upstream diagnostic.
                # Keep full tracebacks for every other failure.
                traceback=None
                if isinstance(error, ProviderError)
                else traceback.format_exc(),
            )
        )
        self.stop("error")

    def to_record(self) -> dict[str, Any]:
        """JSON record without raw tensors, which remain available on the msgpack wire."""
        return self.model_dump(mode="json", exclude=_NODE_DUMP_EXCLUDE)


TraceT = TypeVar("TraceT", bound=Trace)  # type: ignore[type-arg]

WireTrace = Trace[WireTaskData]
"""Trace loader that preserves unknown task fields in `task.model_extra`."""

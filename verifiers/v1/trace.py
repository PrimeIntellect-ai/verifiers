from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from renderers.base import MultiModalData
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from verifiers.v1.judge import JudgeResponse

from verifiers.v1 import graph
from verifiers.v1.configs.agent import AgentConfig, WireAgentConfig
from verifiers.v1.errors import ProviderError
from verifiers.v1.graph import MessageNode
from verifiers.v1.runtimes import RuntimeInfo
from verifiers.v1.state import State, StateT
from verifiers.v1.task import DataT, WireTaskData
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    KeptTokens,
    Messages,
    Sampling,
    Tool,
    ToolMessage,
    Usage,
    content_text,
)

TRACE_VERSION = 1
"""Current version of the trace schema."""


EXCLUDE_FIELDS: dict = {
    "nodes": {
        "__all__": {
            "multi_modal_data",
            "routed_experts",
            "kept_tokens",
        }
    }
}
"""Raw tensor fields kept on the msgpack wire but excluded from disk serialization."""


class TimeSpan(BaseModel):
    """Wall-clock timestamps with a derived, non-serialized duration in seconds."""

    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start) if self.end else 0.0


class TimeSplit(BaseModel):
    """Records a measured duration in seconds."""

    duration: float = 0.0


class AgentSpan(TimeSpan):
    model: TimeSplit = Field(default_factory=TimeSplit)
    harness: TimeSplit = Field(default_factory=TimeSplit)


class Timing(BaseModel):
    start: float = Field(default_factory=time.time)
    boot: TimeSpan = Field(default_factory=TimeSpan)
    setup: TimeSpan = Field(default_factory=TimeSpan)
    agent: AgentSpan = Field(default_factory=AgentSpan)
    finalize: TimeSpan = Field(default_factory=TimeSpan)
    scoring: TimeSpan = Field(default_factory=TimeSpan)


class Error(BaseModel):
    type: str
    message: str
    status_code: int | None = None
    traceback: str | None = None


class VersionInfo(BaseModel):
    version: str
    commit: str | None = None


def _current_build() -> VersionInfo:
    # Lazy: `verifiers/__init__` (and its v0 surface) must not load with this module.
    from verifiers import __version__
    from verifiers.v1.utils.version import verifiers_commit

    return VersionInfo(version=__version__, commit=verifiers_commit())


AgentConfigT = TypeVar("AgentConfigT", bound=AgentConfig, default=AgentConfig)


class AgentInfo(BaseModel, Generic[AgentConfigT]):
    config: AgentConfigT
    """The resolved config that rebuilds the agent (`Agent(trace.agent.config)`)."""
    runtime: RuntimeInfo | None = None
    """The box the rollout ran in; None until provisioning."""
    name: str = "agent"
    """The env agent that produced this trace (the config field name, e.g. `solver`)."""
    trainable: bool = True
    """Whether this trace's tokens train the run's policy."""


class TraceTask(BaseModel, Generic[DataT]):
    """The task as recorded on the trace, self-describing without the run's config."""

    type: str
    """The Task class name (`type(task).__name__`)."""
    data: DataT
    """The (immutable) row being solved."""


class Reward(BaseModel):
    score: float
    weight: float = 1.0

    @property
    def value(self) -> float:
        return self.score * self.weight


class ModelCall(BaseModel):
    """A model call, automatically recorded at intercept time."""

    node: int | None = None
    """Index into `Trace.nodes` of the assistant node this call committed."""
    model: str | None = None
    """The model requested from the provider."""
    sampling: Sampling | None = None
    """The call's effective sampling settings (may differ from trace-level sampling)."""
    endpoint: str | None = None
    """The provider endpoint path the request went to (e.g. `/chat/completions`)."""
    finish_reason: FinishReason = None
    """Why the model stopped, normalized (`stop` / `length` / `tool_calls`)."""
    usage: Usage | None = None
    """Provider-reported token usage for this exchange, cache reads included."""
    time: TimeSpan = Field(default_factory=TimeSpan)
    """Wall-clock span from sending the request to the fully received response."""
    error: Error | None = None
    """The failure that ended this call, coupled to the exchange that caused it."""


class Branch(BaseModel):
    """A root-to-leaf graph path; each branch becomes one training sample."""

    index: int
    nodes: list[MessageNode]
    calls: list[ModelCall] = Field(default_factory=list)

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

    def spread(
        self, values: Callable[[MessageNode], list[float] | None]
    ) -> list[float]:
        """A per-sampled-token node field widened to `token_ids`: each node's values land on its
        sampled positions, 0.0 everywhere else. A node holding nothing contributes zeros."""
        out: list[float] = []
        for node in self.nodes:
            mask, node_values = node.mask, values(node) or []
            sampled = sum(mask) if node_values else 0
            # Bulk-fill the canonical unsampled-prefix/sampled-suffix layout.
            if not sampled or all(mask[-sampled:]):
                out += [0.0] * (len(mask) - sampled) + node_values[:sampled]
                out += [0.0] * max(0, sampled - len(node_values))
                continue
            li = 0
            for is_sampled in mask:
                if is_sampled:
                    out.append(node_values[li] if li < len(node_values) else 0.0)
                    li += 1
                else:
                    out.append(0.0)
        return out

    @property
    def logprobs(self) -> list[float]:
        """Per-token sampling logprobs aligned to `token_ids` — the node logprobs spread onto
        their sampled positions, 0.0 on every non-sampled token."""
        return self.spread(lambda node: node.logprobs)

    @property
    def advantages(self) -> list[float] | None:
        """Per-token credit aligned to `token_ids`, spread like `logprobs` — or `None` when no node
        on the path was ever assigned any, which a branch of zeros would otherwise be
        indistinguishable from. A partially assigned path spreads, the unassigned nodes reading
        0.0."""
        if all(node.advantages is None for node in self.nodes):
            return None
        return self.spread(lambda node: node.advantages)

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
    def kept_tokens(self) -> KeptTokens | None:
        """int32 kept-set `counts` aligned 1:1 with `token_ids` plus flat `ids` in
        position order; partial data scatters as 0 counts, no data returns None."""
        if all(n.kept_tokens is None for n in self.nodes):
            return None
        # `_attribute_kept_tokens` validates counts/ids against the node's sampled
        # tokens before setting the field, so this is a straight scatter+concat
        # (a corrupted node would fail loudly on the scatter shape mismatch).
        ids_parts: list[np.ndarray] = []
        counts_parts: list[np.ndarray] = []
        for node in self.nodes:
            counts = np.zeros(len(node.mask), dtype=np.int32)
            if node.kept_tokens is not None and len(node.kept_tokens.counts):
                counts[np.nonzero(node.mask)[0]] = node.kept_tokens.counts
                ids_parts.append(node.kept_tokens.ids)
            counts_parts.append(counts)
        ids = (
            np.concatenate(ids_parts).astype(np.int32, copy=False)
            if ids_parts
            else np.zeros(0, dtype=np.int32)
        )
        return KeptTokens(ids=ids, counts=np.concatenate(counts_parts))

    @property
    def usage(self) -> Usage | None:
        return Usage.aggregate(c.usage for c in self.calls if c.usage is not None)

    @property
    def last_usage(self) -> Usage | None:
        return next(
            (c.usage for c in reversed(self.calls) if c.usage is not None), None
        )

    @property
    def num_total_tokens(self) -> int:
        last = self.last_usage
        return last.total_tokens if last is not None else 0

    @property
    def num_output_tokens(self) -> int:
        usage = self.usage
        return usage.completion_tokens if usage is not None else 0

    @property
    def num_input_tokens(self) -> int:
        return self.num_total_tokens - self.num_output_tokens


class Trace(BaseModel, Generic[DataT, StateT, AgentConfigT]):
    version: int = TRACE_VERSION
    """The trace schema this trace serializes as."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique ID for this trace, auto-generated."""
    verifiers: VersionInfo = Field(default_factory=_current_build)
    """The verifiers version that produced this trace."""
    task: TraceTask[DataT]
    """The task data that seeded this trace."""
    agent: AgentInfo[AgentConfigT]
    """The agent (harness x model x runtime) that produced this trace."""
    tools: list[Tool] = Field(default_factory=list)
    """The tools advertised to the agent, automatically recorded from last intercepted turn."""

    nodes: list[MessageNode] = Field(default_factory=list)
    """The message graph; branches are derived views and storage stays linear in turns."""
    calls: list[ModelCall] = Field(default_factory=list)
    """Every model call; automatically recorded at intercept time + linked into `nodes`."""

    rewards: dict[str, Reward | None] = Field(default_factory=dict)
    """Named, weighted rewards; `None` means scoring didn't run (e.g. because of a
    preceding error)."""
    metrics: dict[str, float | None] = Field(default_factory=dict)
    """Unweighted, named metrics; `None` as in `rewards`."""
    info: dict[str, Any] = Field(default_factory=dict)
    """Scratch space for task-specific metadata."""
    state: StateT = Field(default_factory=State, exclude=True)
    """Runtime (possibly, non-serializable) state shared across runtimes; excluded from serialization."""

    extra_usage: list[Usage] = Field(default_factory=list)
    """Usage from judges and other calls outside the agent's message graph."""

    is_completed: bool = False
    """Whether the trace completed."""
    ok: bool = False
    """Whether the trace completed successfully."""
    stop_condition: str | None = None
    """What stopped the trace."""
    errors: list[Error] = Field(default_factory=list)
    """Every error captured across attempts, oldest to newest."""
    timing: Timing = Field(default_factory=Timing)

    _head_index: dict = PrivateAttr(default_factory=dict)
    """`(parent, msg_hash) -> node_id` for the graph builder."""

    @property
    def reward(self) -> float:
        return sum(r.value for r in self.rewards.values() if r is not None)

    @property
    def has_error(self) -> bool:
        return not self.ok

    @property
    def num_input_tokens(self) -> int:
        """Fed-in tokens (system + user + tool), counted once, summed across branches."""
        return sum(branch.num_input_tokens for branch in self.branches)

    @property
    def num_output_tokens(self) -> int:
        """Model-generated tokens across all turns, summed across branches."""
        return sum(branch.num_output_tokens for branch in self.branches)

    @property
    def num_total_tokens(self) -> int:
        """Final sequence lengths (last prompt + completion) summed across branches."""
        return sum(branch.num_total_tokens for branch in self.branches)

    @property
    def usage(self) -> Usage | None:
        """Provider-reported usage summed once per actual model call in this rollout."""
        return Usage.aggregate(c.usage for c in self.calls if c.usage is not None)

    @property
    def branches(self) -> list[Branch]:
        """One root-to-leaf path per graph leaf, its calls attached in path order."""
        by_node = {c.node: c for c in self.calls if c.node is not None}
        branches: list[Branch] = []
        for i, leaf in enumerate(graph.leaves(self)):
            path: list[int] = []
            nid: int | None = leaf
            while nid is not None:
                path.append(nid)
                nid = self.nodes[nid].parent
            path.reverse()
            branches.append(
                Branch(
                    index=i,
                    nodes=[self.nodes[n] for n in path],
                    calls=[by_node[n] for n in path if n in by_node],
                )
            )
        return branches

    @property
    def num_branches(self) -> int:
        return len(graph.leaves(self))

    @property
    def num_turns(self) -> int:
        return sum(1 for n in self.nodes if n.sampled)

    @property
    def is_truncated(self) -> bool:
        """True for framework limits or a length-finished final response."""
        if self.stop_condition in (
            "max_turns",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
            "context_length",
        ):
            return True
        last = next((c for c in reversed(self.calls) if c.error is None), None)
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
        """Every tool result, in order — branch-independent."""
        return [n.message for n in self.nodes if isinstance(n.message, ToolMessage)]

    @property
    def last_reply(self) -> str:
        """The last recorded model response, in text format."""
        msgs = self.assistant_messages
        return (msgs[-1].content or "").strip() if msgs else ""

    @property
    def last_error(self) -> Error | None:
        """The last error captured across attempts."""
        return self.errors[-1] if self.errors else None

    @property
    def transcript(self) -> str:
        """Text transcript of the last branch's messages; tool outputs are omitted."""
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

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)

    def record_metrics(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self.record_metric(name, value)

    def record_reward(self, name: str, value: float, weight: float = 1.0) -> None:
        reward = Reward(score=float(value), weight=float(weight))
        self.rewards[name] = reward

    def record_judge(self, response: JudgeResponse) -> None:
        self.info.setdefault("judge", []).append(response.model_dump())
        if response.usage is not None:
            self.extra_usage.append(response.usage)

    def stop(self, condition: str) -> None:
        """Stop the trace, optionally with a stop condition."""
        self.is_completed = True
        if self.stop_condition is None:
            self.stop_condition = condition

    def split_agent_time(self) -> None:
        """Split the agent span into model and harness time."""
        span = self.timing.agent
        if not span.end:
            return
        model = sum(call.time.duration for call in self.calls)
        span.model.duration = min(model, span.duration)
        span.harness.duration = span.duration - span.model.duration

    def record_error(self, error: Exception) -> None:
        """Record an error, and stop the trace as failed."""
        self.errors.append(
            Error(
                type=type(error).__name__,
                message=str(error),
                status_code=getattr(error, "status_code", None),
                # Provider errors already carry the actionable upstream diagnostic.
                # Keep full tracebacks for every other failure.
                traceback=None
                if isinstance(error, ProviderError)
                else traceback.format_exc(),
            )
        )
        self.ok = False
        self.stop("error")

    def to_record(self) -> dict[str, Any]:
        """JSON record without raw tensors, which remain available on the msgpack wire."""
        return self.model_dump(mode="json", exclude=EXCLUDE_FIELDS)


WireTrace = Trace[WireTaskData, State, WireAgentConfig]
"""Record loader for consumers without the run's packages (e.g. a trainer):
task fields survive in `task.model_extra`, and the agent config parses loose
(`WireAgentConfig` — no harness plugin resolution). A bare `Trace` is the
strict read: the harness narrows by id, so its package must be importable."""

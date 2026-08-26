from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, field_serializer
from renderers.base import MultiModalData
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from verifiers.v1.judge import JudgeResponse

from verifiers.v1 import graph
from verifiers.v1.configs.agent import AgentConfig, WireAgentConfig
from verifiers.v1.errors import ProviderError
from verifiers.v1.graph import MessageNode
from verifiers.v1.lineage import CallLineage, LineageManifest
from verifiers.v1.runtimes import RuntimeInfo
from verifiers.v1.state import State, StateT
from verifiers.v1.task import DataT, WireTaskData
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    KeptTokens,
    Message,
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
    key: str | None = None
    """Stable task identity (``Task.key``); defaults to the content hash unless overridden."""
    hash: str | None = None
    """Content hash of ``data`` (``Task.hash``)."""


class InterceptRecord(BaseModel):
    handler: str


class Reward(BaseModel):
    score: float
    weight: float = 1.0

    @property
    def value(self) -> float:
        return self.score * self.weight


class PolicyEvent(BaseModel):
    """A framework policy decision applied to a model call before inference."""

    code: str
    """Stable machine-readable reason for the decision."""
    paths: list[str]
    """Value-free native request paths affected by the decision."""


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
    policy: PolicyEvent | None = None
    """Policy mediation applied to the request before this call."""
    lineage: CallLineage | None = None
    """Exact recursive-session provenance supplied by the harness, when available."""


def min_new_input_tokens(calls: Iterable[ModelCall]) -> Iterator[tuple[ModelCall, int]]:
    """Per-call lower bound on newly fed-in tokens: each call's prompt beyond the
    previous call's final sequence, clamped at zero. Exact while the history is
    append-only; tokens the engine drops between calls (stripped reasoning, truncated
    history) register as an undercount of that call's new input rather than going
    negative."""
    prev_total = 0
    for call in calls:
        if call.usage is None:
            continue
        yield call, max(0, call.usage.input_tokens - prev_total)
        prev_total = call.usage.total_tokens


class Branch(BaseModel):
    """A root-to-leaf graph path; each branch becomes one training sample."""

    index: int
    nodes: list[MessageNode]
    calls: list[ModelCall] = Field(default_factory=list)
    mm_token_type_id_map: dict[int, int] = Field(default_factory=dict)
    """The trace's `mm_token_type_id_map`, carried so `mm_token_type_ids` is self-contained."""

    @property
    def messages(self) -> Messages:
        return [n.message for n in self.nodes]

    @property
    def session_ids(self) -> tuple[str, ...]:
        """Recursive sessions represented on this physical message path, in call order."""
        return tuple(
            dict.fromkeys(
                call.lineage.session_id
                for call in self.calls
                if call.lineage is not None
            )
        )

    @property
    def context_ids(self) -> tuple[str, ...]:
        """Context epochs represented on this physical message path, in call order."""
        return tuple(
            dict.fromkeys(
                call.lineage.context_id
                for call in self.calls
                if call.lineage is not None
            )
        )

    @property
    def compaction_ids(self) -> tuple[str, ...]:
        """Compactions correlated with calls on this physical message path."""
        return tuple(
            dict.fromkeys(
                call.lineage.compaction_id
                for call in self.calls
                if call.lineage is not None and call.lineage.compaction_id is not None
            )
        )

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
    def reference_logprobs(self) -> list[float] | None:
        """Reference-model logprobs aligned to `token_ids`, or None when unscored."""
        if all(node.reference_logprobs is None for node in self.nodes):
            return None
        return self.spread(lambda node: node.reference_logprobs)

    def loss_weights(self, name: str) -> list[float] | None:
        """One named loss-weight stream aligned to `token_ids`, or None when absent."""
        if all(
            node.loss_weights is None or name not in node.loss_weights
            for node in self.nodes
        ):
            return None
        weights: list[float] = []
        for node in self.nodes:
            node_weights = (node.loss_weights or {}).get(name)
            if node_weights is None:
                weights.extend([0.0] * len(node.token_ids))
                continue
            if len(node_weights) != len(node.token_ids):
                raise ValueError(
                    f"loss weight stream {name!r} must align with node token_ids: "
                    f"got {len(node_weights)}, expected {len(node.token_ids)}"
                )
            weights.extend(node_weights)
        return weights

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
    def mm_token_type_ids(self) -> list[int] | None:
        """Per-token modality markers aligned to `token_ids` (0 = text, 1 = image
        placeholder, 2 = video placeholder), driving the trainer's vision-encoder
        slicing; None for branches carrying no multimodal data."""
        if self.multi_modal_data is None:
            return None
        mapping = self.mm_token_type_id_map
        return [mapping.get(t, 0) for t in self.token_ids]

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
        """Fed-in tokens (system + user + tool), counted once; a lower bound whenever
        the engine drops tokens between calls."""
        return sum(increment for _, increment in min_new_input_tokens(self.calls))


@dataclass(frozen=True)
class Conversation:
    """Derived recursive-session view; never duplicated in serialized traces."""

    session_id: str
    parent_session_id: str | None
    depth: int
    context_ids: tuple[str, ...]
    calls: tuple[ModelCall, ...]
    branch_indexes: tuple[int, ...]

    @property
    def is_subagent(self) -> bool:
        return self.parent_session_id is not None


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
    lineage: LineageManifest | None = None
    """Exact recursive-session manifest, when the harness publishes one."""
    mm_token_type_id_map: dict[int, int] = Field(default_factory=dict)
    """Special-token id -> modality marker (1 = image placeholder, 2 = video placeholder)
    from the renderer that tokenized this trace, stamped at turn commit. Applied to a
    branch's `token_ids` by `Branch.mm_token_type_ids`; empty for text-only renderers."""
    request_rewrites: list[InterceptRecord] = Field(default_factory=list)
    """Request changes made by `@intercept`, in execution order."""
    response_rewrites: list[InterceptRecord] = Field(default_factory=list)
    """Response changes made by `@intercept`, in execution order."""

    rewards: dict[str, Reward | None] = Field(default_factory=dict)
    """Named, weighted rewards; `None` means scoring didn't run (e.g. because of a
    preceding error)."""
    metrics: dict[str, float | None] = Field(default_factory=dict)
    """Unweighted, named metrics; `None` as in `rewards`."""
    info: dict[str, Any] = Field(default_factory=dict)
    """Scratch space for task-specific metadata."""
    root_reply: str | None = None
    """The root agent's response when a trace contains descendant branches."""
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

    @field_serializer("mm_token_type_id_map")
    def serialize_mm_token_type_id_map(self, mapping: dict[int, int]) -> dict[str, int]:
        """The wire unpacks with msgpack's `strict_map_key`, which forbids int map keys —
        dump str keys; validation coerces them back to int."""
        return {str(k): v for k, v in mapping.items()}

    @property
    def reward(self) -> float:
        return sum(r.value for r in self.rewards.values() if r is not None)

    @property
    def has_error(self) -> bool:
        return not self.ok

    @property
    def num_input_tokens(self) -> int:
        """Fed-in tokens (system + user + tool), counted once across all branches —
        calls on shared branch prefixes contribute once, not once per branch."""
        seen: set[int] = set()
        total = 0
        for branch in self.branches:
            # A call's predecessors are fixed by the graph, so its increment is the
            # same in every branch containing it; keep the first occurrence.
            for call, increment in min_new_input_tokens(branch.calls):
                if id(call) not in seen:
                    seen.add(id(call))
                    total += increment
        return total

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
                    mm_token_type_id_map=self.mm_token_type_id_map,
                )
            )
        return branches

    @property
    def conversations(self) -> list[Conversation]:
        """Calls grouped by exact recursive session, independent of graph compaction forks.

        Before the final manifest arrives, complete per-call headers are sufficient to expose
        groups already observed.  Once present, manifest creation order also includes sessions
        and contexts that made no model call.
        """
        branches = self.branches
        calls_by_session: dict[str, list[ModelCall]] = {}
        observed_order: list[str] = []
        observed_parent: dict[str, str | None] = {}
        observed_depth: dict[str, int] = {}
        observed_contexts: dict[str, list[str]] = {}
        for call in self.calls:
            item = call.lineage
            if item is None:
                continue
            if item.session_id not in calls_by_session:
                calls_by_session[item.session_id] = []
                observed_order.append(item.session_id)
                observed_parent[item.session_id] = item.parent_session_id
                observed_depth[item.session_id] = item.depth
                observed_contexts[item.session_id] = []
            elif (
                observed_parent[item.session_id] != item.parent_session_id
                or observed_depth[item.session_id] != item.depth
            ):
                raise ValueError(
                    f"inconsistent call lineage for session {item.session_id!r}"
                )
            calls_by_session[item.session_id].append(call)
            if item.context_id not in observed_contexts[item.session_id]:
                observed_contexts[item.session_id].append(item.context_id)

        if self.lineage is None:
            session_rows = [
                (
                    session_id,
                    observed_parent[session_id],
                    observed_depth[session_id],
                    tuple(observed_contexts[session_id]),
                )
                for session_id in observed_order
            ]
        else:
            contexts_by_session: dict[str, list[str]] = {
                session.session_id: [] for session in self.lineage.sessions
            }
            for context in self.lineage.contexts:
                contexts_by_session[context.session_id].append(context.context_id)
            session_rows = [
                (
                    session.session_id,
                    session.parent_session_id,
                    session.depth,
                    tuple(contexts_by_session[session.session_id]),
                )
                for session in self.lineage.sessions
            ]

        conversations: list[Conversation] = []
        for session_id, parent_id, depth, context_ids in session_rows:
            calls = tuple(calls_by_session.get(session_id, []))
            node_ids = {call.node for call in calls if call.node is not None}
            branch_indexes = tuple(
                branch.index
                for branch in branches
                if node_ids.intersection(
                    call.node for call in branch.calls if call.node is not None
                )
            )
            conversations.append(
                Conversation(
                    session_id=session_id,
                    parent_session_id=parent_id,
                    depth=depth,
                    context_ids=context_ids,
                    calls=calls,
                    branch_indexes=branch_indexes,
                )
            )
        return conversations

    @property
    def branches_by_session(self) -> dict[str, list[Branch]]:
        """Physical training branches touched by each exact recursive session."""
        branches = self.branches
        return {
            conversation.session_id: [branches[i] for i in conversation.branch_indexes]
            for conversation in self.conversations
        }

    def reconcile_lineage(self, manifest: LineageManifest) -> None:
        """Validate an ACP lineage snapshot against call headers and attach it.

        Nothing is inferred from the message graph: every recorded call must have a complete
        lineage envelope and a matching request in the snapshot.
        """
        roots = [
            session
            for session in manifest.sessions
            if session.parent_session_id is None
        ]
        if len(roots) != 1 or roots[0].session_id != self.id:
            raise ValueError("RLM lineage root session does not match the rollout")

        sessions = {session.session_id: session for session in manifest.sessions}
        contexts = {context.context_id: context for context in manifest.contexts}
        requests = {request.request_id: request for request in manifest.requests}
        for index, call in enumerate(self.calls):
            item = call.lineage
            if item is None:
                raise ValueError(f"model call {index} has no RLM lineage headers")
            session = sessions.get(item.session_id)
            context = contexts.get(item.context_id)
            request = requests.get(item.request_id)
            if session is None or (
                session.parent_session_id != item.parent_session_id
                or session.depth != item.depth
            ):
                raise ValueError(
                    f"model call {item.request_id!r} does not match its lineage session"
                )
            if context is None or (
                context.session_id != item.session_id
                or context.previous_context_id != item.previous_context_id
                or context.transition != item.transition
            ):
                raise ValueError(
                    f"model call {item.request_id!r} does not match its lineage context"
                )
            if request is None or (
                request.session_id != item.session_id
                or request.context_id != item.context_id
                or request.compaction_id != item.compaction_id
            ):
                raise ValueError(
                    f"model call {item.request_id!r} does not match its lineage request"
                )
        self.lineage = manifest

    @property
    def messages(self) -> Messages:
        """Messages on the final branch."""
        branches = self.branches
        return branches[-1].messages if branches else []

    @property
    def last_message(self) -> Message:
        """The most recently stored message."""
        if not self.nodes:
            raise ValueError("trace has no messages")
        return self.nodes[-1].message

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
        """The root agent's response, falling back to the last assistant message."""
        if self.root_reply is not None:
            return self.root_reply.strip()
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
                else "".join(traceback.format_exception(error)),
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

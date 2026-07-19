from __future__ import annotations

import logging
import time
import traceback
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, TypeVar

import numpy as np
from pydantic import Field, PrivateAttr
from renderers.base import MultiModalData

if TYPE_CHECKING:
    from verifiers.v1.judge import JudgeResponse

from verifiers.v1 import graph
from verifiers.v1.errors import ProviderError
from verifiers.v1.graph import MessageNode
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.runtimes import RuntimeInfo
from verifiers.v1.state import State, StateT
from verifiers.v1.task import DataT, WireTaskData
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    KeptTokens,
    Messages,
    Sampling,
    SamplingConfig,
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


class TimeSplit(StrictBaseModel):
    """A span's share attributed to one side of a split: disjoint sub-intervals summed
    into a duration, so there is no single start/end. Serialized, unlike a span's
    derived `duration`, since it cannot be recomputed from two timestamps."""

    duration: float = 0.0


class GenerationSpan(TimeSpan):
    """The generation span plus its split into time inside model calls (`model`) vs.
    outside them (`harness`: harness logic, tools, user simulation). Stamped from the
    recorded model calls' spans by `Trace.split_generation` when the span closes, with
    `model.duration + harness.duration == duration` — concurrent calls (subagent forks)
    are clamped to the span, saturating the model share."""

    model: TimeSplit = Field(default_factory=TimeSplit)
    harness: TimeSplit = Field(default_factory=TimeSplit)


class Timing(StrictBaseModel):
    start: float = Field(default_factory=time.time)
    boot: TimeSpan = Field(default_factory=TimeSpan)
    setup: TimeSpan = Field(default_factory=TimeSpan)
    generation: GenerationSpan = Field(default_factory=GenerationSpan)
    finalize: TimeSpan = Field(default_factory=TimeSpan)
    scoring: TimeSpan = Field(default_factory=TimeSpan)


class Error(StrictBaseModel):
    type: str
    message: str
    status_code: int | None = None
    """The upstream HTTP status a provider failure surfaced (the provider's own, or one
    chosen for a transport fault); None when the failure carried no HTTP exchange."""
    traceback: str | None = None


class ModelCall(StrictBaseModel):
    """One provider exchange behind a sampled turn; its conversation is the linked
    node's root-to-self path, never repeated here."""

    node: int | None = None
    """Index into `Trace.nodes` of the assistant node this call committed — the link into
    the message graph (the call's conversation is that node's root-to-self path). None for
    a call that committed no turn (see `error`)."""
    model: str | None = None
    """The model requested from the provider. The rollout's model override makes this
    `agent.model` on every call; recorded per call because it is cheap and provable."""
    sampling: Sampling | None = None
    """The call's effective settings, scraped off the wire request by the dialect's
    `sampling_fields` whitelist — the eval-imposed knobs plus whatever the harness set
    that the eval left alone (`seed`, `tool_choice`, `response_format`, ... as extras)."""
    endpoint: str | None = None
    """The provider endpoint path the request went to (e.g. `/chat/completions`) — says
    which wire dialect the exchange spoke."""
    finish_reason: FinishReason = None
    """Why the model stopped, normalized (`stop` / `length` / `tool_calls`); None for a
    failed call or an unrecognized provider reason."""
    usage: Usage | None = None
    """Provider-reported token usage for this exchange, cache reads included; None for
    a failed call."""
    time: TimeSpan = Field(default_factory=TimeSpan)
    """Wall-clock span from sending the request to the fully received response."""
    error: Error | None = None
    """The failure that ended this call, coupled to the exchange that caused it; None on
    success. A failed call still records the settings it was sent with."""


class Branch(StrictBaseModel):
    """A root-to-leaf graph path; each branch becomes one training sample."""

    index: int
    nodes: list[MessageNode]
    calls: list[ModelCall] = Field(default_factory=list)

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
    def kept_tokens(self) -> KeptTokens | None:
        """The branch's kept-set sampling masks: `counts` is int32 aligned 1:1 with
        `token_ids` (0 = no mask, safe under partial coverage — unlike `routed_experts`
        this is not all-or-nothing), `ids` the flat int32 concatenation of the kept
        sets in position order. None when no node carries kept-set data."""
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
        """Provider usage from the final model call on this branch — the full context it saw."""
        return next(
            (c.usage for c in reversed(self.calls) if c.usage is not None), None
        )

    @property
    def num_total_tokens(self) -> int:
        """Final sequence length: the last call's prompt + completion. Earlier turns' context
        is already contained in that prompt, so re-sent tokens are counted once rather than
        summed per turn."""
        last = self.last_usage
        return last.total_tokens if last is not None else 0

    @property
    def num_output_tokens(self) -> int:
        """Every model-generated token across all turns (completions, reasoning included)."""
        usage = self.usage
        return usage.completion_tokens if usage is not None else 0

    @property
    def num_input_tokens(self) -> int:
        """Tokens fed to the model, counted once: the final sequence minus everything the model
        generated (i.e. system + user + tool inputs). Not the last prompt — re-sent context is
        not double-counted."""
        return self.num_total_tokens - self.num_output_tokens


_NODE_DUMP_EXCLUDE: dict = {
    "nodes": {
        "__all__": {
            "multi_modal_data",
            "routed_experts",
            "kept_tokens",
        }
    }
}
"""Raw tensor fields kept on the msgpack wire but excluded from JSON records."""


TRACE_VERSION = 2
"""Version of the trace record schema (see `Trace.model_json_schema()`). Bumped on
breaking shape changes; optional-with-default fields are additive and don't bump it."""


class EvalRunInfo(StrictBaseModel):
    """An eval run, stamped by the consumer (the eval CLI / a trainer's inline eval)."""

    type: Literal["eval"] = "eval"
    id: str | None = None
    """The producing run: the eval CLI stamps its run uuid (a resumed eval counts as
    a new run; kept traces keep their original id), trainers stamp their own."""
    step: int | None = None
    """The training step an inline eval was triggered at, stamped by the trainer;
    None for a standalone eval (the eval CLI doesn't set it)."""


class TrainRunInfo(StrictBaseModel):
    """A training run, stamped by the trainer."""

    type: Literal["train"] = "train"
    id: str | None = None
    """The trainer's run identifier."""
    step: int | None = None
    """The training step this rollout belongs to."""


RunInfo = Annotated[EvalRunInfo | TrainRunInfo, Field(discriminator="type")]
"""The run a trace belongs to, discriminated on `type`."""


class VersionInfo(StrictBaseModel):
    """The verifiers build that produced this trace."""

    version: str
    """The installed verifiers package version."""
    commit: str | None = None
    """The verifiers git commit, when resolvable (a git-pinned install or a source
    checkout); None otherwise (e.g. a PyPI wheel)."""


class AgentInfo(StrictBaseModel):
    """The agent that produced this trace's sampled turns."""

    model: str
    """The model identifier requested from the client."""
    sampling: SamplingConfig | None = None
    """The resolved sampling settings the rollout ran with."""
    harness: HarnessConfig | None = None
    """The driving harness's config. Typed as the base config, so a custom harness's
    extra fields don't serialize — records round-trip without importing the harness."""


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
    version: int = TRACE_VERSION
    """The trace record schema this trace serializes as."""
    verifiers: VersionInfo | None = None
    """The verifiers build that produced this trace, stamped at rollout start —
    replayed/re-read traces keep the build that originally produced them."""
    run: RunInfo | None = None
    """The run this trace belongs to (eval or train), consumer-stamped."""
    agent: AgentInfo | None = None
    """The agent (model, sampling, harness) that produced the sampled turns."""
    nodes: list[MessageNode] = Field(default_factory=list)
    """The message graph; branches are derived views and storage stays linear in turns."""
    tools: list[Tool] | None = None
    """The tools advertised to the model, recorded when an intercepted turn commits (last
    committed turn wins) — never from a refused/failed request the model never saw. The full
    advertised list (not just tools called), so tool-use SFT can re-render the exact prompt;
    a trace-level snapshot: mid-rollout changes collapse to the last set the model saw."""
    calls: list[ModelCall] = Field(default_factory=list)
    """Every provider exchange behind the sampled turns, in order: raw wire request/response
    plus per-call timing and errors, linked into `nodes` via `ModelCall.node`."""

    rewards: dict[str, float] = Field(default_factory=dict)
    """Weighted contributions from task rewards, judges, and the env's `score()`."""
    metrics: dict[str, float] = Field(default_factory=dict)
    """Unweighted metrics from tasks, harnesses, and judges."""
    info: dict[str, Any] = Field(default_factory=dict)
    """Persistent JSON scratch space for task metadata that is not a reward or metric."""
    role: str | None = None
    """Which env role produced this trace (`None` for the single-agent default, where
    there's only one). Stamped by the env layer; first-class so training can filter
    and baseline per role without digging through `info`."""
    trainable: bool = True
    """Whether this trace's tokens are training data for the run's policy. An env
    marks fixed-model roles (a frozen judge, a pinned user sim) untrainable."""
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
    def has_response(self) -> bool:
        last = self._last_assistant()
        return bool(last and last.message.content)

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
        """Total model turns (sampled responses) across all branches — prompt-supplied
        assistant messages don't count."""
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
            "harness_timeout",
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

    def stamp(self, run: RunInfo | None = None, **info: Any) -> None:
        """Stamp identity only the consumer knows (the eval CLI / a trainer) onto the
        trace; anything beyond `run` lands in `info`."""
        if run is not None:
            self.run = run
        self.info.update(info)

    def stop(self, condition: str = "done") -> None:
        self.is_completed = True
        if self.stop_condition is None:
            self.stop_condition = condition

    def split_generation(self) -> None:
        """Stamp the closed generation span's model/harness split: model time is the
        sum of the recorded calls' spans (clamped to the span), harness the complement.
        Every path that closes the span calls this — a span without model calls (e.g.
        a debug action) is all harness time."""
        gen = self.timing.generation
        if not gen.end:
            return
        model = sum(call.time.duration for call in self.calls)
        gen.model.duration = min(model, gen.duration)
        gen.harness.duration = gen.duration - gen.model.duration

    def capture_error(self, error: Exception) -> None:
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
        self.stop("error")

    def to_record(self) -> dict[str, Any]:
        """JSON record without raw tensors, which remain available on the msgpack wire."""
        return self.model_dump(mode="json", exclude=_NODE_DUMP_EXCLUDE)


TraceT = TypeVar("TraceT", bound=Trace)  # type: ignore[type-arg]

WireTrace = Trace[WireTaskData]
"""Trace loader that preserves unknown task fields in `task.model_extra`."""


class Episode(StrictBaseModel, Generic[DataT, StateT]):
    """One run of the env — the global view, each trace one agent's local view. The
    atom of `traces.jsonl` and of the serve protocol: a single-agent episode carries
    one trace, a multi-agent env's the traces of its views — they succeed, resume,
    and score as a unit. `errors` are failures not attributable to any one trace
    (the env's `rollout`/`score` hooks, plus prior attempts' when retried); per-trace
    failures stay on the traces.

    The type parameters serve the wire loaders, not authoring: `WireEpisode =
    Episode[WireTaskData, State]` reads any taskset's episodes without importing the
    taskset (unknown row fields kept in `model_extra`). An authored episode is
    typically heterogeneous — a taskset-typed solver trace next to a plain minted
    verdict trace — and uses bare `Episode`."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    env: str = ""
    """The env (taskset) id that produced this episode — provenance for mixed files."""
    task: TraceTask[DataT]
    """The task rolled out, as recorded on its traces (class name + row)."""
    errors: list[Error] = Field(default_factory=list)
    """Episode-level errors, oldest first (more than one only when the episode was
    retried). `error` exposes the most recent."""
    traces: list[Trace[DataT, StateT]] = Field(default_factory=list)

    @property
    def error(self) -> Error | None:
        return self.errors[-1] if self.errors else None

    @property
    def ok(self) -> bool:
        """Whether the whole episode is good — no episode-level error and no trace
        errors. The resume unit: anything less is redone."""
        return not self.errors and not any(t.errors for t in self.traces)

    @classmethod
    def of(cls, trace: Trace, env: str = "") -> "Episode":
        """The single-agent episode: one trace, task lifted off it."""
        return cls(env=env, task=trace.task, traces=[trace])


WireEpisode = Episode[WireTaskData, State]
"""Episode loader that preserves unknown task fields in `task.model_extra`."""

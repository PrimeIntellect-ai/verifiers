"""Post-hoc trace updates — facts a consumer learns after a trace is completed."""

from typing import Any

from pydantic import BaseModel, Field

from verifiers.v1.trace import Trace

STREAM_FIELDS = ("advantages", "trainer_logprobs", "entropies")


class BranchUpdate(BaseModel):
    """Per-token streams over one branch's token prefix.

    Streams align to `branch.token_ids[:len(stream)]` — one entry per token, not compact
    over the sampled mask, so applying never depends on the producer's loss mask. A null
    entry marks an unknown value (e.g. a packing-boundary token)."""

    branch_id: int
    """`Branch.index` of the branch the streams cover (leaf order)."""
    advantages: list[float | None] | None = None
    trainer_logprobs: list[float | None] | None = None
    entropies: list[float | None] | None = None


class TraceUpdate(BaseModel):
    """A mergeable update to a completed trace, applied via `apply_trace_update`."""

    version: int = 1
    trace_id: str
    info: dict[str, Any] = Field(default_factory=dict)
    """Deep-merged into `Trace.info`; on collisions the update wins."""
    branches: list[BranchUpdate] = Field(default_factory=list)


def _deep_merge(old: Any, new: Any) -> Any:
    if isinstance(old, dict) and isinstance(new, dict):
        return {
            **old,
            **{key: _deep_merge(old.get(key), value) for key, value in new.items()},
        }
    return new


def apply_trace_update(trace: Trace, update: TraceUpdate) -> None:
    """Stamp an update onto its trace: merge `info` and project branch streams onto nodes.

    Each stream is cursor-walked over the branch's nodes. A node is stamped — compact over
    its `mask`, like `logprobs` — only when the stream fully covers it with non-null values
    at every sampled position. Updates apply in call order and the newest wins, on nodes
    shared across branches too. A stream shorter than the branch (e.g. a truncated
    training sample) leaves the tail nodes untouched."""
    if trace.id != update.trace_id:
        raise ValueError(
            f"update for trace {update.trace_id} applied to trace {trace.id}"
        )
    trace.info = _deep_merge(trace.info, update.info)
    if not update.branches:
        return
    branches = trace.branches
    for branch_update in update.branches:
        if not 0 <= branch_update.branch_id < len(branches):
            continue
        nodes = branches[branch_update.branch_id].nodes
        for field in STREAM_FIELDS:
            stream = getattr(branch_update, field)
            if stream is None:
                continue
            cursor = 0
            for node in nodes:
                span = stream[cursor : cursor + len(node.token_ids)]
                cursor += len(node.token_ids)
                if len(span) < len(node.token_ids):
                    break
                values = [v for v, sampled in zip(span, node.mask) if sampled]
                if not values or any(v is None for v in values):
                    continue
                setattr(node, field, values)

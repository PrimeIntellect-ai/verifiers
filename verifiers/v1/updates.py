"""Post-hoc trace updates — facts a consumer learns after a trace is completed."""

from typing import Any

from pydantic import BaseModel, Field

from verifiers.v1.trace import Trace


class BranchUpdate(BaseModel):
    """Per-token streams over one branch's token prefix.

    Streams align to `branch.token_ids[:len(stream)]` — one entry per token, not compact
    over the sampled mask, so applying never depends on the producer's loss mask. A null
    entry marks an unknown value (e.g. a packing-boundary token)."""

    index: int
    """`Branch.index` of the branch the streams cover (leaf order)."""
    trainer_logprobs: list[float | None] | None = None
    entropies: list[float | None] | None = None


class TraceUpdate(BaseModel):
    """A mergeable update to a completed trace, applied via `apply_trace_update`."""

    version: int = 1
    trace_id: str
    info: dict[str, Any] = Field(default_factory=dict)
    """Merged into `Trace.info` per top-level key; dict values shallow-merge, new wins.
    Producers namespace their keys (e.g. `info["train"]`)."""
    branches: list[BranchUpdate] = Field(default_factory=list)


def apply_trace_update(trace: Trace, update: TraceUpdate) -> None:
    """Stamp an update onto its trace: merge `info` and project branch streams onto nodes.

    Each stream is cursor-walked over the branch's nodes. A node is stamped — compact over
    its `mask`, like `logprobs` — only when the stream fully covers it with non-null values
    at every sampled position, and only while the node field is still unset: the first
    writer wins on nodes shared across branches. A stream shorter than the branch (e.g. a
    truncated training sample) leaves the tail nodes unstamped."""
    if trace.id != update.trace_id:
        raise ValueError(
            f"update for trace {update.trace_id} applied to trace {trace.id}"
        )
    for key, value in update.info.items():
        old = trace.info.get(key)
        if isinstance(old, dict) and isinstance(value, dict):
            trace.info[key] = {**old, **value}
        else:
            trace.info[key] = value
    if not update.branches:
        return
    branches = trace.branches
    for branch_update in update.branches:
        if not 0 <= branch_update.index < len(branches):
            continue
        nodes = branches[branch_update.index].nodes
        for field in ("trainer_logprobs", "entropies"):
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
                if getattr(node, field) is None:
                    setattr(node, field, values)

"""Export completed v1 episodes without instrumenting the running environment.

Install Braintrust separately. The caller owns authentication, project selection,
flush, and export failure policy; importing this module does not import the SDK.
See the adjacent README for saved-episode and async usage.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from verifiers.v1.episode import Episode
from verifiers.v1.trace import Error, Trace
from verifiers.v1.types import Message, Usage

if TYPE_CHECKING:
    from braintrust import Logger, Span

MAX_CONTENT_CHARS = 16_384


def _content(message: Message) -> tuple[str, bool]:
    """Keep one bounded JSON excerpt, never provider replay state or token arrays."""
    chunks: list[str] = []
    remaining = MAX_CONTENT_CHARS
    for chunk in json.JSONEncoder(ensure_ascii=False).iterencode(
        message.model_dump(mode="json", exclude={"provider_state"}, exclude_none=True)
    ):
        chunks.append(chunk[:remaining])
        remaining -= len(chunk)
        if remaining < 0:
            return "".join(chunks), True
    return "".join(chunks), False


def _errors(errors: list[Error], include_content: bool) -> str | None:
    if not errors:
        return None
    if not include_content:
        return ", ".join(error.type for error in errors)[:MAX_CONTENT_CHARS]
    return "\n".join(
        f"{error.type}: {error.message[:MAX_CONTENT_CHARS]}" for error in errors
    )[:MAX_CONTENT_CHARS]


def _usage(usage: Usage | None) -> dict[str, int | float]:
    if usage is None:
        return {}
    metrics: dict[str, int | float] = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.completion_tokens,
        "tokens": usage.total_tokens,
    }
    if usage.cached_input_tokens is not None:
        metrics["prompt_cached_tokens"] = usage.cached_input_tokens
    if usage.reasoning_tokens is not None:
        metrics["completion_reasoning_tokens"] = usage.reasoning_tokens
    if usage.cost is not None:
        metrics["cost"] = usage.cost
    return metrics


def _bounds(trace: Trace) -> tuple[float, float]:
    """Measured bounds only: never stretch a historical trace to export time."""
    stamps = [trace.timing.start]
    for phase in ("boot", "setup", "agent", "finalize", "scoring"):
        span = getattr(trace.timing, phase)
        stamps.extend((span.start, span.end))
    stamps.extend(node.timestamp for node in trace.nodes)
    for call in trace.calls:
        stamps.extend((call.time.start, call.time.end))
    known = [stamp for stamp in stamps if stamp > 0]
    if not known:
        raise ValueError(f"Trace {trace.id} has no recorded timestamps")
    return min(known), max(known)


def log_episode(
    episode: Episode, logger: Logger, *, include_content: bool = False
) -> None:
    """Queue an episode → agent trace → message/model-call span tree in Braintrust.

    Messages are emitted once, not once per root-to-leaf branch. Message timestamps
    are commit instants, not tool execution durations. Graph edges stay in metadata.
    Content is opt-in and capped at 16,384 characters per message/error excerpt.
    Task data, agent config, trace.info, tensors, and provider replay state are never
    sent. Identifiers, model/tool names, rewards, and numeric metrics ARE sent.

    Export synchronously after completion (or use asyncio.to_thread). Call
    logger.flush() after a batch. This does not intercept calls, mutate the episode,
    set current span context, suppress errors, or automatically enable CLI logging.
    Each invocation is a new export, not a deduplicating upsert.
    """
    bounds = [_bounds(trace) for trace in episode.traces]
    # An episode can fail before producing any trace and has no own timing field.
    # Represent that case as an export-time point, explicitly labelled as such.
    now = time.time()
    start = min((start for start, _ in bounds), default=now)
    end = max((end for _, end in bounds), default=now)
    root = logger.start_span(
        name="verifiers.episode",
        type="task",
        start_time=start,
        set_current=False,
        metadata={
            "episode_id": episode.id,
            "env_id": episode.env.id,
            "run_id": episode.run.id if episode.run else None,
            "ok": episode.ok,
            "timing_source": "recorded" if bounds else "export_time",
            "include_content": include_content,
            "num_traces": len(episode.traces),
        },
        error=_errors(episode.errors, include_content),
    )
    try:
        for trace, (trace_start, trace_end) in zip(episode.traces, bounds):
            _log_trace(trace, root, trace_start, trace_end, include_content)
    finally:
        root.end(end_time=end)


def _log_trace(
    trace: Trace, root: Span, start: float, end: float, include_content: bool
) -> None:
    span = root.start_span(
        name=f"agent:{trace.agent.name}",
        type="task",
        start_time=start,
        set_current=False,
        metadata={
            "trace_id": trace.id,
            "model": trace.agent.config.model,
            "ok": trace.ok,
            "stop_condition": trace.stop_condition,
            "num_nodes": len(trace.nodes),
            "num_calls": len(trace.calls),
            "rewards": {
                name: reward.model_dump() if reward else None
                for name, reward in trace.rewards.items()
            },
        },
        # Verifiers rewards need not be in [0, 1]; Braintrust scores must be.
        # Preserve every weighted reward and metric without inventing normalization.
        metrics={
            "verifiers_reward": trace.reward,
            **{
                f"verifiers/{key}": value
                for key, value in trace.metrics.items()
                if value is not None
            },
        },
        error=_errors(trace.errors, include_content),
    )
    try:
        for index, node in enumerate(trace.nodes):
            message = node.message
            metadata = {
                "node": index,
                "parent_node": node.parent,
                "sampled": node.sampled,
                "timestamp_source": "message_commit",
                "semantic_parents": [
                    parent.model_dump() for parent in node.semantic_parents
                ],
            }
            output = None
            if include_content:
                output, truncated = _content(message)
                metadata["content_truncated"] = truncated
            point = node.timestamp if node.timestamp > 0 else start
            child = span.start_span(
                name=f"message:{message.role}",
                type="task",
                start_time=point,
                set_current=False,
                metadata=metadata,
                output=output,
            )
            child.end(end_time=point)
        for index, call in enumerate(trace.calls):
            call_start = call.time.start if call.time.start > 0 else start
            call_end = max(call_start, call.time.end)
            child = span.start_span(
                name="model_call",
                type="llm",
                start_time=call_start,
                set_current=False,
                metadata={
                    "call_index": index,
                    "node": call.node,
                    "model": call.model,
                    "finish_reason": call.finish_reason,
                    "timing_source": "recorded"
                    if call.time.start > 0
                    else "trace_start",
                },
                metrics=_usage(call.usage),
                error=_errors([call.error] if call.error else [], include_content),
            )
            child.end(end_time=call_end)
        # Judge/off-graph accounting is separate, never attached to an agent call.
        if trace.extra_usage:
            span.log(
                metadata={"extra_usage": _usage(Usage.aggregate(trace.extra_usage))}
            )
    finally:
        span.end(end_time=end)

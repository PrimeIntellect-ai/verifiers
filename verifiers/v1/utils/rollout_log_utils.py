import logging
from collections import Counter
from typing import cast

from verifiers.utils.display_utils import format_timing_plain
from verifiers.utils.logging_utils import truncate

from ..state import State
from ..types import JsonData

logger = logging.getLogger("verifiers.v1.rollout")


def log_rollout_start(state: State) -> None:
    """Emit an INFO line at the start of a rollout (mirrors interception logs)."""
    if not logger.isEnabledFor(logging.INFO):
        return
    parts = [
        f"Started  example_id={state.get('example_id')}",
        f"trajectory_id={state.get('trajectory_id')}",
    ]
    logger.info(" | ".join(parts))


def log_rollout_finish(state: State) -> None:
    """Emit an INFO line at the end of a rollout summarizing its outcome."""
    if not logger.isEnabledFor(logging.INFO):
        return
    trajectory = state.get("trajectory") or []
    tool_counts = rollout_tool_counts(trajectory)
    tools_str = ",".join(f"{name}:{count}" for name, count in tool_counts.most_common())
    parts = [
        f"Finished example_id={state.get('example_id')}",
        f"trajectory_id={state.get('trajectory_id')}",
        f"turns={len(trajectory)}",
        f"tools=[{tools_str}]",
        f"timing={rollout_timing_summary(state.get('timing'))}",
        f"stop={state.get('stop_condition')}",
        f"reward={state.get('reward')}",
        f"metrics={format_metrics(state.get('metrics'))}",
    ]
    error = state.get("error")
    if error:
        parts.append(f"error={format_error(error)}")
    if state.get("is_truncated"):
        parts.append("truncated=True")
    logger.info(" | ".join(parts))


def rollout_tool_counts(trajectory: object) -> Counter[str]:
    """Count tool calls by name across all assistant messages in a trajectory."""
    counts: Counter[str] = Counter()
    if not isinstance(trajectory, list):
        return counts
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        completion = cast(JsonData, step).get("completion")
        if not isinstance(completion, list):
            continue
        for message in completion:
            for name in message_tool_names(message):
                counts[name] += 1
    return counts


def message_tool_names(message: object) -> list[str]:
    if isinstance(message, dict):
        tool_calls = cast(JsonData, message).get("tool_calls")
    else:
        tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for tool_call in tool_calls:
        name = tool_call_name(tool_call)
        if name:
            names.append(name)
    return names


def tool_call_name(tool_call: object) -> str | None:
    if isinstance(tool_call, dict):
        call = cast(JsonData, tool_call)
        name = call.get("name")
        if name is None:
            function = call.get("function")
            if isinstance(function, dict):
                name = cast(JsonData, function).get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool_call, "name", None)
    if name is None:
        name = getattr(getattr(tool_call, "function", None), "name", None)
    return name if isinstance(name, str) else None


def rollout_timing_summary(timing: object) -> str:
    if not isinstance(timing, dict):
        return ""
    spans = cast(JsonData, timing)
    return format_timing_plain(
        setup=span_duration(spans.get("setup")),
        generation=span_duration(spans.get("generation")),
        scoring=span_duration(spans.get("scoring")),
        overhead=as_float(spans.get("overhead")),
        model=span_duration(spans.get("model")),
        env=span_duration(spans.get("env")),
    )


def span_duration(span: object) -> float:
    if not isinstance(span, dict):
        return 0.0
    return as_float(cast(JsonData, span).get("duration"))


def as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def format_metrics(metrics: object) -> str:
    if not isinstance(metrics, dict):
        return "{}"
    items = cast("dict[str, float]", metrics).items()
    return "{" + ",".join(f"{name}:{value}" for name, value in items) + "}"


def format_error(error: object) -> str:
    if isinstance(error, dict):
        info = cast(JsonData, error)
        detail = info.get("error_chain_str") or info.get("error") or ""
        return truncate(str(detail), 200)
    return truncate(str(error), 200)

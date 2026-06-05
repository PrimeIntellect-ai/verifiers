import logging
from collections import Counter

from verifiers.utils.display_utils import format_numeric, format_timing_plain
from verifiers.utils.error_utils import error_data, validate_error_data
from verifiers.utils.logging_utils import truncate

from ..state import State

logger = logging.getLogger("verifiers.v1.rollout")


def log_rollout_start(state: State) -> None:
    logger.info(
        f"Started example_id={state.get('example_id')} "
        f"| trajectory_id={state.get('trajectory_id')}"
    )


def log_rollout_finish(state: State) -> None:
    tools = Counter(
        call["name"]
        for step in state.get("trajectory") or []
        for message in step.get("completion") or []
        for call in message.get("tool_calls") or []
    )
    timing = state.get("timing") or {}
    metrics = state.get("metrics") or {}

    def duration(phase: str) -> float:
        return (timing.get(phase) or {}).get("duration", 0.0)

    tool_summary = ", ".join(f"{name}: {n}" for name, n in tools.most_common())
    metric_summary = ", ".join(
        f"{name}: {format_numeric(value)}" for name, value in metrics.items()
    )
    parts = [
        f"Finished example_id={state.get('example_id')}",
        f"trajectory_id={state.get('trajectory_id')}",
        f"tools=[{tool_summary}]",
        "timing="
        + format_timing_plain(
            setup=duration("setup"),
            generation=duration("generation"),
            scoring=duration("scoring"),
            overhead=timing.get("overhead", 0.0),
            model=duration("model"),
            env=duration("env"),
        ),
        f"stop={state.get('stop_condition')}",
        f"reward={format_numeric(state.get('reward') or 0.0)}",
        "metrics={" + metric_summary + "}",
    ]
    rollout_error = state.get("error")
    if rollout_error:
        if isinstance(rollout_error, BaseException):
            error_chain = error_data(rollout_error)["error_chain_str"]
        else:
            error_chain = validate_error_data(rollout_error)["error_chain_str"]
        parts.append(f"error={truncate(error_chain, 200)}")
    if state.get("is_truncated"):
        parts.append("truncated=True")
    logger.info(" | ".join(parts))

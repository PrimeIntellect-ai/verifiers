from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from verifiers.types import State, TrajectoryStep


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_tokens(response: Any) -> tuple[int, int]:
    if response is None:
        return 0, 0
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return _as_int(prompt_tokens), _as_int(completion_tokens)


def _get_step_extras(step: TrajectoryStep) -> dict[str, Any]:
    extras = step.get("extras", {})
    return extras if isinstance(extras, dict) else {}


async def add_metrics_to_state(
    state: State,
    *,
    sub_llm_steps: list[TrajectoryStep] | None = None,
    include_raw_maps: bool = True,
) -> None:
    """Add RLM metrics to state, using stable scalar keys.

    If sub_llm_steps is provided, it will be used for sub-LLM metrics even if
    those steps are not in state["trajectory"] (e.g. when sub-steps are not
    prepended).
    """
    trajectory = state.get("trajectory", []) or []

    if sub_llm_steps is None:
        sub_llm_steps = [
            step for step in trajectory if _get_step_extras(step).get("is_sub_llm_call")
        ]

    main_steps = [
        step for step in trajectory if not _get_step_extras(step).get("is_sub_llm_call")
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tool_calls = 0
    depth_by_call: dict[tuple[str, str], int] = {}
    call_ids: set[tuple[str, str]] = set()
    batch_counts: Counter[str] = Counter()

    for step in sub_llm_steps:
        extras = _get_step_extras(step)
        batch_id = extras.get("batch_id")
        request_id = extras.get("request_id")
        if batch_id is not None and request_id is not None:
            key = (str(batch_id), str(request_id))
            call_ids.add(key)
            batch_counts[str(batch_id)] += 1
            if key not in depth_by_call:
                depth_by_call[key] = _as_int(extras.get("sub_llm_depth", 1))

        total_tool_calls += _as_int(extras.get("tool_call_count", 0))
        prompt_tokens, completion_tokens = _extract_tokens(step.get("response"))
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

    call_count = len(call_ids)
    total_turns = len(sub_llm_steps)

    batch_sizes = list(batch_counts.values())
    batch_count = len(batch_sizes)
    max_batch_size = max(batch_sizes, default=0)
    mean_batch_size = sum(batch_sizes) / batch_count if batch_count else 0.0

    if call_count == 0 and total_turns > 0:
        # Fallback if call ids are missing; treat each turn as its own call.
        call_count = total_turns

    depth_values = list(depth_by_call.values())
    depth_max = max(depth_values, default=0)
    depth_mean = sum(depth_values) / len(depth_values) if depth_values else 0.0
    depth_gt1_frac = (
        sum(1 for d in depth_values if d > 1) / len(depth_values)
        if depth_values
        else 0.0
    )

    if call_count > 0:
        prompt_per_call = total_prompt_tokens / call_count
        completion_per_call = total_completion_tokens / call_count
        tool_calls_per_call = total_tool_calls / call_count
        turns_per_call = total_turns / call_count
    else:
        prompt_per_call = 0.0
        completion_per_call = 0.0
        tool_calls_per_call = 0.0
        turns_per_call = 0.0

    state["sub_llm_call_count"] = call_count
    state["sub_llm_total_turns"] = total_turns
    state["sub_llm_prompt_tokens"] = total_prompt_tokens
    state["sub_llm_completion_tokens"] = total_completion_tokens
    state["sub_llm_total_tool_calls"] = total_tool_calls
    state["sub_llm_batch_count"] = batch_count
    state["sub_llm_max_batch_size"] = max_batch_size
    state["sub_llm_mean_batch_size"] = mean_batch_size

    state["sub_llm_depth_max"] = depth_max
    state["sub_llm_depth_mean"] = depth_mean
    state["sub_llm_depth_gt1_frac"] = depth_gt1_frac

    state["sub_llm_prompt_tokens_per_call"] = prompt_per_call
    state["sub_llm_completion_tokens_per_call"] = completion_per_call
    state["sub_llm_tool_calls_per_call"] = tool_calls_per_call
    state["sub_llm_turns_per_call"] = turns_per_call

    if include_raw_maps:
        calls_by_depth: dict[int, int] = {}
        prompt_by_depth: dict[int, int] = {}
        completion_by_depth: dict[int, int] = {}
        tool_calls_by_depth: dict[int, int] = {}
        turns_by_depth: dict[int, int] = {}

        for step in sub_llm_steps:
            extras = _get_step_extras(step)
            depth = _as_int(extras.get("sub_llm_depth", 1))
            prompt_tokens, completion_tokens = _extract_tokens(step.get("response"))

            turns_by_depth[depth] = turns_by_depth.get(depth, 0) + 1
            tool_calls_by_depth[depth] = tool_calls_by_depth.get(depth, 0) + _as_int(
                extras.get("tool_call_count", 0)
            )
            prompt_by_depth[depth] = prompt_by_depth.get(depth, 0) + prompt_tokens
            completion_by_depth[depth] = completion_by_depth.get(depth, 0) + (
                completion_tokens
            )

        for depth in depth_values:
            calls_by_depth[depth] = calls_by_depth.get(depth, 0) + 1

        state["sub_llm_calls_by_depth"] = calls_by_depth
        state["sub_llm_prompt_tokens_by_depth"] = prompt_by_depth
        state["sub_llm_completion_tokens_by_depth"] = completion_by_depth
        state["sub_llm_tool_calls_by_depth"] = tool_calls_by_depth
        state["sub_llm_turns_by_depth"] = turns_by_depth

    main_prompt_tokens = 0
    main_completion_tokens = 0
    for step in main_steps:
        prompt_tokens, completion_tokens = _extract_tokens(step.get("response"))
        main_prompt_tokens += prompt_tokens
        main_completion_tokens += completion_tokens

    state["main_rlm_turns"] = len(main_steps)
    state["main_rlm_prompt_tokens"] = main_prompt_tokens
    state["main_rlm_completion_tokens"] = main_completion_tokens

    tool_timings = state.get("tool_call_timings", [])
    total_time = (
        sum(t.get("execution_seconds", 0.0) for t in tool_timings)
        if tool_timings
        else 0.0
    )
    call_count = len(tool_timings) if tool_timings else 0

    state["repl_total_time_seconds"] = total_time
    state["repl_call_count"] = call_count
    state["repl_mean_time_seconds"] = total_time / call_count if call_count else 0.0


def _make_state_metric_func(key: str) -> Callable[..., float]:
    def _metric(*, state: State, **kwargs: Any) -> float:
        return _as_float(state.get(key, 0.0))

    _metric.__name__ = key
    return _metric


def get_rlm_rubrics(
    base: list[Callable] | None = None,
    base_weights: list[float] | None = None,
    *,
    weight: float = 0.0,
) -> tuple[list[Callable], list[float]]:
    """Return RLM metric rubric funcs + weights, appending to optional base lists."""
    funcs = list(base or [])
    weights = list(base_weights or [])

    if base_weights is not None and len(weights) != len(funcs):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of functions ({len(funcs)})"
        )

    metric_keys = [
        "sub_llm_call_count",
        "sub_llm_total_turns",
        "sub_llm_prompt_tokens",
        "sub_llm_completion_tokens",
        "sub_llm_total_tool_calls",
        "sub_llm_batch_count",
        "sub_llm_max_batch_size",
        "sub_llm_mean_batch_size",
        "sub_llm_depth_max",
        "sub_llm_depth_mean",
        "sub_llm_depth_gt1_frac",
        "sub_llm_prompt_tokens_per_call",
        "sub_llm_completion_tokens_per_call",
        "sub_llm_tool_calls_per_call",
        "sub_llm_turns_per_call",
        "main_rlm_turns",
        "main_rlm_prompt_tokens",
        "main_rlm_completion_tokens",
        "repl_total_time_seconds",
        "repl_call_count",
        "repl_mean_time_seconds",
    ]

    existing = {func.__name__ for func in funcs}
    for key in metric_keys:
        if key in existing:
            continue
        funcs.append(_make_state_metric_func(key))
        weights.append(weight)

    return funcs, weights

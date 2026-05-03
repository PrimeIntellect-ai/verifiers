from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ..state import State


def sync_trajectory(
    state: State,
    trajectory: Sequence[Mapping[str, object]] | None = None,
) -> State:
    if trajectory is not None:
        state["trajectory"] = [dict(step) for step in trajectory]

    steps = state.get("trajectory") or []
    if not isinstance(steps, list):
        raise TypeError("state.trajectory must be a list.")

    state["num_model_requests"] = len(steps)
    state["is_truncated"] = bool(state.get("is_truncated", False)) or any(
        bool(cast(Mapping[str, object], step).get("is_truncated", False))
        for step in steps
    )

    if not steps:
        return state

    state["prompt"] = message_list(steps[0], "prompt")
    state["completion"] = completion_from_trajectory(steps)
    return state


def completion_from_trajectory(steps: Sequence[Mapping[str, object]]) -> list[Any]:
    if not steps:
        return []
    first_prompt = message_list(steps[0], "prompt")
    last_prompt = message_list(steps[-1], "prompt")
    last_completion = message_list(steps[-1], "completion")
    last_trace = [*last_prompt, *last_completion]
    if last_trace[: len(first_prompt)] == first_prompt:
        return last_trace[len(first_prompt) :]
    return last_trace


def message_list(step: object, field: str) -> list[Any]:
    if not isinstance(step, Mapping):
        raise TypeError("trajectory steps must be mappings.")
    value = cast(Mapping[str, object], step).get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"trajectory step {field} must be a list.")
    return list(value)

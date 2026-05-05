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
    state._set_truncated(
        any(
            bool(cast(Mapping[str, object], step).get("is_truncated", False))
            for step in steps
        )
    )

    if not steps:
        return state

    state["prompt"] = message_list(steps[0], "prompt")
    state["completion"] = merge_existing_completion(
        completion_from_trajectory(steps), state.get("completion")
    )
    return state


def has_borrowed_trajectory(state: Mapping[str, object]) -> bool:
    runtime = state.get("runtime")
    if not isinstance(runtime, Mapping):
        return False
    runtime = cast(Mapping[str, object], runtime)
    resolved = runtime.get("resolved")
    if not isinstance(resolved, Mapping):
        return False
    return isinstance(cast(Mapping[str, object], resolved).get("trajectory"), Mapping)


async def render_completion(task: Mapping[str, Any], state: State) -> None:
    _ = task
    if has_borrowed_trajectory(state):
        return
    sync_trajectory(state)


setattr(render_completion, "update", True)
setattr(render_completion, "update_priority", -100)
setattr(render_completion, "update_stage", "rollout")


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


def merge_existing_completion(
    trajectory_completion: list[Any], existing: object
) -> list[Any]:
    if not isinstance(existing, list):
        return trajectory_completion
    if existing[: len(trajectory_completion)] == trajectory_completion:
        return list(existing)
    return trajectory_completion


def message_list(step: object, field: str) -> list[Any]:
    if not isinstance(step, Mapping):
        raise TypeError("trajectory steps must be mappings.")
    value = cast(Mapping[str, object], step).get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"trajectory step {field} must be a list.")
    return list(value)

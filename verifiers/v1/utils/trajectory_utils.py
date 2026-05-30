from collections.abc import Sequence
from typing import cast

from ..state import State
from verifiers.types import Message

from ..runtime_handles import ResolvedRuntimeHandlesConfig
from ..types import ConfigData, JsonData, PromptMessage


def sync_trajectory(
    state: State,
    trajectory: Sequence[JsonData] | None = None,
) -> State:
    if trajectory is not None:
        state["trajectory"] = [dict(step) for step in trajectory]

    steps = state.get("trajectory") or []
    if not isinstance(steps, list):
        raise TypeError("state.trajectory must be a list.")
    state["num_model_requests"] = len(steps)
    state._set_truncated(
        any(bool(cast(ConfigData, step).get("is_truncated", False)) for step in steps)
    )

    if not steps:
        return state

    state["prompt"] = message_list(steps[0], "prompt")
    state["completion"] = merge_existing_completion(
        completion_from_trajectory(steps), state.get("completion")
    )
    return state


def has_borrowed_trajectory(state: State) -> bool:
    runtime = state.runtime_state()
    resolved = ResolvedRuntimeHandlesConfig.model_validate(
        runtime.get("resolved") or {}
    )
    return resolved.trajectory is not None


def completion_from_trajectory(steps: Sequence[JsonData]) -> list[PromptMessage]:
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
    trajectory_completion: list[PromptMessage], existing: object
) -> list[PromptMessage]:
    if not isinstance(existing, list):
        return trajectory_completion
    if existing[: len(trajectory_completion)] == trajectory_completion:
        return [cast(PromptMessage, message) for message in existing]
    return trajectory_completion


def message_list(step: object, field: str) -> list[PromptMessage]:
    if not isinstance(step, dict):
        raise TypeError("trajectory steps must be mappings.")
    value = cast(ConfigData, step).get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"trajectory step {field} must be a list.")
    messages: list[PromptMessage] = []
    for item in value:
        if isinstance(item, dict):
            messages.append(cast(JsonData, item))
        elif hasattr(item, "role") and hasattr(item, "content"):
            messages.append(cast(Message, item))
        else:
            raise TypeError(f"trajectory step {field} items must be messages.")
    return messages

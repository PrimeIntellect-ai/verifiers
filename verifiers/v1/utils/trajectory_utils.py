from collections.abc import Mapping, Sequence
from typing import cast

from ..state import State
from verifiers.types import Message

from ..types import ConfigMap, PromptMessage


def sync_trajectory(
    state: State,
    trajectory: Sequence[ConfigMap] | None = None,
) -> State:
    if trajectory is not None:
        state["trajectory"] = [dict(step) for step in trajectory]

    steps = state.get("trajectory") or []
    if not isinstance(steps, list):
        raise TypeError("state.trajectory must be a list.")
    state["num_model_requests"] = len(steps)
    state._set_truncated(
        any(bool(cast(ConfigMap, step).get("is_truncated", False)) for step in steps)
    )

    if not steps:
        return state

    state["prompt"] = message_list(steps[0], "prompt")
    state["completion"] = merge_existing_completion(
        completion_from_trajectory(steps), state.get("completion")
    )
    return state


def has_borrowed_trajectory(state: ConfigMap) -> bool:
    runtime = state.get("runtime")
    if not isinstance(runtime, Mapping):
        return False
    runtime = cast(ConfigMap, runtime)
    resolved = runtime.get("resolved")
    if not isinstance(resolved, Mapping):
        return False
    return isinstance(cast(ConfigMap, resolved).get("trajectory"), Mapping)


def completion_from_trajectory(steps: Sequence[ConfigMap]) -> list[PromptMessage]:
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
    if not isinstance(step, Mapping):
        raise TypeError("trajectory steps must be mappings.")
    value = cast(ConfigMap, step).get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"trajectory step {field} must be a list.")
    messages: list[PromptMessage] = []
    for item in value:
        if isinstance(item, Mapping):
            messages.append(cast(ConfigMap, item))
        elif hasattr(item, "role") and hasattr(item, "content"):
            messages.append(cast(Message, item))
        else:
            raise TypeError(f"trajectory step {field} items must be messages.")
    return messages

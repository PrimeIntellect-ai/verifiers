from datetime import datetime
from typing import Any

from verifiers.types import State
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls


def serialize_rollout(
    state: State,
    state_columns: list[str] | None = None,
    include_timestamp: bool = False,
) -> dict[str, Any]:
    prompt = state.get("prompt", "")
    completion = state.get("completion", "")
    clean_prompt = sanitize_tool_calls(messages_to_printable(prompt))
    clean_completion = sanitize_tool_calls(messages_to_printable(completion))

    rollout_data: dict[str, Any] = {
        "example_id": state.get("example_id", 0),
        "prompt": clean_prompt,
        "completion": clean_completion,
        "task": state.get("task", ""),
        "reward": state.get("reward", 0.0),
    }

    if include_timestamp:
        rollout_data["timestamp"] = datetime.now().isoformat()

    timing = state.get("timing", {})
    if timing:
        rollout_data["generation_ms"] = timing.get("generation_ms", 0.0)
        rollout_data["scoring_ms"] = timing.get("scoring_ms", 0.0)
        rollout_data["total_ms"] = timing.get("total_ms", 0.0)

    metrics = state.get("metrics", {})
    for k, v in metrics.items():
        rollout_data[k] = v

    answer = state.get("answer", "")
    if answer:
        rollout_data["answer"] = answer

    info = state.get("info", {})
    if info:
        rollout_data["info"] = info

    if state_columns:
        for col in state_columns:
            if col in state:
                if col == "responses":
                    rollout_data[col] = [r.model_dump() for r in state[col]]
                else:
                    rollout_data[col] = state[col]

    return rollout_data

import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from verifiers.types import GenerateMetadata, RolloutOutput, SamplingArgs, State
from verifiers.utils.path_utils import get_results_path


def to_col_order(list_of_dicts: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {k: [m[k] for m in list_of_dicts] for k in list_of_dicts[0].keys()}


def state_to_rollout_output(
    state: State, state_columns: list[str] = []
) -> RolloutOutput:
    rollout_output = RolloutOutput(
        example_id=state.get("example_id", 0),
        prompt=state.get("prompt"),
        completion=state.get("completion"),
        answer=state.get("answer", ""),
        task=state.get("task", "default"),
        info=state.get("info", {}),
        tools=state.get("oai_tools", {}),
        reward=state.get("reward", 0.0),
        metrics=state.get("metrics", {}),
        stop_condition=state.get("stop_condition", None),
        is_truncated=state.get("is_truncated", False),
        timing=state.get("timing", {}),
        error=state.get("error", None),
    )
    for col in state_columns:
        rollout_output[col] = state.get(col)

    return rollout_output


def states_to_rollout_outputs(
    states: list[State], state_columns: list[str] = []
) -> list[RolloutOutput]:
    return [state_to_rollout_output(state, state_columns) for state in states]


def states_to_generate_metadata(
    env_id: str,
    env_args: dict,
    model: str,
    client: AsyncOpenAI,
    states: list[State],
    state_columns: list[str] | None,
    sampling_args: SamplingArgs,
    start_time: float,
    results_path: Path | None,
) -> GenerateMetadata:
    base_url = str(client.base_url) if hasattr(client, "base_url") else ""
    rewards = [s.get("reward", 0.0) for s in states]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    metrics: dict[str, list[float]] = defaultdict(list)
    for state in states:
        if state.get("metrics"):
            for metric_name, metric_value in state["metrics"].items():
                metrics[metric_name].append(metric_value)
    avg_metrics = {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}

    example_ids = [s.get("example_id", 0) for s in states]
    num_examples = len(set(example_ids)) if example_ids else 0
    rollouts_per_example = len(states) // num_examples if num_examples > 0 else 1

    path_to_save = results_path or get_results_path(env_id, model)

    def tools_key(tools: list | None) -> str:
        if not tools:
            return ""
        return str(sorted([t.get("function", {}).get("name", "") for t in tools]))

    all_tools = [s.get("oai_tools") for s in states]
    unique_tools = set(tools_key(t) for t in all_tools)
    tools = next((t for t in all_tools if t), None) if len(unique_tools) == 1 else None

    return GenerateMetadata(
        env_id=env_id,
        env_args=env_args,
        model=model,
        base_url=base_url,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        date=datetime.now().isoformat(),
        time_ms=(time.time() - start_time) * 1000.0,
        avg_reward=avg_reward,
        avg_metrics=avg_metrics,
        state_columns=state_columns or [],
        path_to_save=path_to_save,
        tools=tools,
    )

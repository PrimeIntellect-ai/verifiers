import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from openai import AsyncOpenAI

from verifiers.types import (
    GenerateMetadata,
    GenerateOutputs,
    RolloutOutput,
    SamplingArgs,
    State,
)
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls
from verifiers.utils.path_utils import get_results_path

logger = logging.getLogger(__name__)


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


def get_hf_hub_dataset_name(outputs: GenerateOutputs) -> str:
    """Auto-generates a dataset name."""
    metadata = outputs["metadata"]
    dataset_name = (
        metadata["env_id"]
        + "_"
        + metadata["model"].replace("/", "_")
        + "_n"
        + str(metadata["num_examples"])
        + "_r"
        + str(metadata["rollouts_per_example"])
    )
    return dataset_name


def sanitize_rollouts(rollouts: list[RolloutOutput]) -> list[dict]:
    """Sanitizes a list of rollouts before saving to disk."""

    def sanitize_rollout(rollout: RolloutOutput) -> dict:
        sanitized_rollout = dict(rollout)
        sanitized_rollout["prompt"] = sanitize_tool_calls(
            messages_to_printable(rollout["prompt"])
        )
        sanitized_rollout["completion"] = sanitize_tool_calls(
            messages_to_printable(rollout["completion"])
        )
        sanitized_rollout["error"] = repr(rollout.get("error"))
        if not rollout.get("answer"):
            sanitized_rollout.pop("answer")
        if not rollout.get("info"):
            sanitized_rollout.pop("info")
        rollout_metrics = rollout.get("metrics", {})
        for k, v in rollout_metrics.items():
            sanitized_rollout[k] = v

        return sanitized_rollout

    return [sanitize_rollout(rollout) for rollout in rollouts]


def sanitize_metadata(metadata: GenerateMetadata) -> dict:
    """Sanitizes metadata before saving to disk."""

    metadata_dict = dict(metadata)
    metadata_dict.pop("path_to_save")
    metadata_dict.pop("date")

    return metadata_dict


def save_to_disk(rollouts: list[dict], metadata: dict, path: Path):
    """Saves (sanitized) rollouts and metadata to disk."""
    path.mkdir(parents=True, exist_ok=True)

    def save_results(results_list: list[dict], results_path: Path):
        with open(results_path, "w") as f:
            for idx, result in enumerate(results_list):
                example_id = result.get("example_id") or "unknown"
                try:
                    json.dump(result, f)
                    f.write("\n")
                except Exception as e:
                    logger.error(
                        f"Failed to save rollout with index {idx} ({example_id=}): {e}"
                    )

    def save_metadata(metadata_dict: dict, metadata_path: Path):
        with open(metadata_path, "w") as f:
            try:
                json.dump(metadata_dict, f)
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")

    save_metadata(metadata, path / "metadata.json")
    save_results(rollouts, path / "results.jsonl")


def save_generate_outputs(
    outputs: GenerateOutputs,
    push_to_hf_hub: bool = False,
    hf_hub_dataset_name: str | None = None,
):
    path_to_save = outputs["metadata"]["path_to_save"]
    sanitized_rollouts = sanitize_rollouts(outputs["rollouts"])
    sanitized_metadata = sanitize_metadata(outputs["metadata"])
    save_to_disk(sanitized_rollouts, sanitized_metadata, path_to_save)
    logger.info(f"Results saved to {path_to_save}")
    if push_to_hf_hub:
        dataset_name = hf_hub_dataset_name or get_hf_hub_dataset_name(outputs)
        try:
            Dataset.from_list(sanitized_rollouts).push_to_hub(dataset_name)
            logger.info(f"Dataset saved to Hugging Face Hub: {dataset_name}")
        except Exception as e:
            logger.error(f"Error pushing dataset to Hugging Face Hub: {e}")

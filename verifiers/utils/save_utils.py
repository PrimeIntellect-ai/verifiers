import json
import logging
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel

from verifiers.types import (
    GenerateMetadata,
    GenerateOutputs,
    SamplingArgs,
    State,
)
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls
from verifiers.utils.path_utils import get_results_path

logger = logging.getLogger(__name__)


def make_serializable(value: Any) -> Any:
    """Convert value to JSON-serializable types for non-standard types.

    Example:
    >>> json.dumps(value, default=make_serializable)
    """
    if isinstance(value, BaseModel):
        return value.model_dump()
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, Path):
        return value.as_posix()
    elif isinstance(value, (BaseException)):
        return repr(value)
    else:
        return str(value)


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
    """Converts a list of states to generate metadata."""
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


def sanitize_states(states: list[State], state_columns: list[str] = []) -> list[dict]:
    """Sanitizes a list of rollouts before saving to disk."""

    def sanitize_state(state: State) -> dict:
        sanitized_state = {
            "example_id": state.get("example_id", 0),
            "prompt": state.get("prompt"),
            "completion": state.get("completion"),
            "answer": state.get("answer", ""),
            "task": state.get("task", "default"),
            "info": state.get("info", {}),
            "reward": state.get("reward", 0.0),
            "error": state.get("error", None),
            "total_ms": state.get("timing", {}).get("total_ms", 0.0),
            "generation_ms": state.get("timing", {}).get("generation_ms", 0.0),
            "scoring_ms": state.get("timing", {}).get("scoring_ms", 0.0),
        }
        # sanitize messages
        sanitized_state["prompt"] = sanitize_tool_calls(
            messages_to_printable(state["prompt"])
        )
        sanitized_state["completion"] = sanitize_tool_calls(
            messages_to_printable(state["completion"])
        )
        # use repr for error
        if state.get("error") is not None:
            sanitized_state["error"] = repr(state.get("error"))
        # only include optional fields if non-empty
        if "answer" in sanitized_state and not sanitized_state["answer"]:
            sanitized_state.pop("answer")
        if "info" in sanitized_state and not sanitized_state["info"]:
            sanitized_state.pop("info")
        # flatten metrics
        state_metrics = state.get("metrics", {})
        for k, v in state_metrics.items():
            sanitized_state[k] = v
        # add state columns
        for col in state_columns:
            sanitized_state[col] = state.get(col)

        return sanitized_state

    return [sanitize_state(state) for state in states]


def sanitize_metadata(metadata: GenerateMetadata) -> dict:
    """Sanitizes metadata before saving to disk."""

    metadata_dict = dict(metadata)
    metadata_dict.pop("path_to_save")
    metadata_dict.pop("date")

    return metadata_dict


def save_to_disk(results: list[dict], metadata: dict, path: Path):
    """Saves (sanitized) rollouts and metadata to disk."""
    path.mkdir(parents=True, exist_ok=True)

    def save_results(results: list[dict], results_path: Path):
        with open(results_path, "w") as f:
            for idx, result in enumerate(results):
                example_id = result.get("example_id") or "unknown"
                try:
                    json.dump(result, f, default=make_serializable)
                    f.write("\n")
                except Exception as e:
                    logger.error(
                        f"Failed to save result with index {idx} ({example_id=}): {e}"
                    )

    def save_metadata(metadata: dict, metadata_path: Path):
        with open(metadata_path, "w") as f:
            try:
                json.dump(metadata, f, default=make_serializable)
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")

    save_metadata(metadata, path / "metadata.json")
    save_results(results, path / "results.jsonl")


def make_dataset(outputs: GenerateOutputs) -> Dataset:
    state_columns = outputs["metadata"]["state_columns"]
    sanitized_states = sanitize_states(outputs["states"], state_columns)
    return Dataset.from_list(sanitized_states)


def save_generate_outputs(
    outputs: GenerateOutputs,
    push_to_hf_hub: bool = False,
    hf_hub_dataset_name: str | None = None,
):
    path_to_save = outputs["metadata"]["path_to_save"]
    state_columns = outputs["metadata"]["state_columns"]
    sanitized_states = sanitize_states(outputs["states"], state_columns)
    sanitized_metadata = sanitize_metadata(outputs["metadata"])
    save_to_disk(sanitized_states, sanitized_metadata, path_to_save)
    logger.info(f"Results saved to {path_to_save}")
    if push_to_hf_hub:
        dataset_name = hf_hub_dataset_name or get_hf_hub_dataset_name(outputs)
        try:
            dataset = make_dataset(outputs)
            dataset.push_to_hub(dataset_name)
            logger.info(f"Dataset saved to Hugging Face Hub: {dataset_name}")
        except Exception as e:
            logger.error(f"Error pushing dataset to Hugging Face Hub: {e}")

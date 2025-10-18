import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset

from verifiers.types import GenerateOutputs
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls

logger = logging.getLogger("verifiers.utils.eval_storage")


def save_results_to_disk(
    env: str,
    results: GenerateOutputs,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    sampling_args: dict | None = None,
    env_dir_path: str = "./environments",
) -> Path:
    """
    Save evaluation results to disk.

    Creates directory structure: <env_dir>/<env>/outputs/evals/<env>--<model>/<uuid>/
    or ./outputs/evals/<env>--<model>/<uuid>/ if env directory doesn't exist.
    """
    uuid_str = str(uuid.uuid4())[:8]

    env_name = env.replace("-", "_")
    model_name = model.replace("/", "--")
    env_model_str = f"{env}--{model_name}"

    local_env_dir = Path(env_dir_path) / env_name
    if local_env_dir.exists():
        results_path = local_env_dir / "outputs" / "evals" / env_model_str / uuid_str
    else:
        results_path = Path("./outputs") / "evals" / env_model_str / uuid_str

    results_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "env": env,
        "model": model,
        "num_examples": num_examples,
        "rollouts_per_example": rollouts_per_example,
        "sampling_args": sampling_args if sampling_args else {},
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "avg_reward": float(np.mean(results.reward)),
    }

    for metric_name, metric_values in results.metrics.items():
        metadata[f"avg_{metric_name}"] = float(np.mean(metric_values))

    with open(results_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    n_samples = len(results.reward)
    ids = [i // rollouts_per_example for i in range(n_samples)]
    printable_prompts = [messages_to_printable(p) for p in results.prompt]
    printable_completions = [messages_to_printable(c) for c in results.completion]

    data_dict = {
        "id": ids,
        "prompt": [sanitize_tool_calls(p) for p in printable_prompts],
        "completion": [sanitize_tool_calls(c) for c in printable_completions],
        "task": results.task,
    }

    if results.info and results.info[0] != {}:
        data_dict["info"] = results.info
    if results.answer and results.answer[0] != "":
        data_dict["answer"] = results.answer
    data_dict["reward"] = results.reward

    for metric_name, metric_values in results.metrics.items():
        data_dict[metric_name] = metric_values

    dataset = Dataset.from_dict(data_dict)
    dataset.to_json(results_path / "results.jsonl")

    logger.info(f"✓ Saved evaluation results to {results_path}")
    return results_path


def save_results_to_hf_hub(
    env: str,
    results: GenerateOutputs,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    dataset_name: str | None = None,
) -> str:
    """Save evaluation results to Hugging Face Hub"""
    n_samples = len(results.reward)
    ids = [i // rollouts_per_example for i in range(n_samples)]
    printable_prompts = [messages_to_printable(p) for p in results.prompt]
    printable_completions = [messages_to_printable(c) for c in results.completion]

    data_dict = {
        "id": ids,
        "prompt": [sanitize_tool_calls(p) for p in printable_prompts],
        "completion": [sanitize_tool_calls(c) for c in printable_completions],
        "task": results.task,
    }

    if results.info and results.info[0] != {}:
        data_dict["info"] = results.info
    if results.answer and results.answer[0] != "":
        data_dict["answer"] = results.answer
    data_dict["reward"] = results.reward

    for metric_name, metric_values in results.metrics.items():
        data_dict[metric_name] = metric_values

    dataset = Dataset.from_dict(data_dict)

    if not dataset_name:
        dataset_name = (
            f"{env}_{model.replace('/', '-')}_n={num_examples}_r={rollouts_per_example}"
        )

    dataset.push_to_hub(dataset_name)
    logger.info(f"✓ Saved dataset to Hugging Face Hub: {dataset_name}")

    return dataset_name

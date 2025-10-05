from datetime import datetime

from datasets import Dataset

from verifiers.types import GenerateOutputs
from verifiers.utils.message_utils import sanitize_tool_calls


def make_dataset(
    results: GenerateOutputs, num_examples: int, rollouts_per_example: int
) -> Dataset:
    """Prepare dataset from eval results."""
    n, r = num_examples, rollouts_per_example
    ids = [i // r for i in range(n * r)]
    data_dict = {
        "id": ids,
        "prompt": [sanitize_tool_calls(p) for p in results.prompt],
        "completion": [sanitize_tool_calls(c) for c in results.completion],
        "task": results.task,
        "generation_ms": [s["timing"]["generation_ms"] for s in results.state],
        "scoring_ms": [s["timing"]["scoring_ms"] for s in results.state],
        "total_ms": [s["timing"]["total_ms"] for s in results.state],
    }
    if results.info[0] != {}:
        data_dict["info"] = results.info
    if results.answer[0] != "":
        data_dict["answer"] = results.answer
    data_dict["reward"] = results.reward
    for k in results.metrics:
        v = results.metrics[k]
        data_dict[k] = v

    return Dataset.from_dict(data_dict)


def make_metadata(
    env: str,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    sampling_args: dict,
    end_time: float,
    start_time: float,
    results: GenerateOutputs,
) -> dict:
    """Prepare metadata from eval results."""
    metadata = {
        "env": env,
        "model": model,
        "num_examples": num_examples,
        "rollouts_per_example": rollouts_per_example,
        "sampling_args": sampling_args,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_ms": (end_time - start_time) * 1000,
        "avg_reward": sum(results.reward) / len(results.reward),
    }
    for k in results.metrics:
        metadata[f"avg_{k}"] = sum(results.metrics[k]) / len(results.metrics[k])

    return metadata

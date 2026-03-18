"""Generate ARC-1D dataset from reasoning-gym and push to HuggingFace.

Wraps 1D arrays as single-row 2D grids to match bhoy/arc-agi-2 schema,
so existing grid utilities (format_grid, grid_similarity, etc.) work unchanged.

Usage:
    pip install reasoning-gym datasets huggingface_hub
    python scripts/generate_arc1d_dataset.py
"""

import json
import reasoning_gym
from datasets import Dataset


HF_REPO = "bhoy/arc-1d"
DATASET_SIZE = 5000
SEED = 42


def wrap_1d_as_2d(arr: list[int]) -> list[list[int]]:
    """Wrap a 1D array as a single-row 2D grid."""
    return [arr]


def main():
    print(f"Generating {DATASET_SIZE} arc_1d examples...")
    data = reasoning_gym.create_dataset("arc_1d", size=DATASET_SIZE, seed=SEED)

    records = []
    for i, example in enumerate(data):
        metadata = example["metadata"]
        train_examples = metadata["train_examples"]
        test_example = metadata["test_example"]
        task_name = metadata.get("task_name", "unknown")

        train_pairs = [
            {"input": wrap_1d_as_2d(ex["input"]), "output": wrap_1d_as_2d(ex["output"])}
            for ex in train_examples
        ]
        test_input = wrap_1d_as_2d(test_example["input"])
        test_output = wrap_1d_as_2d(test_example["output"])

        records.append({
            "train_pairs": json.dumps(train_pairs),
            "test_input": json.dumps(test_input),
            "test_output": json.dumps(test_output),
            "task_id": f"{task_name}_{i}",
        })

    print(f"Generated {len(records)} records")

    ds = Dataset.from_list(records)
    print(f"Dataset: {ds}")
    print(f"Sample record:\n{json.dumps(records[0], indent=2)}")

    print(f"\nPushing to {HF_REPO}...")
    ds.push_to_hub(HF_REPO, split="train")
    print("Done!")


if __name__ == "__main__":
    main()

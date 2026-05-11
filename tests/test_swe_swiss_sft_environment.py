import importlib.util
import json
from pathlib import Path

from datasets import Dataset


ENV_PATH = (
    Path(__file__).resolve().parents[1]
    / "environments"
    / "swe_swiss_sft"
    / "swe_swiss_sft.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("swe_swiss_sft_env", ENV_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_messages():
    return [
        {"role": "system", "content": "You are a repair agent."},
        {"role": "user", "content": "Fix the failing test."},
        {
            "role": "assistant",
            "content": "### app.py\n<<<<<<< SEARCH\nbad\n=======\ngood\n>>>>>>> REPLACE",
        },
    ]


def sample_dataset(messages_as_json=False) -> Dataset:
    messages = sample_messages()
    return Dataset.from_list(
        [
            {
                "messages": json.dumps(messages) if messages_as_json else messages,
                "__index_level_0__": 7,
            }
        ]
    )


def test_parse_messages_accepts_list_and_json_string():
    swe_swiss = load_module()

    assert swe_swiss.parse_messages(sample_messages()) == sample_messages()
    assert swe_swiss.parse_messages(json.dumps(sample_messages())) == sample_messages()


def test_parse_messages_skips_null_content():
    swe_swiss = load_module()
    messages = [
        {"role": "system", "content": None},
        {"role": "user", "content": "Fix this."},
        {"role": "assistant", "content": "Done."},
    ]

    parsed = swe_swiss.parse_messages(messages)

    assert parsed == [
        {"role": "user", "content": "Fix this."},
        {"role": "assistant", "content": "Done."},
    ]
    assert all(message["content"] != "None" for message in parsed)


def test_row_to_example_uses_prompt_before_first_assistant():
    swe_swiss = load_module()
    row = sample_dataset()[0]

    example = swe_swiss.row_to_example(row)

    assert example["prompt"] == sample_messages()[:2]
    assert example["question"] == "Fix the failing test."
    assert ">>>>>>> REPLACE" in example["answer"]
    assert example["info"]["benchmark"] == "swe-swiss"
    assert example["info"]["source_index"] == 7


def test_load_dataset_uses_requested_hf_dataset_and_limits(monkeypatch):
    swe_swiss = load_module()
    calls = {}

    def fake_load_dataset(name, split):
        calls["name"] = name
        calls["split"] = split
        return sample_dataset(messages_as_json=True)

    monkeypatch.setattr(swe_swiss, "load_dataset", fake_load_dataset)

    dataset = swe_swiss.load_swe_swiss_dataset(
        dataset_name="SWE-Swiss/SWESwiss-SFT-Merged-10K",
        split="train",
        num_examples=1,
        shuffle_seed=None,
    )

    assert calls == {"name": "SWE-Swiss/SWESwiss-SFT-Merged-10K", "split": "train"}
    assert len(dataset) == 1
    assert {"prompt", "question", "answer", "info"}.issubset(dataset.column_names)


def test_similarity_reward_scores_exact_higher_than_wrong_answer():
    swe_swiss = load_module()
    answer = "replace bad with good"

    exact = swe_swiss.normalized_similarity_reward(
        [{"role": "assistant", "content": "replace bad with good"}],
        answer,
    )
    wrong = swe_swiss.normalized_similarity_reward(
        [{"role": "assistant", "content": "unrelated text"}],
        answer,
    )

    assert exact == 1.0
    assert wrong < exact


def test_environment_loads_without_building_dataset():
    swe_swiss = load_module()

    env = swe_swiss.load_environment(num_train_examples=1, num_eval_examples=1)

    assert env.system_prompt == swe_swiss.SYSTEM_PROMPT
    assert callable(env.dataset_source)
    assert callable(env.eval_dataset_source)

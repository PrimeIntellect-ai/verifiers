import importlib.util
from pathlib import Path

from datasets import Dataset


ENV_PATH = (
    Path(__file__).resolve().parents[1]
    / "environments"
    / "gsm_infinite"
    / "gsm_infinite.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("gsm_infinite_env", ENV_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "problem": "There are 2 foxes.",
                "question": "How many foxes are there?",
                "solution": "Define foxes as F. F = 2. Answer: 2.",
                "op": 2,
                "id": 4,
                "length": "zero_context",
                "d": 2,
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Problem: There are 2 foxes."},
                    {"role": "user", "content": "Question: How many foxes? Solution:"},
                ],
            }
        ]
    )


def test_dataset_name_and_split_validation():
    gsm_infinite = load_module()

    assert (
        gsm_infinite.dataset_name("medium", "8k")
        == "InfiniAILab/gsm_infinite_medium_8k"
    )
    assert (
        gsm_infinite.dataset_name("symbolic", "0")
        == "InfiniAILab/gsm_infinite_symbolic_0"
    )
    assert (
        gsm_infinite.dataset_name("symbolic", "8k")
        == "InfiniAILab/gsm_infinite_symbolic_8k"
    )
    assert gsm_infinite.split_name(12) == "ops_12"


def test_row_to_example_maps_messages_and_answer():
    gsm_infinite = load_module()
    row = sample_dataset()[0]

    example = gsm_infinite.row_to_example(row)

    assert "Problem: There are 2 foxes." in example["question"]
    assert "How many foxes?" in example["question"]
    assert example["answer"] == "2"
    assert example["info"]["benchmark"] == "gsm-infinite"


def test_row_to_example_falls_back_when_answer_list_is_none():
    gsm_infinite = load_module()
    row = dict(sample_dataset()[0])
    row["answer_list"] = None
    row["solution"] = "Solve it. Answer: 7."

    example = gsm_infinite.row_to_example(row)

    assert example["answer"] == "7"


def test_load_dataset_uses_hf_name_split_and_limits(monkeypatch):
    gsm_infinite = load_module()
    calls = {}

    def fake_load_dataset(name, split):
        calls["name"] = name
        calls["split"] = split
        return sample_dataset()

    monkeypatch.setattr(gsm_infinite, "load_dataset", fake_load_dataset)

    dataset = gsm_infinite.load_gsm_infinite_dataset(
        subset="medium",
        context_length="0",
        operations=2,
        num_examples=1,
    )

    assert calls == {"name": "InfiniAILab/gsm_infinite_medium_0", "split": "ops_2"}
    assert len(dataset) == 1
    assert {"question", "answer", "info"}.issubset(dataset.column_names)


def test_reward_accepts_answer_prefix_and_exact_value():
    gsm_infinite = load_module()
    completion = [{"role": "assistant", "content": "Reasoning...\nAnswer: 2."}]

    assert gsm_infinite.exact_answer_reward(completion, "2") == 1.0
    assert gsm_infinite.exact_answer_reward(completion, "3") == 0.0


def test_reward_preserves_multi_value_list_answers():
    gsm_infinite = load_module()
    completion = [{"role": "assistant", "content": "Reasoning...\nAnswer: 4, 9."}]

    assert gsm_infinite.extract_answer(["4", "9"]) == "4, 9"
    assert gsm_infinite.exact_answer_reward(completion, "4, 9") == 1.0
    assert gsm_infinite.exact_answer_reward(completion, "9") == 0.0


def test_extract_answer_uses_answer_prefix_without_greedy_fallback():
    gsm_infinite = load_module()

    assert gsm_infinite.extract_answer("work\nAnswer: 2. Therefore done") == "2"
    assert gsm_infinite.extract_answer("work\nANSWER: V670487.") == "V670487"
    assert (
        gsm_infinite.extract_answer("I think answer: 5 is wrong.\nFinal Answer: 10.")
        == "10"
    )


def test_environment_loads_without_building_dataset():
    gsm_infinite = load_module()

    env = gsm_infinite.load_environment(num_train_examples=1, num_eval_examples=1)

    assert env.system_prompt == gsm_infinite.SYSTEM_PROMPT
    assert callable(env.dataset_source)
    assert callable(env.eval_dataset_source)


def test_environment_preserves_explicit_zero_eval_operations(monkeypatch):
    gsm_infinite = load_module()
    calls = []

    def fake_load_dataset(name, split):
        calls.append(split)
        return sample_dataset()

    monkeypatch.setattr(gsm_infinite, "load_dataset", fake_load_dataset)
    env = gsm_infinite.load_environment(
        operations=2,
        eval_operations=0,
        num_train_examples=1,
        num_eval_examples=1,
    )

    env.eval_dataset_source()

    assert calls == ["ops_0"]

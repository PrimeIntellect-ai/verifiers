import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "environments"
    / "kimi_k2_tool_sim"
    / "kimi_k2_tool_sim.py"
)
SPEC = importlib.util.spec_from_file_location("kimi_k2_tool_sim", MODULE_PATH)
assert SPEC is not None
kimi_k2_tool_sim = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(kimi_k2_tool_sim)


def test_build_dataset_has_expected_columns():
    dataset = kimi_k2_tool_sim.build_dataset()

    assert len(dataset) == 4
    assert set(dataset.column_names) == {"question", "answer", "info"}
    assert "Available tools" in dataset[0]["question"]
    assert "tool_calls" in json.loads(dataset[0]["answer"])


def test_extract_json_object_accepts_fenced_json():
    parsed = kimi_k2_tool_sim.extract_json_object(
        """```json
{"tool_calls": [{"name": "orders.find", "arguments": {"email": "sam@example.com"}}], "answer": "done"}
```"""
    )

    assert parsed == {
        "tool_calls": [
            {"name": "orders.find", "arguments": {"email": "sam@example.com"}}
        ],
        "answer": "done",
    }


def test_tool_sequence_reward_exact_match():
    dataset = kimi_k2_tool_sim.build_dataset(1)
    answer = dataset[0]["answer"]
    completion = [{"role": "assistant", "content": answer}]

    assert kimi_k2_tool_sim.tool_sequence_reward(completion, answer) == 1.0


def test_tool_sequence_reward_gives_partial_argument_credit():
    dataset = kimi_k2_tool_sim.build_dataset(1)
    expected = json.loads(dataset[0]["answer"])
    expected["tool_calls"][0]["arguments"]["city"] = "Dallas"
    completion = json.dumps(expected)

    score = kimi_k2_tool_sim.tool_sequence_reward(completion, dataset[0]["answer"])

    assert 0.0 < score < 1.0


def test_tool_sequence_reward_counts_non_dict_arguments_as_misses():
    dataset = kimi_k2_tool_sim.build_dataset(1)
    expected = json.loads(dataset[0]["answer"])
    expected["tool_calls"][1]["arguments"] = None
    completion = json.dumps(expected)

    score = kimi_k2_tool_sim.tool_sequence_reward(completion, dataset[0]["answer"])

    assert score < 1.0


def test_tool_sequence_reward_rejects_bad_json():
    dataset = kimi_k2_tool_sim.build_dataset(1)

    assert (
        kimi_k2_tool_sim.tool_sequence_reward("not json", dataset[0]["answer"]) == 0.0
    )


def test_load_environment_returns_single_turn_env():
    env = kimi_k2_tool_sim.load_environment(num_train_examples=2, num_eval_examples=1)

    assert env.dataset is not None
    assert len(env.dataset) == 2
    assert env.eval_dataset is not None
    assert len(env.eval_dataset) == 1

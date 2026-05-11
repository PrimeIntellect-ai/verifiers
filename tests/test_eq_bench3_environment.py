import importlib.util
import json
from pathlib import Path


ENV_PATH = (
    Path(__file__).resolve().parents[1] / "environments" / "eq_bench3" / "eq_bench3.py"
)
GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "environments"
    / "eq_bench3"
    / "generate_eq_bench3_prompts.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eq_bench3_generator_is_deterministic_and_structured(tmp_path):
    generator = load_module(GENERATOR_PATH, "eq_bench3_generator")

    rows = generator.generate_rows(num_examples=8, seed=321)
    rows_again = generator.generate_rows(num_examples=8, seed=321)

    assert rows == rows_again
    assert len(rows) == 8
    assert {row["info"]["source"] for row in rows} == {"synthetic-uncontaminated-v1"}
    assert all("Return only JSON" in row["question"] for row in rows)

    output = tmp_path / "sample.jsonl"
    generator.write_jsonl(rows, output)
    assert output.read_text(encoding="utf-8").count("\n") == 8


def test_eq_bench3_generator_rejects_too_many_examples():
    generator = load_module(GENERATOR_PATH, "eq_bench3_generator_limit")

    try:
        generator.generate_rows(num_examples=97, seed=321)
    except ValueError as exc:
        assert "num_examples must be <= 96" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_eq_bench3_sample_dataset_loads():
    eq_bench3 = load_module(ENV_PATH, "eq_bench3_env")

    dataset = eq_bench3.build_dataset()

    assert len(dataset) == 64
    assert {"question", "answer", "info"}.issubset(dataset.column_names)
    first = dataset[0]
    assert first["info"]["benchmark"] == "eq-bench3"
    assert first["info"]["source"] == "synthetic-uncontaminated-v1"
    assert isinstance(json.loads(first["answer"]), dict)


def test_eq_bench3_reward_scores_exact_json_as_one():
    eq_bench3 = load_module(ENV_PATH, "eq_bench3_reward")
    answer = '{"angry": 7, "sad": 3}'
    completion = [{"role": "assistant", "content": '{"angry": 7, "sad": 3}'}]

    assert eq_bench3.emotion_score_reward(completion, answer) == 1.0


def test_eq_bench3_environment_loads_without_building_dataset():
    eq_bench3 = load_module(ENV_PATH, "eq_bench3_env_load")

    env = eq_bench3.load_environment()

    assert env.system_prompt == eq_bench3.SYSTEM_PROMPT
    assert callable(env.dataset_source)

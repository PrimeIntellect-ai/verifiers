import importlib.util
from pathlib import Path


ENV_PATH = (
    Path(__file__).resolve().parents[1]
    / "environments"
    / "spiral_bench"
    / "spiral_bench.py"
)
GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "environments"
    / "spiral_bench"
    / "generate_spiral_prompts.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spiral_prompt_generator_is_deterministic_and_structured(tmp_path):
    generator = load_module(GENERATOR_PATH, "spiral_prompt_generator")

    rows = generator.generate_rows(num_examples=8, seed=123)
    rows_again = generator.generate_rows(num_examples=8, seed=123)

    assert rows == rows_again
    assert len(rows) == 8
    assert {row["info"]["source"] for row in rows} == {"synthetic-uncontaminated-v1"}
    assert all(
        "Spiral-Bench style safety conversation" in row["question"] for row in rows
    )

    output = tmp_path / "sample.jsonl"
    generator.write_jsonl(rows, output)
    assert output.read_text(encoding="utf-8").count("\n") == 8


def test_spiral_prompt_generator_rejects_too_many_examples():
    generator = load_module(GENERATOR_PATH, "spiral_prompt_generator_limit")

    try:
        generator.generate_rows(num_examples=601, seed=123)
    except ValueError as exc:
        assert "num_examples must be <= 600" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_spiral_bench_sample_dataset_loads():
    spiral_bench = load_module(ENV_PATH, "spiral_bench_env")

    dataset = spiral_bench.build_dataset()

    assert len(dataset) == 64
    assert {"question", "answer", "info"}.issubset(dataset.column_names)
    first = dataset[0]
    assert first["info"]["benchmark"] == "spiral-bench"
    assert first["info"]["source"] == "synthetic-uncontaminated-v1"


def test_spiral_bench_environment_loads_without_api_call(monkeypatch):
    spiral_bench = load_module(ENV_PATH, "spiral_bench_env_load")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    env = spiral_bench.load_environment()

    assert env.system_prompt == spiral_bench.SYSTEM_PROMPT
    assert callable(env.dataset_source)

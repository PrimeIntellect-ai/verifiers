import os
import json
from pathlib import Path

from verifiers.utils.path_utils import (
    find_latest_incomplete_eval_results_path,
    is_valid_eval_results_path,
)


def test_find_latest_incomplete_eval_results_path_picks_newest_matching(
    tmp_path: Path, monkeypatch
):
    env_id = "dummy-env"
    model = "openai/gpt-4.1-mini"
    runs_dir = tmp_path / "outputs" / "evals" / f"{env_id}--{model.replace('/', '--')}"

    old_run = runs_dir / "11111111"
    new_run = runs_dir / "22222222"
    complete_run = runs_dir / "33333333"
    for run in [old_run, new_run, complete_run]:
        run.mkdir(parents=True)

    metadata = (
        '{"env_id":"dummy-env","model":"openai/gpt-4.1-mini",'
        '"num_examples":4,"rollouts_per_example":1}'
    )
    for run in [old_run, new_run, complete_run]:
        (run / "metadata.json").write_text(metadata, encoding="utf-8")

    (old_run / "results.jsonl").write_text('{"example_id":0}\n', encoding="utf-8")
    (new_run / "results.jsonl").write_text(
        '{"example_id":0}\n{"example_id":1}\n', encoding="utf-8"
    )
    (complete_run / "results.jsonl").write_text(
        '{"example_id":0}\n{"example_id":1}\n{"example_id":2}\n{"example_id":3}\n',
        encoding="utf-8",
    )

    os.utime(old_run, (1, 1))
    os.utime(new_run, (2, 2))
    os.utime(complete_run, (3, 3))

    monkeypatch.chdir(tmp_path)

    result = find_latest_incomplete_eval_results_path(
        env_id=env_id,
        model=model,
        num_examples=4,
        rollouts_per_example=1,
        env_dir_path=str(tmp_path / "environments"),
    )

    assert result is not None
    assert result.resolve() == new_run.resolve()


def test_find_latest_incomplete_eval_results_path_returns_none_when_no_match(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    result = find_latest_incomplete_eval_results_path(
        env_id="dummy-env",
        model="openai/gpt-4.1-mini",
        num_examples=4,
        rollouts_per_example=1,
        env_dir_path=str(tmp_path / "environments"),
    )
    assert result is None


def test_find_latest_incomplete_eval_results_path_filters_offline_mode_and_source(
    tmp_path: Path, monkeypatch
):
    env_id = "dummy-env"
    model = "openai/gpt-4.1-mini"
    runs_dir = tmp_path / "outputs" / "evals" / f"{env_id}--{model.replace('/', '--')}"
    wrong_source_run = runs_dir / "11111111"
    correct_source_run = runs_dir / "22222222"
    wrong_mode_run = runs_dir / "33333333"
    for run in [wrong_source_run, correct_source_run, wrong_mode_run]:
        run.mkdir(parents=True)

    prepared_a = tmp_path / "prepared-a.jsonl"
    prepared_b = tmp_path / "prepared-b.jsonl"
    prepared_a.write_text("", encoding="utf-8")
    prepared_b.write_text("", encoding="utf-8")

    (wrong_source_run / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": "dummy-env",
                "model": "openai/gpt-4.1-mini",
                "num_examples": 4,
                "rollouts_per_example": 1,
                "offline_mode": "prepared_completions",
                "prepared_completions_path": str(prepared_a.resolve()),
            }
        ),
        encoding="utf-8",
    )
    (correct_source_run / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": "dummy-env",
                "model": "openai/gpt-4.1-mini",
                "num_examples": 4,
                "rollouts_per_example": 1,
                "offline_mode": "prepared_completions",
                "prepared_completions_path": str(prepared_b.resolve()),
            }
        ),
        encoding="utf-8",
    )
    (wrong_mode_run / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": "dummy-env",
                "model": "openai/gpt-4.1-mini",
                "num_examples": 4,
                "rollouts_per_example": 1,
                "offline_mode": "ground_truth",
                "prepared_completions_path": None,
            }
        ),
        encoding="utf-8",
    )

    for run in [wrong_source_run, correct_source_run, wrong_mode_run]:
        (run / "results.jsonl").write_text('{"example_id":0}\n', encoding="utf-8")

    os.utime(wrong_source_run, (1, 1))
    os.utime(correct_source_run, (2, 2))
    os.utime(wrong_mode_run, (3, 3))

    monkeypatch.chdir(tmp_path)

    result = find_latest_incomplete_eval_results_path(
        env_id=env_id,
        model=model,
        num_examples=4,
        rollouts_per_example=1,
        env_dir_path=str(tmp_path / "environments"),
        offline_mode="prepared_completions",
        prepared_completions_path=prepared_b,
    )

    assert result is not None
    assert result.resolve() == correct_source_run.resolve()


def test_is_valid_eval_results_path_requires_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "results.jsonl").mkdir()
    (run_dir / "metadata.json").mkdir()

    assert not is_valid_eval_results_path(run_dir)


def test_is_valid_eval_results_path_accepts_expected_layout(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "results.jsonl").write_text("", encoding="utf-8")
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")

    assert is_valid_eval_results_path(run_dir)

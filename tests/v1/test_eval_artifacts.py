import json

import pytest

from verifiers.v1.cli.output import (
    convert_results_for_upload,
    discover_eval_artifact_dirs,
    has_eval_artifacts,
    read_upload_data,
    resolve_eval_artifact_dir,
)


def test_convert_results_for_upload_supports_v1_and_legacy_samples():
    samples = [
        {"id": 3, "reward": 0.5, "completion": "legacy"},
        {
            "id": "trace-1",
            "task": {"idx": 7, "prompt": "What is 2 + 2?", "answer": "4"},
            "nodes": [
                {
                    "message": {"role": "user", "content": "What is 2 + 2?"},
                    "token_ids": [1, 2],
                    "mask": [False, False],
                    "is_content": [True, True],
                },
                {
                    "parent": 0,
                    "message": {"role": "assistant", "content": "4"},
                    "sampled": True,
                    "token_ids": [3, 4],
                    "mask": [False, True],
                    "is_content": [False, True],
                    "logprobs": [-0.1],
                },
            ],
            "rewards": {"correct": 1.0},
            "metrics": {"exact_match": 1.0},
            "info": {"source": "test"},
            "is_completed": True,
            "grader_note": "kept as metadata",
        },
    ]

    legacy, v1 = convert_results_for_upload(samples)

    assert legacy == {
        "id": 3,
        "example_id": 3,
        "reward": 0.5,
        "completion": "legacy",
    }
    assert v1["sample_id"] == "trace-1"
    assert v1["example_id"] == 7
    assert v1["rollout_number"] == 1
    assert v1["answer"] == "4"
    assert v1["reward"] == 1.0
    assert v1["completion"][-1] == {"role": "assistant", "content": "4"}
    assert v1["trajectory"] == [
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"},
            ],
            "reward": 1.0,
            "num_input_tokens": 3,
            "num_output_tokens": 1,
        }
    ]
    assert v1["info"] == {"source": "test", "grader_note": "kept as metadata"}


def test_convert_results_for_upload_uses_provider_token_counts_without_token_ids():
    samples = [
        {
            "id": "trace-usage",
            "task": {"idx": 1, "prompt": "Question"},
            "nodes": [
                {"message": {"role": "user", "content": "Question"}},
                {
                    "parent": 0,
                    "message": {"role": "assistant", "content": "Answer"},
                    "sampled": True,
                    "usage": {"prompt_tokens": 11, "completion_tokens": 3},
                },
            ],
            "rewards": {"correct": 1.0},
        }
    ]

    (converted,) = convert_results_for_upload(samples)

    assert converted["trajectory"][0]["num_input_tokens"] == 11
    assert converted["trajectory"][0]["num_output_tokens"] == 3


def test_eval_artifact_path_helpers_support_native_and_legacy(tmp_path):
    native = tmp_path / "outputs" / "native"
    native.mkdir(parents=True)
    (native / "config.toml").write_text('model = "test-model"\n')
    (native / "results.jsonl").write_text("")

    legacy = tmp_path / "outputs" / "legacy"
    legacy.mkdir()
    (legacy / "metadata.json").write_text("{}")
    (legacy / "results.jsonl").write_text("")

    assert has_eval_artifacts(native) is True
    assert has_eval_artifacts(legacy) is True
    assert resolve_eval_artifact_dir(native / "config.toml") == native
    assert discover_eval_artifact_dirs(tmp_path / "outputs") == [legacy, native]


def test_resolve_eval_artifact_dir_reports_incomplete_artifacts(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")

    with pytest.raises(ValueError, match="missing results.jsonl"):
        resolve_eval_artifact_dir(tmp_path)

    with pytest.raises(
        ValueError, match="must contain both metadata.json and results.jsonl"
    ):
        resolve_eval_artifact_dir(tmp_path / "metadata.json")


def test_read_upload_data_supports_native_runs(tmp_path):
    (tmp_path / "config.toml").write_text(
        """
model = "gpt-4"
num_tasks = 2
num_rollouts = 3

[taskset]
id = "gsm8k-v1"
""".lstrip()
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"id": 1, "reward": 1.0}) + "\n"
        "not json\n" + json.dumps({"id": 2, "reward": 0.0}) + "\n"
    )

    upload = read_upload_data(tmp_path)

    assert upload.eval_name == "gsm8k-v1-gpt-4"
    assert upload.model_name == "gpt-4"
    assert upload.env == "gsm8k-v1"
    assert upload.metrics == {"reward": 0.5}
    assert upload.metadata["framework"] == "verifiers"
    assert upload.metadata["num_examples"] == 2
    assert upload.metadata["rollouts_per_example"] == 3
    assert upload.results[0]["example_id"] == 1
    assert len(upload.invalid_results) == 1
    assert upload.as_dict()["results"] == upload.results


def test_read_upload_data_supports_legacy_runs(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"env": "owner/gsm8k", "model": "gpt-4", "avg_reward": 0.75})
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"id": 1, "reward": 0.75}) + "\n"
    )

    upload = read_upload_data(tmp_path)

    assert upload.eval_name == "owner/gsm8k-gpt-4"
    assert upload.metrics == {"reward": 0.75}
    assert upload.metadata == {"env": "owner/gsm8k", "model": "gpt-4"}
    assert upload.results == [{"id": 1, "reward": 0.75, "example_id": 1}]

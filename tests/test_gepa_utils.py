import json
from types import SimpleNamespace

from verifiers.gepa.gepa_utils import save_gepa_results


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_save_gepa_results_writes_best_system_prompt_verbatim(tmp_path):
    prompt = "System line one.\n\nKeep this exact spacing.\n"
    result = SimpleNamespace(
        best_idx=0,
        best_candidate={"system_prompt": prompt},
        val_aggregate_scores=[1.0],
    )

    save_gepa_results(tmp_path, result)

    assert (tmp_path / "system_prompt.txt").read_text(encoding="utf-8") == prompt
    assert not (tmp_path / "best_prompt.txt").exists()


def test_save_gepa_results_handles_none_best_score(tmp_path):
    prompt = "System prompt.\n"
    result = SimpleNamespace(
        best_idx=0,
        best_candidate={"system_prompt": prompt},
        val_aggregate_scores=[None],
    )

    save_gepa_results(tmp_path, result)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(tmp_path / "results.jsonl")
    assert metadata["best_score"] is None
    assert rows[0]["score"] is None
    assert rows[0]["reward"] == 0.0


def test_save_gepa_results_writes_upload_schema_without_task_fields(tmp_path):
    prompt = "Answer with one word.\n"
    result = SimpleNamespace(
        best_idx=0,
        best_candidate={"system_prompt": prompt},
        val_aggregate_scores=[0.75],
    )

    save_gepa_results(
        tmp_path,
        result,
        config={
            "env_id": "wordle",
            "model": "openai/gpt-5.4-mini",
            "reflection_model": "openai/gpt-5.4-mini",
        },
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "verifiers.gepa.v1"
    assert metadata["eval_kind"] == "gepa"
    assert metadata["optimization_target"] == "system_prompt"
    assert metadata["env_id"] == "wordle"
    assert metadata["best_score"] == 0.75
    assert "task" not in metadata
    assert "task_type" not in metadata

    rows = _read_jsonl(tmp_path / "results.jsonl")
    assert rows == [
        {
            "example_id": 0,
            "reward": 0.75,
            "score": 0.75,
            "info": {
                "schema_version": "verifiers.gepa.v1",
                "eval_kind": "gepa",
                "sample_type": "gepa_candidate",
                "optimization_target": "system_prompt",
                "candidate_idx": 0,
                "is_best": True,
                "system_prompt": prompt,
                "system_prompt_sha256": metadata["best_prompt_sha256"],
                "diff_from_initial": "",
                "parent_candidate_idxs": [],
                "val_subscores": {},
                "num_val_examples": 0,
            },
        }
    ]
    assert "task" not in rows[0]
    assert "task_type" not in rows[0]
    assert "task" not in rows[0]["info"]
    assert "task_type" not in rows[0]["info"]


def test_save_gepa_results_writes_candidate_diffs_and_frontier(tmp_path):
    result = SimpleNamespace(
        best_idx=1,
        best_candidate={"system_prompt": "Base prompt.\nBetter final answer.\n"},
        candidates=[
            {"system_prompt": "Base prompt.\nInitial final answer.\n"},
            {"system_prompt": "Base prompt.\nBetter final answer.\n"},
        ],
        val_aggregate_scores=[0.3, 0.7],
        val_subscores=[{0: 0.2, 1: 0.4}, {0: 0.8, 1: 0.6}],
        parents=[None, 0],
        discovery_eval_counts=[2, 5],
    )

    save_gepa_results(tmp_path, result)

    rows = _read_jsonl(tmp_path / "results.jsonl")
    assert len(rows) == 2
    assert rows[1]["reward"] == 0.7
    assert rows[1]["info"]["parent_candidate_idxs"] == [0]
    assert rows[1]["info"]["val_subscores"] == {"0": 0.8, "1": 0.6}
    assert rows[1]["info"]["discovery_metric_calls"] == 5
    assert "--- initial/system_prompt.txt" in rows[1]["info"]["diff_from_initial"]
    assert "+++ candidate/system_prompt.txt" in rows[1]["info"]["diff_from_initial"]

    frontier_rows = _read_jsonl(tmp_path / "pareto_frontier.jsonl")
    assert frontier_rows == [
        {
            "schema_version": "verifiers.gepa.v1",
            "valset_row": 0,
            "best_score": 0.8,
            "num_best_candidates": 1,
            "best_candidates": [
                {
                    "candidate_idx": 1,
                    "system_prompt_sha256": rows[1]["info"]["system_prompt_sha256"],
                    "score": 0.8,
                }
            ],
        },
        {
            "schema_version": "verifiers.gepa.v1",
            "valset_row": 1,
            "best_score": 0.6,
            "num_best_candidates": 1,
            "best_candidates": [
                {
                    "candidate_idx": 1,
                    "system_prompt_sha256": rows[1]["info"]["system_prompt_sha256"],
                    "score": 0.6,
                }
            ],
        },
    ]
    assert "system_prompt" not in frontier_rows[0]["best_candidates"][0]

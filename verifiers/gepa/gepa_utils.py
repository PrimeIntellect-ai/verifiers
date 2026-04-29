"""
Simple artifact saving for GEPA optimization.

Saves:
- results.jsonl: Candidate-centric rows suitable for Prime Evals upload
- pareto_frontier.jsonl: Per valset row, the best prompt(s) and their scores
- system_prompt.txt: The single best overall system prompt
- metadata.json: Run configuration and summary
"""

import difflib
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

GEPA_SCHEMA_VERSION = "verifiers.gepa.v1"
GEPA_EVAL_KIND = "gepa"
GEPA_OPTIMIZATION_TARGET = "system_prompt"


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _prompt_from_candidate(candidate: Any) -> str:
    if isinstance(candidate, dict):
        value = candidate.get("system_prompt", "")
        return value if isinstance(value, str) else str(value)
    if isinstance(candidate, str):
        return candidate
    return ""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_score(
    candidate_idx: int,
    val_scores: list[Any],
    val_subscores: list[dict[Any, Any]],
) -> float | None:
    if candidate_idx < len(val_scores):
        return _to_float(val_scores[candidate_idx])
    if candidate_idx >= len(val_subscores):
        return None
    scores = [_to_float(score) for score in val_subscores[candidate_idx].values()]
    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return None
    return sum(valid_scores) / len(valid_scores)


def _prompt_diff(initial_prompt: str, candidate_prompt: str) -> str:
    if initial_prompt == candidate_prompt:
        return ""
    diff = difflib.unified_diff(
        initial_prompt.splitlines(),
        candidate_prompt.splitlines(),
        fromfile="initial/system_prompt.txt",
        tofile="candidate/system_prompt.txt",
        lineterm="",
    )
    return "\n".join(diff) + "\n"


def _normalize_parents(parents: Any, candidate_idx: int) -> list[int]:
    if not isinstance(parents, list) or candidate_idx >= len(parents):
        return []
    raw_parent_idxs = parents[candidate_idx]
    if raw_parent_idxs is None:
        return []
    if not isinstance(raw_parent_idxs, list):
        raw_parent_idxs = [raw_parent_idxs]
    normalized = []
    for raw_parent_idx in raw_parent_idxs:
        if raw_parent_idx is None:
            continue
        try:
            normalized.append(int(raw_parent_idx))
        except (TypeError, ValueError):
            continue
    return normalized


def _jsonable_scores(scores: dict[Any, Any]) -> dict[str, float]:
    jsonable = {}
    for key, value in scores.items():
        score = _to_float(value)
        if score is not None:
            jsonable[str(key)] = score
    return jsonable


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_candidate_records(
    candidates: list[Any],
    val_scores: list[Any],
    val_subscores: list[dict[Any, Any]],
    parents: Any,
    discovery_eval_counts: Any,
    best_idx: int,
) -> list[dict[str, Any]]:
    initial_prompt = _prompt_from_candidate(candidates[0]) if candidates else ""
    records = []
    for candidate_idx, candidate in enumerate(candidates):
        system_prompt = _prompt_from_candidate(candidate)
        score = _candidate_score(candidate_idx, val_scores, val_subscores)
        subscores = (
            _jsonable_scores(val_subscores[candidate_idx])
            if candidate_idx < len(val_subscores)
            else {}
        )
        info: dict[str, Any] = {
            "schema_version": GEPA_SCHEMA_VERSION,
            "eval_kind": GEPA_EVAL_KIND,
            "sample_type": "gepa_candidate",
            "optimization_target": GEPA_OPTIMIZATION_TARGET,
            "candidate_idx": candidate_idx,
            "is_best": candidate_idx == best_idx,
            "system_prompt": system_prompt,
            "system_prompt_sha256": _sha256_text(system_prompt),
            "diff_from_initial": _prompt_diff(initial_prompt, system_prompt),
            "parent_candidate_idxs": _normalize_parents(parents, candidate_idx),
            "val_subscores": subscores,
            "num_val_examples": len(subscores),
        }
        if isinstance(discovery_eval_counts, list) and candidate_idx < len(
            discovery_eval_counts
        ):
            info["discovery_metric_calls"] = discovery_eval_counts[candidate_idx]

        records.append(
            {
                "example_id": candidate_idx,
                "reward": score if score is not None else 0.0,
                "score": score,
                "info": info,
            }
        )
    return records


def save_gepa_results(
    run_dir: Path,
    result: Any,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Save GEPA optimization results to disk.

    Args:
        run_dir: Directory to save results
        result: Result from gepa.optimize()
        config: Optional run configuration dict
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load detailed state from gepa_state.bin (saved by GEPA library)
    state_file = run_dir / "gepa_state.bin"
    state = {}
    candidates = []
    val_subscores = []
    parents = []
    discovery_eval_counts = []

    if state_file.exists():
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        candidates = _get_field(state, "program_candidates", [])
        val_subscores = _get_field(state, "prog_candidate_val_subscores", [])
        parents = _get_field(state, "parent_program_for_candidate", [])
        discovery_eval_counts = _get_field(state, "num_metric_calls_by_discovery", [])

    # Fallback to result object if state file doesn't have data
    # Clear val_subscores too since they reference state file candidate indices
    if not candidates:
        candidates = getattr(result, "candidates", [])
        val_subscores = getattr(result, "val_subscores", [])
        parents = getattr(result, "parents", [])
        discovery_eval_counts = getattr(result, "discovery_eval_counts", [])

    try:
        best_idx = int(getattr(result, "best_idx", 0))
    except (TypeError, ValueError):
        best_idx = 0
    best_candidate = getattr(result, "best_candidate", {})
    candidates = _as_list(candidates)
    val_subscores = _as_list(val_subscores)
    if not candidates and best_candidate:
        candidates = [best_candidate]
    if not best_candidate and candidates and best_idx < len(candidates):
        best_candidate = candidates[best_idx]

    val_scores = _as_list(getattr(result, "val_aggregate_scores", None))

    # Build per-row frontier: for each valset row, which prompts did best
    records = []
    if val_subscores and candidates:
        # Get all valset row indices
        all_rows: set[int] = set()
        for subscores in val_subscores:
            if isinstance(subscores, dict):
                all_rows.update(subscores.keys())

        for row_idx in sorted(all_rows):
            # Collect scores for this row from all candidates
            row_scores = []
            for cand_idx, subscores in enumerate(val_subscores):
                if isinstance(subscores, dict) and row_idx in subscores:
                    score = _to_float(subscores[row_idx])
                    if score is not None:
                        row_scores.append((cand_idx, score))

            if not row_scores:
                continue

            best_score = max(score for _, score in row_scores)
            best_candidates = [
                {
                    "candidate_idx": cand_idx,
                    "system_prompt_sha256": _sha256_text(
                        _prompt_from_candidate(candidates[cand_idx])
                    ),
                    "score": score,
                }
                for cand_idx, score in row_scores
                if score == best_score
            ]

            records.append(
                {
                    "schema_version": GEPA_SCHEMA_VERSION,
                    "valset_row": int(row_idx),
                    "best_score": best_score,
                    "num_best_candidates": len(best_candidates),
                    "best_candidates": best_candidates,
                }
            )

    # Save upload-friendly candidate rows as JSONL.
    candidate_records = _build_candidate_records(
        candidates=candidates,
        val_scores=val_scores,
        val_subscores=val_subscores,
        parents=parents,
        discovery_eval_counts=discovery_eval_counts,
        best_idx=best_idx,
    )
    _write_jsonl(run_dir / "results.jsonl", candidate_records)

    # Save frontier as auxiliary JSONL.
    if records:
        _write_jsonl(run_dir / "pareto_frontier.jsonl", records)

    # Save best system prompt as plain text.
    system_prompt = _prompt_from_candidate(best_candidate)
    (run_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    # Build and save metadata
    metadata = {
        "schema_version": GEPA_SCHEMA_VERSION,
        "eval_kind": GEPA_EVAL_KIND,
        "framework": "verifiers",
        "optimizer": "gepa",
        "optimization_target": GEPA_OPTIMIZATION_TARGET,
        "env_id": config.get("env_id") if config else None,
        "model": config.get("model") if config else None,
        "reflection_model": config.get("reflection_model") if config else None,
        "num_candidates": len(candidates),
        "best_idx": best_idx,
        "best_score": float(val_scores[best_idx])
        if val_scores and best_idx < len(val_scores)
        else None,
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "initial_prompt_sha256": _sha256_text(_prompt_from_candidate(candidates[0]))
        if candidates
        else None,
        "best_prompt_sha256": _sha256_text(system_prompt),
        "system_prompt_path": "system_prompt.txt",
        "results_path": "results.jsonl",
        "completed_at": datetime.now().isoformat(),
    }
    if config:
        metadata["config"] = config

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

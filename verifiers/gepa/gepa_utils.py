"""
Simple artifact saving for GEPA optimization.

Saves:
- pareto_frontier.jsonl: Per valset row, the best prompt(s) and their scores
- best_prompt.txt: The single best overall system prompt
- metadata.json: Run configuration and summary
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset


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
    candidates = []
    val_subscores = []

    if state_file.exists():
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        candidates = state.get("program_candidates", [])
        val_subscores = state.get("prog_candidate_val_subscores", [])

    # Fallback to result object if state file doesn't have data
    # Clear val_subscores too since they reference state file candidate indices
    if not candidates:
        candidates = getattr(result, "candidates", [])
        val_subscores = []

    best_idx = getattr(result, "best_idx", 0)
    best_candidate = getattr(result, "best_candidate", {})

    # Build per-row frontier: for each valset row, which prompts did best
    records = []
    if val_subscores and candidates:
        # Get all valset row indices
        all_rows: set[int] = set()
        for subscores in val_subscores:
            all_rows.update(subscores.keys())

        for row_idx in sorted(all_rows):
            # Collect scores for this row from all candidates
            row_scores = []
            for cand_idx, subscores in enumerate(val_subscores):
                if row_idx in subscores:
                    row_scores.append((cand_idx, subscores[row_idx]))

            if not row_scores:
                continue

            best_score = max(score for _, score in row_scores)
            best_prompts = [
                {
                    "candidate_idx": cand_idx,
                    "system_prompt": candidates[cand_idx].get("system_prompt", ""),
                    "score": score,
                }
                for cand_idx, score in row_scores
                if score == best_score
            ]

            records.append(
                {
                    "valset_row": row_idx,
                    "best_score": best_score,
                    "num_best_prompts": len(best_prompts),
                    "best_prompts": best_prompts,
                }
            )

    # Save frontier as JSONL
    if records:
        frontier_ds = Dataset.from_list(records)
        frontier_ds.to_json(run_dir / "pareto_frontier.jsonl")

    # Save best prompt as plain text
    best_prompt = best_candidate.get("system_prompt", "")
    (run_dir / "best_prompt.txt").write_text(best_prompt)

    # Build and save metadata
    val_scores = getattr(result, "val_aggregate_scores", [])
    metadata = {
        "num_candidates": len(candidates),
        "best_idx": best_idx,
        "best_score": float(val_scores[best_idx])
        if val_scores and best_idx < len(val_scores)
        else None,
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "completed_at": datetime.now().isoformat(),
    }
    if config:
        metadata["config"] = config

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

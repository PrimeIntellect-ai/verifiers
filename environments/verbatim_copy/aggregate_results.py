#!/usr/bin/env python3
"""
Aggregate ablation results from verbatim_copy experiments.

Reads all results.jsonl files from outputs/evals/ and creates a summary CSV
with metrics grouped by ablation parameters.

Usage:
    python aggregate_results.py [--output results_summary.csv]
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_all_results(outputs_dir: Path) -> list[dict]:
    """Load all results from jsonl files in the outputs directory."""
    all_results = []

    # Find all results.jsonl files
    results_files = list(outputs_dir.glob("evals/**/results.jsonl"))

    if not results_files:
        print(f"No results found in {outputs_dir}")
        return []

    print(f"Found {len(results_files)} result files")

    for results_file in results_files:
        # Read metadata.json for model info
        metadata_file = results_file.parent / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        model = metadata.get("model", "unknown")

        with open(results_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    result["_model"] = model
                    all_results.append(result)

    print(f"Loaded {len(all_results)} total rollouts")
    return all_results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results list to a flat DataFrame."""
    rows = []

    for r in results:
        info = r.get("info", {})
        row = {
            # Model (from metadata)
            "model": r.get("_model", "unknown"),
            # Ablation parameters (from info)
            "mode": info.get(
                "mode", "standard"
            ),  # Inference mode: standard, rlm, rlm_tips
            "content_type": info.get("content_type"),
            "target_length": info.get("target_length"),
            "mean_fragment_length": info.get("mean_fragment_length"),
            "sample_id": info.get("id"),
            # Metrics
            "reward": r.get("reward"),
            "exact_match": r.get("exact_match"),
            "char_accuracy": r.get("char_accuracy"),
            "levenshtein_similarity": r.get("levenshtein_similarity"),
            # Timing
            "generation_ms": r.get("generation_ms"),
            "scoring_ms": r.get("scoring_ms"),
            "total_ms": r.get("total_ms"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by ablation parameters."""
    group_cols = [
        "model",
        "mode",
        "content_type",
        "target_length",
        "mean_fragment_length",
    ]
    metric_cols = ["reward", "exact_match", "char_accuracy", "levenshtein_similarity"]

    # Group and compute stats per content type
    summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(
        ["mean", "std", "count"]
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Also create "all" aggregation across content types
    # This aggregates results for each (model, mode, target_length, mean_fragment_length) combination
    all_group_cols = ["model", "mode", "target_length", "mean_fragment_length"]
    all_content_summary = df.groupby(all_group_cols, dropna=False)[metric_cols].agg(
        ["mean", "std", "count"]
    )
    all_content_summary.columns = [
        "_".join(col).strip() for col in all_content_summary.columns.values
    ]
    all_content_summary = all_content_summary.reset_index()
    all_content_summary["content_type"] = "all"

    # Combine both summaries
    summary = pd.concat([summary, all_content_summary], ignore_index=True)

    return summary


def normalize_model_name(model: str) -> str:
    """Create display-friendly model name."""
    # Handle OpenRouter format: openrouter/provider/model-name
    if model.startswith("openrouter/"):
        parts = model.split("/")
        return parts[-1] if len(parts) >= 3 else model
    # Handle other formats like organization/model
    if "/" in model:
        return model.split("/")[-1]
    return model


def print_summary_table(summary: pd.DataFrame):
    """Print a nicely formatted summary table."""
    print("\n" + "=" * 120)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 120)

    # Sort by ablation parameters for readability
    summary = summary.sort_values(
        ["model", "mode", "content_type", "target_length", "mean_fragment_length"],
        na_position="first",
    )

    print(
        f"\n{'Model':<20} {'Mode':<10} {'Config':<35} {'Reward':>12} {'Exact Match':>12} {'Char Acc':>12}"
    )
    print("-" * 120)

    for _, row in summary.iterrows():
        model = normalize_model_name(row.get("model", "unknown"))
        mode = row.get("mode", "standard")
        content_type = row["content_type"] or "all"
        length = int(row["target_length"]) if pd.notna(row["target_length"]) else "?"
        frag = (
            int(row["mean_fragment_length"])
            if pd.notna(row["mean_fragment_length"])
            else "None"
        )

        config = f"type={content_type}, len={length}, frag={frag}"

        reward = f"{row['reward_mean']:.3f}±{row['reward_std']:.3f}"
        exact = f"{row['exact_match_mean']:.3f}±{row['exact_match_std']:.3f}"
        char_acc = f"{row['char_accuracy_mean']:.3f}±{row['char_accuracy_std']:.3f}"

        print(
            f"{model:<20} {mode:<10} {config:<35} {reward:>12} {exact:>12} {char_acc:>12}"
        )

    print("-" * 120)
    print(f"Total configurations: {len(summary)}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate verbatim_copy ablation results"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
        help="Path to outputs directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV file path (optional)",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Output CSV with all individual results (optional)",
    )
    args = parser.parse_args()

    # Load results
    results = load_all_results(args.outputs_dir)
    if not results:
        return

    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Save raw results if requested
    if args.raw_output:
        df.to_csv(args.raw_output, index=False)
        print(f"Raw results saved to: {args.raw_output}")

    # Compute summary
    summary = compute_summary(df)

    # Print summary
    print_summary_table(summary)

    # Save summary if output path provided
    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    main()

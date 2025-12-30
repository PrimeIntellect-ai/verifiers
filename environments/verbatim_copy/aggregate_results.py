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

        # Extract mode flags from env_args
        env_args = metadata.get("env_args", {})
        use_rlm = env_args.get("use_rlm", False)
        include_env_tips = env_args.get("include_env_tips", False)
        updated_tips = env_args.get("updated_tips", False)

        with open(results_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    result["_model"] = model
                    result["_use_rlm"] = use_rlm
                    result["_include_env_tips"] = include_env_tips
                    result["_updated_tips"] = updated_tips
                    all_results.append(result)

    print(f"Loaded {len(all_results)} total rollouts")
    return all_results


def get_mode(result: dict) -> str:
    """Get mode from metadata flags."""
    use_rlm = result.get("_use_rlm", False)
    include_env_tips = result.get("_include_env_tips", False)
    updated_tips = result.get("_updated_tips", False)

    if not use_rlm:
        return "standard"
    if not include_env_tips:
        return "rlm"
    # Tips are enabled - check if using updated tips
    if updated_tips:
        return "rlm_tips_v2"
    return "rlm_tips"


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results list to a flat DataFrame."""
    rows = []

    for r in results:
        info = r.get("info", {})
        row = {
            # Model (from metadata)
            "model": r.get("_model", "unknown"),
            # Mode (from metadata flags)
            "mode": get_mode(r),
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
            # Sub-LLM metrics (will be 0 for standard mode)
            "sub_llm_call_count": r.get("sub_llm_call_count", 0),
            "sub_llm_prompt_tokens": r.get("sub_llm_prompt_tokens", 0),
            "sub_llm_completion_tokens": r.get("sub_llm_completion_tokens", 0),
            "sub_llm_total_tool_calls": r.get("sub_llm_total_tool_calls", 0),
            "sub_llm_total_turns": r.get("sub_llm_total_turns", 0),
            "sub_llm_batch_count": r.get("sub_llm_batch_count", 0),
            "sub_llm_max_batch_size": r.get("sub_llm_max_batch_size", 0),
            "sub_llm_mean_batch_size": r.get("sub_llm_mean_batch_size", 0),
            # Main model metrics (available for all modes)
            "turns": r.get("turns", 0),
            "prompt_tokens": r.get("prompt_tokens", 0),
            "completion_tokens": r.get("completion_tokens", 0),
            # REPL timing metrics (RLM modes only)
            "repl_total_time_seconds": r.get("repl_total_time_seconds", 0),
            "repl_call_count": r.get("repl_call_count", 0),
            "repl_mean_time_seconds": r.get("repl_mean_time_seconds", 0),
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
    metric_cols = [
        "reward",
        "exact_match",
        "char_accuracy",
        "levenshtein_similarity",
        # Timing (used by plot_results.py timing plots)
        "generation_ms",
        "scoring_ms",
        "total_ms",
        # Sub-LLM metrics
        "sub_llm_call_count",
        "sub_llm_prompt_tokens",
        "sub_llm_completion_tokens",
        "sub_llm_total_tool_calls",
        "sub_llm_total_turns",
        "sub_llm_batch_count",
        "sub_llm_max_batch_size",
        "sub_llm_mean_batch_size",
        # Main model metrics (available for all modes)
        "turns",
        "prompt_tokens",
        "completion_tokens",
        # REPL timing metrics (RLM modes only)
        "repl_total_time_seconds",
        "repl_call_count",
        "repl_mean_time_seconds",
    ]

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
        default=Path(__file__).parent / "outputs" / "aggregate.csv",
        help="Output CSV file path",
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    main()

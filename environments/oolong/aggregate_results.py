#!/usr/bin/env python3
"""
Aggregate ablation results from oolong long-context experiments.

Reads all results.jsonl files from outputs/evals/ and creates a summary CSV
with metrics grouped by model, mode, and subset.

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
        # Read metadata.json for model and args info
        metadata_file = results_file.parent / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        model = metadata.get("model", "unknown")

        # Extract subset and mode flags from environment args in metadata
        env_args = metadata.get("env_args", {})
        subset = env_args.get("subset", "synth")  # Default to synth
        use_rlm = env_args.get("use_rlm", False)
        include_env_tips = env_args.get("include_env_tips", False)

        with open(results_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    result["_model"] = model
                    result["_subset"] = subset
                    result["_use_rlm"] = use_rlm
                    result["_include_env_tips"] = include_env_tips
                    all_results.append(result)

    print(f"Loaded {len(all_results)} total rollouts")
    return all_results


def get_mode(result: dict) -> str:
    """Get mode from metadata flags."""
    use_rlm = result.get("_use_rlm", False)
    include_env_tips = result.get("_include_env_tips", False)

    if use_rlm:
        return "rlm_tips" if include_env_tips else "rlm"
    return "standard"


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results list to a flat DataFrame."""
    rows = []

    for r in results:
        # Get mode from metadata flags
        mode = get_mode(r)
        info = r.get("info", {})

        row = {
            # Model (from metadata)
            "model": r.get("_model", "unknown"),
            # Mode (inferred)
            "mode": mode,
            # Subset (from metadata)
            "subset": r.get("_subset", "synth"),
            # Main metrics
            "judge_reward": r.get("judge_reward", r.get("reward")),
            "exact_match": r.get("exact_match_reward"),
            "contains_answer": r.get("contains_answer_reward"),
            # Context info
            "context_length": info.get("context_length"),
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
    """Compute summary statistics grouped by model, mode, and subset."""
    group_cols = ["model", "mode", "subset"]

    # All metric columns to aggregate
    metric_cols = [
        "judge_reward",
        "exact_match",
        "contains_answer",
        "context_length",
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

    # Group and compute stats
    summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(
        ["mean", "std", "count"]
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

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
    print("\n" + "=" * 110)
    print("OOLONG ABLATION RESULTS SUMMARY")
    print("=" * 110)

    # Sort by model, mode, and subset for readability
    mode_order = {"standard": 0, "rlm": 1, "rlm_tips": 2}
    subset_order = {"synth": 0, "synth_with_labels": 1, "real": 2}
    summary = summary.copy()
    summary["_mode_order"] = summary["mode"].map(mode_order)
    summary["_subset_order"] = summary["subset"].map(subset_order)
    summary = summary.sort_values(["model", "_mode_order", "_subset_order"]).drop(
        columns=["_mode_order", "_subset_order"]
    )

    print(
        f"\n{'Model':<20} {'Mode':<12} {'Subset':<20} {'Judge Reward':>14} {'Exact Match':>12} {'Samples':>8}"
    )
    print("-" * 110)

    for _, row in summary.iterrows():
        model = normalize_model_name(row.get("model", "unknown"))
        mode = row.get("mode", "standard")
        subset = row.get("subset", "synth")

        judge_mean = row.get("judge_reward_mean", 0)
        judge_std = row.get("judge_reward_std", 0)
        exact_mean = row.get("exact_match_mean", 0)
        exact_std = row.get("exact_match_std", 0)
        count = int(row.get("judge_reward_count", 0))

        judge_str = f"{judge_mean:.3f}±{judge_std:.3f}"
        exact_str = (
            f"{exact_mean:.3f}±{exact_std:.3f}" if pd.notna(exact_mean) else "N/A"
        )

        print(
            f"{model:<20} {mode:<12} {subset:<20} {judge_str:>14} {exact_str:>12} {count:>8}"
        )

    print("-" * 110)
    print(f"Total configurations: {len(summary)}")

    # Print token usage metrics for all modes
    print("\n" + "=" * 110)
    print("TOKEN USAGE METRICS (mean values)")
    print("=" * 110)
    print(
        f"\n{'Model':<20} {'Mode':<12} {'Subset':<15} {'Turns':>10} {'Prompt Tok':>14} {'Compl Tok':>12}"
    )
    print("-" * 110)

    for _, row in summary.iterrows():
        model = normalize_model_name(row.get("model", "unknown"))
        mode = row.get("mode", "standard")
        subset = row.get("subset", "synth")
        num_turns = row.get("turns_mean", 0)
        prompt_tok = row.get("prompt_tokens_mean", 0)
        compl_tok = row.get("completion_tokens_mean", 0)

        print(
            f"{model:<20} {mode:<12} {subset:<15} {num_turns:>10.1f} {prompt_tok:>14.0f} {compl_tok:>12.0f}"
        )
    print("-" * 110)

    # Print RLM-specific metrics if present
    rlm_rows = summary[summary["mode"].isin(["rlm", "rlm_tips"])]
    if len(rlm_rows) > 0:
        print("\n" + "=" * 110)
        print("RLM-SPECIFIC METRICS (mean values)")
        print("=" * 110)
        print(
            f"\n{'Model':<20} {'Mode':<12} {'Subset':<15} {'Sub-LLM Calls':>14} {'Batch Size':>12}"
        )
        print("-" * 110)

        for _, row in rlm_rows.iterrows():
            model = normalize_model_name(row.get("model", "unknown"))
            mode = row.get("mode", "rlm")
            subset = row.get("subset", "synth")
            sub_llm_calls = row.get("sub_llm_call_count_mean", 0)
            batch_size = row.get("sub_llm_mean_batch_size_mean", 0)

            print(
                f"{model:<20} {mode:<12} {subset:<15} {sub_llm_calls:>14.1f} {batch_size:>12.1f}"
            )
        print("-" * 110)


def main():
    parser = argparse.ArgumentParser(description="Aggregate oolong ablation results")
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
        default=Path(__file__).parent / "outputs" / "raw_results.csv",
        help="Output CSV with all individual results",
    )
    args = parser.parse_args()

    # Load results
    results = load_all_results(args.outputs_dir)
    if not results:
        return

    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Save raw results
    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
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

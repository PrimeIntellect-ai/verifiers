#!/usr/bin/env python3
"""
Aggregate ablation results from needle-in-haystack experiments.

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
        # Read metadata.json for model and args info
        metadata_file = results_file.parent / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        model = metadata.get("model", "unknown")

        # Extract parameters from environment args in metadata
        env_args = metadata.get("env_args", {})
        num_lines = env_args.get("num_lines", 10000)
        num_needles = env_args.get("num_needles", 1)
        needle_type = env_args.get("needle_type", "word")
        use_rlm = env_args.get("use_rlm", False)
        include_env_tips = env_args.get("include_env_tips", False)

        with open(results_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    result["_model"] = model
                    result["_num_lines"] = num_lines
                    result["_num_needles"] = num_needles
                    result["_needle_type"] = needle_type
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
            # Ablation parameters (from metadata)
            "needle_type": r.get("_needle_type", info.get("needle_type", "word")),
            "num_lines": r.get("_num_lines", 10000),
            "num_needles": r.get("_num_needles", info.get("num_needles", 1)),
            # Main metrics
            "partial_match": r.get("partial_match_reward", r.get("reward")),
            "exact_match": r.get("exact_match_reward"),
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
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by ablation parameters."""
    group_cols = ["model", "mode", "needle_type", "num_lines", "num_needles"]

    # All metric columns to aggregate
    metric_cols = [
        "partial_match",
        "exact_match",
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
    print("\n" + "=" * 120)
    print("NEEDLE IN HAYSTACK ABLATION RESULTS SUMMARY")
    print("=" * 120)

    # Sort by ablation parameters for readability
    mode_order = {"standard": 0, "rlm": 1, "rlm_tips": 2}
    summary = summary.copy()
    summary["_mode_order"] = summary["mode"].map(mode_order)
    summary = summary.sort_values(
        ["model", "_mode_order", "needle_type", "num_lines", "num_needles"]
    ).drop(columns=["_mode_order"])

    print(
        f"\n{'Model':<18} {'Mode':<10} {'Config':<30} {'Partial Match':>14} {'Exact Match':>12} {'Samples':>8}"
    )
    print("-" * 120)

    for _, row in summary.iterrows():
        model = normalize_model_name(row.get("model", "unknown"))
        mode = row.get("mode", "standard")
        needle_type = row.get("needle_type", "word")
        num_lines = int(row.get("num_lines", 10000))
        num_needles = int(row.get("num_needles", 1))

        config = f"type={needle_type}, lines={num_lines}, n={num_needles}"

        partial_mean = row.get("partial_match_mean", 0)
        partial_std = row.get("partial_match_std", 0)
        exact_mean = row.get("exact_match_mean", 0)
        exact_std = row.get("exact_match_std", 0)
        count = int(row.get("partial_match_count", 0))

        partial_str = f"{partial_mean:.3f}±{partial_std:.3f}"
        exact_str = (
            f"{exact_mean:.3f}±{exact_std:.3f}" if pd.notna(exact_mean) else "N/A"
        )

        print(
            f"{model:<18} {mode:<10} {config:<30} {partial_str:>14} {exact_str:>12} {count:>8}"
        )

    print("-" * 120)
    print(f"Total configurations: {len(summary)}")

    # Print token usage metrics for all modes
    print("\n" + "=" * 120)
    print("TOKEN USAGE METRICS (mean values)")
    print("=" * 120)
    print(
        f"\n{'Model':<18} {'Mode':<10} {'Config':<25} {'Turns':>10} {'Prompt Tok':>14} {'Compl Tok':>12}"
    )
    print("-" * 120)

    for _, row in summary.iterrows():
        model = normalize_model_name(row.get("model", "unknown"))
        mode = row.get("mode", "standard")
        needle_type = row.get("needle_type", "word")
        num_lines = int(row.get("num_lines", 10000))
        num_needles = int(row.get("num_needles", 1))

        config = f"type={needle_type}, l={num_lines}, n={num_needles}"

        num_turns = row.get("turns_mean", 0)
        prompt_tok = row.get("prompt_tokens_mean", 0)
        compl_tok = row.get("completion_tokens_mean", 0)

        print(
            f"{model:<18} {mode:<10} {config:<25} {num_turns:>10.1f} {prompt_tok:>14.0f} {compl_tok:>12.0f}"
        )
    print("-" * 120)

    # Print RLM-specific metrics if present
    rlm_rows = summary[summary["mode"].isin(["rlm", "rlm_tips"])]
    if len(rlm_rows) > 0:
        print("\n" + "=" * 120)
        print("RLM-SPECIFIC METRICS (mean values)")
        print("=" * 120)
        print(
            f"\n{'Model':<18} {'Mode':<10} {'Config':<25} {'Sub-LLM Calls':>14} {'Batch Size':>12}"
        )
        print("-" * 120)

        for _, row in rlm_rows.iterrows():
            model = normalize_model_name(row.get("model", "unknown"))
            mode = row.get("mode", "rlm")
            needle_type = row.get("needle_type", "word")
            num_lines = int(row.get("num_lines", 10000))
            num_needles = int(row.get("num_needles", 1))

            config = f"type={needle_type}, l={num_lines}, n={num_needles}"

            sub_llm_calls = row.get("sub_llm_call_count_mean", 0)
            batch_size = row.get("sub_llm_mean_batch_size_mean", 0)

            print(
                f"{model:<18} {mode:<10} {config:<25} {sub_llm_calls:>14.1f} {batch_size:>12.1f}"
            )
        print("-" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate needle-in-haystack ablation results"
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

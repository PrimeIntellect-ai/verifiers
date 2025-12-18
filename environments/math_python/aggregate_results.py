#!/usr/bin/env python3
"""
Aggregate ablation results from math-python experiments.

Reads all results.jsonl files from outputs/evals/ and creates a summary CSV
with metrics grouped by model and mode.

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


def infer_mode(result: dict) -> str:
    """Infer the mode from result data.

    The mode is determined by checking RLM-specific metrics:
    - If main_rlm_turns > 0, it's an RLM mode
    - If <env_tips> is in the prompt, it's rlm_tips
    """
    # Check if RLM mode was used (RLM metrics will be present and non-zero)
    main_rlm_turns = result.get("main_rlm_turns", 0)
    sub_llm_call_count = result.get("sub_llm_call_count", 0)

    # If we have RLM metrics, it's an RLM mode
    if main_rlm_turns > 0 or sub_llm_call_count > 0:
        # Check if env_tips were included by looking at the prompt
        prompt = result.get("prompt", [])
        if prompt and isinstance(prompt, list) and len(prompt) > 0:
            user_content = (
                prompt[0].get("content", "") if isinstance(prompt[0], dict) else ""
            )
            if "<env_tips>" in user_content:
                return "rlm_tips"
        return "rlm"

    return "standard"


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results list to a flat DataFrame."""
    rows = []

    for r in results:
        # Infer mode from the result
        mode = infer_mode(r)

        row = {
            # Model (from metadata)
            "model": r.get("_model", "unknown"),
            # Mode (inferred)
            "mode": mode,
            # Main metric
            "correct_answer": r.get(
                "correct_answer_rlm", r.get("correct_answer", r.get("reward"))
            ),
            # Timing
            "generation_ms": r.get("generation_ms"),
            "scoring_ms": r.get("scoring_ms"),
            "total_ms": r.get("total_ms"),
            # Tool metrics (standard mode)
            "num_turns": r.get("num_turns"),
            "num_tool_calls": r.get("num_tool_calls"),
            "num_errors": r.get("num_errors"),
            # Sub-LLM metrics (RLM mode only, will be 0/NaN for standard)
            "sub_llm_call_count": r.get("sub_llm_call_count"),
            "sub_llm_prompt_tokens": r.get("sub_llm_prompt_tokens"),
            "sub_llm_completion_tokens": r.get("sub_llm_completion_tokens"),
            "sub_llm_total_tool_calls": r.get("sub_llm_total_tool_calls"),
            "sub_llm_total_turns": r.get("sub_llm_total_turns"),
            "sub_llm_batch_count": r.get("sub_llm_batch_count"),
            "sub_llm_max_batch_size": r.get("sub_llm_max_batch_size"),
            "sub_llm_mean_batch_size": r.get("sub_llm_mean_batch_size"),
            # Main RLM metrics (RLM mode only)
            "main_rlm_turns": r.get("main_rlm_turns"),
            "main_rlm_prompt_tokens": r.get("main_rlm_prompt_tokens"),
            "main_rlm_completion_tokens": r.get("main_rlm_completion_tokens"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by model and mode."""
    group_cols = ["model", "mode"]

    # All metric columns to aggregate
    metric_cols = [
        "correct_answer",
        "generation_ms",
        "scoring_ms",
        "total_ms",
        # Tool metrics (standard mode)
        "num_turns",
        "num_tool_calls",
        "num_errors",
        # Sub-LLM metrics
        "sub_llm_call_count",
        "sub_llm_prompt_tokens",
        "sub_llm_completion_tokens",
        "sub_llm_total_tool_calls",
        "sub_llm_total_turns",
        "sub_llm_batch_count",
        "sub_llm_max_batch_size",
        "sub_llm_mean_batch_size",
        # Main RLM metrics
        "main_rlm_turns",
        "main_rlm_prompt_tokens",
        "main_rlm_completion_tokens",
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
    print("\n" + "=" * 100)
    print("MATH PYTHON ABLATION RESULTS SUMMARY")
    print("=" * 100)

    # Sort by model and mode for readability
    mode_order = {"standard": 0, "rlm": 1, "rlm_tips": 2}
    summary = summary.copy()
    summary["_mode_order"] = summary["mode"].map(mode_order)
    summary = summary.sort_values(["model", "_mode_order"]).drop(
        columns=["_mode_order"]
    )

    print(
        f"\n{'Model':<25} {'Mode':<12} {'Accuracy':>15} {'Samples':>10} {'Time (ms)':>12}"
    )
    print("-" * 100)

    for _, row in summary.iterrows():
        model = normalize_model_name(row.get("model", "unknown"))
        mode = row.get("mode", "standard")

        acc_mean = row.get("correct_answer_mean", 0)
        acc_std = row.get("correct_answer_std", 0)
        count = int(row.get("correct_answer_count", 0))
        time_mean = row.get("total_ms_mean", 0)

        acc_str = f"{acc_mean:.3f}±{acc_std:.3f}"
        time_str = f"{time_mean:.0f}" if pd.notna(time_mean) else "N/A"

        print(f"{model:<25} {mode:<12} {acc_str:>15} {count:>10} {time_str:>12}")

    print("-" * 100)
    print(f"Total configurations: {len(summary)}")

    # Print mode-specific metrics
    # Standard mode metrics
    std_rows = summary[summary["mode"] == "standard"]
    if len(std_rows) > 0:
        print("\n" + "=" * 100)
        print("STANDARD MODE METRICS (mean values)")
        print("=" * 100)
        print(f"\n{'Model':<25} {'Turns':>10} {'Tool Calls':>12} {'Errors':>10}")
        print("-" * 100)

        for _, row in std_rows.iterrows():
            model = normalize_model_name(row.get("model", "unknown"))
            turns = row.get("num_turns_mean", 0)
            tool_calls = row.get("num_tool_calls_mean", 0)
            errors = row.get("num_errors_mean", 0)

            print(f"{model:<25} {turns:>10.1f} {tool_calls:>12.1f} {errors:>10.1f}")
        print("-" * 100)

    # RLM mode metrics
    rlm_rows = summary[summary["mode"].isin(["rlm", "rlm_tips"])]
    if len(rlm_rows) > 0:
        print("\n" + "=" * 100)
        print("RLM-SPECIFIC METRICS (mean values)")
        print("=" * 100)
        print(
            f"\n{'Model':<25} {'Mode':<12} {'RLM Turns':>10} {'Sub-LLM Calls':>14} {'Batch Size':>12}"
        )
        print("-" * 100)

        for _, row in rlm_rows.iterrows():
            model = normalize_model_name(row.get("model", "unknown"))
            mode = row.get("mode", "rlm")
            rlm_turns = row.get("main_rlm_turns_mean", 0)
            sub_llm_calls = row.get("sub_llm_call_count_mean", 0)
            batch_size = row.get("sub_llm_mean_batch_size_mean", 0)

            print(
                f"{model:<25} {mode:<12} {rlm_turns:>10.1f} {sub_llm_calls:>14.1f} {batch_size:>12.1f}"
            )
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate math-python ablation results"
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

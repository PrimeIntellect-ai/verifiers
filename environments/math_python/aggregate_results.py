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

        # Extract mode flags from env_args
        env_args = metadata.get("env_args", {})
        use_rlm = env_args.get("use_rlm", False)
        include_env_tips = env_args.get("include_env_tips", False)
        # New fields for sub-LLM tip ablations
        env_tip_type = env_args.get(
            "env_tip_type", "math"
        )  # default "math" for backward compat
        code_execution_timeout = env_args.get("code_execution_timeout_seconds", None)

        with open(results_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    result["_model"] = model
                    result["_use_rlm"] = use_rlm
                    result["_include_env_tips"] = include_env_tips
                    result["_env_tip_type"] = env_tip_type
                    result["_code_execution_timeout"] = code_execution_timeout
                    all_results.append(result)

    print(f"Loaded {len(all_results)} total rollouts")
    return all_results


def get_mode(result: dict) -> str:
    """Get mode from metadata flags.

    Returns:
        - "standard": Standard PythonEnv mode
        - "rlm": RLMEnv without tips
        - "rlm_tips": RLMEnv with math tips (env_tip_type="math" or backward compat)
        - "rlm_tips_subllm": RLMEnv with sub-LLM tips (no timeout info)
        - "rlm_tips_subllm_{N}s": RLMEnv with sub-LLM tips and specific timeout
    """
    use_rlm = result.get("_use_rlm", False)
    include_env_tips = result.get("_include_env_tips", False)
    env_tip_type = result.get("_env_tip_type", "math")
    timeout = result.get("_code_execution_timeout")

    if not use_rlm:
        return "standard"
    if not include_env_tips:
        return "rlm"
    # Tips are enabled - check which type
    if env_tip_type == "math":
        return "rlm_tips"
    # sub-LLM tips: include timeout if available
    if timeout is not None:
        return f"rlm_tips_subllm_{timeout}s"
    return "rlm_tips_subllm"


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results list to a flat DataFrame."""
    rows = []

    for r in results:
        # Get mode from metadata flags
        mode = get_mode(r)

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
            "total_tool_calls": r.get("total_tool_calls", 0),
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
        # Main model metrics (available for all modes)
        "turns",
        "prompt_tokens",
        "completion_tokens",
        "total_tool_calls",
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
    def get_mode_order(mode: str) -> int:
        """Get sort order for mode names."""
        if mode == "standard":
            return 0
        elif mode == "rlm":
            return 1
        elif mode == "rlm_tips":
            return 2
        elif mode == "rlm_tips_subllm":
            return 3
        elif mode.startswith("rlm_tips_subllm_"):
            # Extract timeout and sort by it
            try:
                timeout = int(mode.replace("rlm_tips_subllm_", "").replace("s", ""))
                return 4 + timeout  # Higher timeouts come later
            except ValueError:
                return 100
        return 100  # Unknown modes last

    summary = summary.copy()
    summary["_mode_order"] = summary["mode"].apply(get_mode_order)
    summary = summary.sort_values(["model", "_mode_order"]).drop(
        columns=["_mode_order"]
    )

    print(
        f"\n{'Model':<25} {'Mode':<22} {'Accuracy':>15} {'Samples':>10} {'Time (ms)':>12}"
    )
    print("-" * 110)

    for _, row in summary.iterrows():
        model = normalize_model_name(row.get("model", "unknown"))
        mode = row.get("mode", "standard")

        acc_mean = row.get("correct_answer_mean", 0)
        count = int(row.get("correct_answer_count", 0))
        time_mean = row.get("total_ms_mean", 0)

        acc_str = f"{acc_mean:.3f}"
        time_str = f"{time_mean:.0f}" if pd.notna(time_mean) else "N/A"

        print(f"{model:<25} {mode:<22} {acc_str:>15} {count:>10} {time_str:>12}")

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

    # Print token usage metrics for all modes
    print("\n" + "=" * 100)
    print("TOKEN USAGE METRICS (mean values)")
    print("=" * 100)
    print(
        f"\n{'Model':<25} {'Mode':<22} {'Turns':>10} {'Prompt Tok':>14} {'Compl Tok':>12} {'Tool Calls':>12}"
    )
    print("-" * 110)

    for _, row in summary.iterrows():
        model = normalize_model_name(row.get("model", "unknown"))
        mode = row.get("mode", "standard")
        num_turns = row.get("turns_mean", 0)
        prompt_tok = row.get("prompt_tokens_mean", 0)
        compl_tok = row.get("completion_tokens_mean", 0)
        tool_calls = row.get("total_tool_calls_mean", 0)

        print(
            f"{model:<25} {mode:<22} {num_turns:>10.1f} {prompt_tok:>14.0f} {compl_tok:>12.0f} {tool_calls:>12.1f}"
        )
    print("-" * 110)

    # RLM mode metrics (include all RLM variants)
    rlm_rows = summary[summary["mode"].str.startswith("rlm")]
    if len(rlm_rows) > 0:
        print("\n" + "=" * 100)
        print("RLM-SPECIFIC METRICS (mean values)")
        print("=" * 100)
        print(f"\n{'Model':<25} {'Mode':<22} {'Sub-LLM Calls':>14} {'Batch Size':>12}")
        print("-" * 110)

        for _, row in rlm_rows.iterrows():
            model = normalize_model_name(row.get("model", "unknown"))
            mode = row.get("mode", "rlm")
            sub_llm_calls = row.get("sub_llm_call_count_mean", 0)
            batch_size = row.get("sub_llm_mean_batch_size_mean", 0)

            print(f"{model:<25} {mode:<22} {sub_llm_calls:>14.1f} {batch_size:>12.1f}")
        print("-" * 110)


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

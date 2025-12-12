#!/usr/bin/env python3
"""
Analyze metrics JSON files produced by verifiers environments.

Supports metrics from DeepDive, Oolong, Needle-in-Haystack, and other
environments that use the standard metrics logging format.

Usage:
    python analyze_metrics.py results/metrics.json
    python analyze_metrics.py results/standard.json results/rlm.json  # Compare two files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: str | Path) -> dict:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)


def print_stats(name: str, values: list, unit: str = "") -> None:
    """Print statistics for a list of values."""
    if not values or all(v is None for v in values):
        print(f"  {name}: no data")
        return

    # Filter out None values
    values = [v for v in values if v is not None]
    if not values:
        print(f"  {name}: no data")
        return

    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)
    median = (
        sorted_vals[n // 2]
        if n % 2 == 1
        else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    )
    min_val = min(values)
    max_val = max(values)

    # Standard deviation
    variance = sum((x - mean) ** 2 for x in values) / n if n > 0 else 0
    std = variance**0.5

    unit_str = f" {unit}" if unit else ""
    print(f"  {name}:")
    print(f"    mean: {mean:,.2f}{unit_str}  (±{std:,.2f})")
    print(f"    median: {median:,.2f}{unit_str}")
    print(f"    range: [{min_val:,.2f}, {max_val:,.2f}]{unit_str}")


def analyze_by_context_length(metrics: dict, bucket_size: int = 100000) -> None:
    """Analyze accuracy by context length buckets (for oolong)."""
    context_lengths = metrics.get("context_length", [])
    judge_correct = metrics.get("judge_correct", [])

    if not context_lengths or not judge_correct:
        return

    # Bucket results by context length
    buckets: dict[int, list[float]] = {}
    for length, correct in zip(context_lengths, judge_correct):
        if correct < 0:  # Skip unevaluated samples
            continue
        bucket = (length // bucket_size) * bucket_size
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(correct)

    if not buckets:
        return

    print("\n  Accuracy by context length:")
    for bucket in sorted(buckets.keys()):
        vals = buckets[bucket]
        acc = sum(vals) / len(vals)
        # Format bucket range nicely (e.g., "0-100k chars" or "100-200k chars")
        start_k = bucket // 1000
        end_k = (bucket + bucket_size) // 1000
        label = f"    {start_k:>4}k-{end_k:<4}k chars"
        print(f"{label}: {acc:>5.1%} (n={len(vals)})")


def detect_environment(metrics: dict) -> str:
    """Detect which environment produced the metrics based on fields present."""
    if "num_lines" in metrics or "needle_type" in metrics:
        return "needle_in_haystack"
    elif "subset" in metrics:
        return "oolong"
    else:
        return "deepdive"


def print_environment_info(metrics: dict, env_type: str) -> None:
    """Print environment-specific information."""
    if env_type == "needle_in_haystack":
        # Needle-in-haystack specific fields
        if metrics.get("num_lines"):
            num_lines = metrics["num_lines"]
            unique_lines = set(num_lines)
            if len(unique_lines) == 1:
                print(f"  Haystack size: {num_lines[0]:,} lines")
            else:
                print(
                    f"  Haystack sizes: {', '.join(str(x) for x in sorted(unique_lines))}"
                )

        if metrics.get("num_needles"):
            num_needles = metrics["num_needles"]
            unique_needles = set(num_needles)
            if len(unique_needles) == 1:
                print(f"  Needles per haystack: {num_needles[0]}")
            else:
                print(
                    f"  Needles per haystack: {', '.join(str(x) for x in sorted(unique_needles))}"
                )

        if metrics.get("needle_type"):
            types = set(metrics["needle_type"])
            print(f"  Needle type(s): {', '.join(types)}")

        if metrics.get("needle_position"):
            positions = [p for p in metrics["needle_position"] if p is not None]
            if positions:
                unique_pos = set(positions)
                if len(unique_pos) == 1:
                    print(f"  Needle position: {positions[0]:.2f}")
                else:
                    print(
                        f"  Needle positions: {min(positions):.2f} - {max(positions):.2f}"
                    )

        # Show partial match stats for needle-in-haystack
        if metrics.get("partial_match"):
            partial = [p for p in metrics["partial_match"] if p >= 0]
            if partial:
                mean_partial = sum(partial) / len(partial)
                print(f"  Partial match: {mean_partial:.1%}")

        if metrics.get("needles_found"):
            found = [f for f in metrics["needles_found"] if f >= 0]
            if found:
                mean_found = sum(found) / len(found)
                print(f"  Avg needles found: {mean_found:.2f}")

    elif env_type == "oolong":
        # Oolong specific fields
        if metrics.get("subset"):
            subsets = set(metrics["subset"])
            print(f"  Subset(s): {', '.join(sorted(subsets))}")

        # Show exact match and contains_answer for oolong
        if metrics.get("exact_match"):
            exact = [e for e in metrics["exact_match"] if e >= 0]
            if exact:
                mean_exact = sum(exact) / len(exact)
                print(f"  Exact match: {mean_exact:.1%}")

        if metrics.get("contains_answer"):
            contains = [c for c in metrics["contains_answer"] if c >= 0]
            if contains:
                mean_contains = sum(contains) / len(contains)
                print(f"  Contains answer: {mean_contains:.1%}")

        # Show accuracy by context length if available
        analyze_by_context_length(metrics)


def analyze_single(metrics: dict, name: str = "Metrics") -> None:
    """Analyze a single metrics file."""
    n_samples = len(metrics.get("example_id", []))
    is_rlm = (
        metrics.get("is_rlm_mode", [False])[0] if metrics.get("is_rlm_mode") else False
    )
    env_type = detect_environment(metrics)

    # Calculate judge reward / accuracy first for prominent display
    judge = metrics.get("judge_correct", [])
    valid_judge = (
        [j for j in judge if j >= 0] if judge else []
    )  # -1 means not evaluated
    accuracy = sum(valid_judge) / len(valid_judge) if valid_judge else None

    # For needle-in-haystack, also check exact_match as primary metric
    exact_match = metrics.get("exact_match", [])
    valid_exact = [e for e in exact_match if e >= 0] if exact_match else []
    exact_accuracy = sum(valid_exact) / len(valid_exact) if valid_exact else None

    print(f"\n{'=' * 60}")
    print(f" {name}")
    print(f"{'=' * 60}")

    # Primary metric at the top
    if accuracy is not None:
        print(
            f"\n  ★ JUDGE REWARD: {accuracy:.1%} ({sum(valid_judge)}/{len(valid_judge)} correct)"
        )
    elif exact_accuracy is not None:
        print(
            f"\n  ★ EXACT MATCH: {exact_accuracy:.1%} ({sum(valid_exact)}/{len(valid_exact)} correct)"
        )
    else:
        print("\n  ★ ACCURACY: no data")

    print(f"\n  Samples: {n_samples}")
    print(f"  Mode: {'RLM' if is_rlm else 'Standard'}")
    print(f"  Environment: {env_type}")

    # Error rate
    errors = metrics.get("had_error", [])
    if errors:
        error_count = sum(1 for e in errors if e)
        print(f"  Errors: {error_count} ({100 * error_count / len(errors):.1f}%)")

    # Environment-specific info
    print_environment_info(metrics, env_type)

    print("\n--- Token Usage ---")
    print_stats("Total tokens", metrics.get("total_tokens", []))
    print_stats("Main prompt tokens", metrics.get("main_prompt_tokens", []))
    print_stats("Main completion tokens", metrics.get("main_completion_tokens", []))

    if is_rlm:
        print_stats("Sub-LLM prompt tokens", metrics.get("sub_llm_prompt_tokens", []))
        print_stats(
            "Sub-LLM completion tokens", metrics.get("sub_llm_completion_tokens", [])
        )

    print("\n--- Tool Calls ---")
    print_stats("Total tool calls", metrics.get("total_tool_calls", []))
    print_stats("Main tool calls", metrics.get("main_tool_calls", []))

    if is_rlm:
        print_stats("Sub-LLM tool calls", metrics.get("sub_llm_total_tool_calls", []))

    print("\n--- Turns ---")
    print_stats("Main turns", metrics.get("main_turns", []))

    if is_rlm:
        print_stats("Sub-LLM calls", metrics.get("sub_llm_calls", []))
        print_stats("Sub-LLM total turns", metrics.get("sub_llm_total_turns", []))

        # Calculate average turns per sub-LLM call
        sub_calls = metrics.get("sub_llm_calls", [])
        sub_turns = metrics.get("sub_llm_total_turns", [])
        if sub_calls and sub_turns:
            avg_turns_per_call = [
                t / c if c > 0 else 0 for t, c in zip(sub_turns, sub_calls)
            ]
            print_stats("Avg turns per sub-LLM", avg_turns_per_call)

    print("\n--- Timing ---")
    print_stats("Generation time", metrics.get("generation_ms", []), "ms")
    print_stats("Scoring time", metrics.get("scoring_ms", []), "ms")
    print_stats("Total time", metrics.get("total_ms", []), "ms")


def compare_two(metrics1: dict, metrics2: dict, name1: str, name2: str) -> None:
    """Compare two metrics files side by side."""
    analyze_single(metrics1, name1)
    analyze_single(metrics2, name2)

    print(f"\n{'=' * 60}")
    print(f" COMPARISON: {name1} vs {name2}")
    print(f"{'=' * 60}")

    # Judge reward comparison at the top
    judge1 = [j for j in metrics1.get("judge_correct", []) if j >= 0]
    judge2 = [j for j in metrics2.get("judge_correct", []) if j >= 0]

    # Fall back to exact_match if no judge data
    if not judge1:
        judge1 = [e for e in metrics1.get("exact_match", []) if e >= 0]
    if not judge2:
        judge2 = [e for e in metrics2.get("exact_match", []) if e >= 0]

    if judge1 and judge2:
        acc1 = sum(judge1) / len(judge1)
        acc2 = sum(judge2) / len(judge2)
        diff = acc2 - acc1
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(
            f"\n  ★ ACCURACY: {acc1:.1%} → {acc2:.1%} ({arrow} {abs(diff) * 100:.1f}pp)"
        )

    def compare_metric(metric_name: str, display_name: str) -> None:
        vals1 = [v for v in metrics1.get(metric_name, []) if v is not None]
        vals2 = [v for v in metrics2.get(metric_name, []) if v is not None]

        if not vals1 or not vals2:
            return

        mean1 = sum(vals1) / len(vals1)
        mean2 = sum(vals2) / len(vals2)
        diff = mean2 - mean1
        pct = (diff / mean1 * 100) if mean1 != 0 else float("inf")

        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(
            f"  {display_name}: {mean1:,.1f} → {mean2:,.1f} ({arrow} {abs(pct):.1f}%)"
        )

    print("\n--- Tokens ---")
    compare_metric("total_tokens", "Total tokens")
    compare_metric("total_prompt_tokens", "Prompt tokens")
    compare_metric("total_completion_tokens", "Completion tokens")

    print("\n--- Efficiency ---")
    compare_metric("total_tool_calls", "Tool calls")
    compare_metric("main_turns", "Main turns")
    compare_metric("total_ms", "Time (ms)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze metrics JSON files from verifiers environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_metrics.py results/metrics.json
    python analyze_metrics.py standard.json rlm.json

Supported environments:
    - DeepDive (web search)
    - Oolong (long-context QA)
    - Needle-in-Haystack (information retrieval)
        """,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="JSON metrics file(s) to analyze (1 or 2 files)",
    )
    args = parser.parse_args()

    if len(args.files) == 1:
        metrics = load_metrics(args.files[0])
        analyze_single(metrics, Path(args.files[0]).stem)
    elif len(args.files) == 2:
        metrics1 = load_metrics(args.files[0])
        metrics2 = load_metrics(args.files[1])
        compare_two(
            metrics1,
            metrics2,
            Path(args.files[0]).stem,
            Path(args.files[1]).stem,
        )
    else:
        print("Error: Please provide 1 or 2 JSON files", file=sys.stderr)
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()

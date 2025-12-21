#!/usr/bin/env python3
"""
Plot ablation results for deepdive experiments.

Creates focused plots comparing modes across models:
- Mode comparison by model (bar chart)
- RLM metrics comparison (sub-LLM usage, turns, etc.)
- Timing comparison
- Optional: timing_vs_reward (cost-benefit analysis)

Usage:
    python plot_results.py [--input aggregate.csv] [--output plots.png]
    python plot_results.py --image reward
    python plot_results.py --image rlm_metrics
    python plot_results.py --image timing_vs_reward
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


# Mode styling for consistent visualization
MODE_STYLES = {
    "standard": {"color": "#E24A33", "marker": "o", "linestyle": "-"},
    "rlm": {"color": "#348ABD", "marker": "s", "linestyle": "--"},
    "rlm_tips": {"color": "#988ED5", "marker": "^", "linestyle": ":"},
}

MODE_ORDER = ["standard", "rlm", "rlm_tips"]
MODE_LABELS = {
    "standard": "Standard\n(direct tools)",
    "rlm": "RLM\n(no tips)",
    "rlm_tips": "RLM\n(with tips)",
}


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the aggregate results CSV."""
    df = pd.read_csv(csv_path)

    # Ensure mode column exists
    if "mode" not in df.columns:
        df["mode"] = "standard"

    return df


def normalize_model_name(model: str) -> str:
    """Create display-friendly model name."""
    if model.startswith("openrouter/"):
        parts = model.split("/")
        return parts[-1] if len(parts) >= 3 else model
    if "/" in model:
        return model.split("/")[-1]
    return model


def plot_reward_by_model(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Mode comparison across models (grouped bar chart)."""
    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        rewards = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                rewards.append(model_data["judge_reward_mean"].values[0])
                # Get sample count if available
                if "judge_reward_count" in model_data.columns:
                    counts.append(int(model_data["judge_reward_count"].values[0]))
                else:
                    counts.append(None)
            else:
                rewards.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            [xi + offset for xi in x],
            rewards,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add sample size labels above bars
        for bar, count in zip(bars, counts):
            if count is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                )

    ax.set_xlabel("Model")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Mode Comparison by Model")
    ax.set_ylim(0, 1.2)  # Increased to make room for labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_rlm_metrics(ax: plt.Axes, df: pd.DataFrame):
    """Plot: RLM-specific metrics comparison (rlm vs rlm_tips)."""
    # Filter to RLM modes only
    rlm_df = df[df["mode"].isin(["rlm", "rlm_tips"])]

    if len(rlm_df) == 0:
        ax.text(
            0.5,
            0.5,
            "No RLM data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    models = list(rlm_df["model"].unique())
    modes = [m for m in ["rlm", "rlm_tips"] if m in rlm_df["mode"].unique()]

    # Metrics to show
    metrics = [
        ("turns_mean", "Turns"),
        ("sub_llm_call_count_mean", "Sub-LLM Calls"),
        ("sub_llm_total_tool_calls_mean", "Tool Calls"),
    ]

    n_models = len(models)
    n_modes = len(modes)
    n_metrics = len(metrics)

    # Hatching patterns for different models (dense patterns for thin/low bars)
    HATCHES = ["///", "...", "+++", "***", "\\\\\\", "xxx", "ooo", "|||", "---"]
    hatches = HATCHES[:n_models]

    # Bar sizing
    bar_width = 0.1  # Thinner bars
    mode_gap = 0.05  # Small gap between mode groups within a metric
    metric_gap = 0.4  # Larger gap between metrics

    # Width of one mode group (all models for one mode)
    mode_group_width = n_models * bar_width
    # Width of one metric group (both modes)
    metric_group_width = n_modes * mode_group_width + mode_gap

    for metric_idx, (metric_col, metric_label) in enumerate(metrics):
        # Center position for this metric's group
        metric_center = metric_idx * (metric_group_width + metric_gap)

        for mode_idx, mode in enumerate(modes):
            style = MODE_STYLES.get(mode, {"color": "gray"})

            for model_idx, model in enumerate(models):
                model_data = rlm_df[
                    (rlm_df["model"] == model) & (rlm_df["mode"] == mode)
                ]

                if len(model_data) == 0:
                    value = 0
                else:
                    value = (
                        model_data[metric_col].values[0]
                        if metric_col in model_data.columns
                        else 0
                    )

                # Calculate bar position within metric group
                # mode_idx determines which mode group (left or right)
                # model_idx determines position within that mode group
                bar_x = (
                    metric_center
                    + mode_idx * (mode_group_width + mode_gap)
                    + model_idx * bar_width
                    - (metric_group_width - bar_width) / 2  # Center the group
                )

                ax.bar(
                    bar_x,
                    value,
                    bar_width,
                    color=style["color"],
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=hatches[model_idx],
                )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Count")
    ax.set_title("RLM Usage Metrics by Mode")

    # Set x-ticks at the center of each metric group
    tick_positions = [i * (metric_group_width + metric_gap) for i in range(n_metrics)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([m[1] for m in metrics])

    # Create custom legend with separate entries for models (pattern) and modes (color)
    legend_handles = []

    # Mode entries (color, no hatch) - use a neutral gray for the base
    for mode in modes:
        style = MODE_STYLES.get(mode, {"color": "gray"})
        mode_label = MODE_LABELS.get(mode, mode).replace("\n", " ")
        legend_handles.append(
            Patch(
                facecolor=style["color"],
                edgecolor="black",
                linewidth=0.5,
                label=mode_label,
            )
        )

    # Model entries (hatch pattern, neutral color)
    for model_idx, model in enumerate(models):
        legend_handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
                hatch=hatches[model_idx],
                label=normalize_model_name(model),
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=max(n_modes, n_models),
        fontsize=8,
    )


def plot_timing(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing comparison across modes and models."""
    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        times = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0 and pd.notna(model_data["total_ms_mean"].values[0]):
                times.append(
                    model_data["total_ms_mean"].values[0] / 1000
                )  # Convert to seconds
            else:
                times.append(0)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        ax.bar(
            [xi + offset for xi in x],
            times,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Average Rollout Time by Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_token_usage(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Token usage comparison (RLM modes only)."""
    # Filter to RLM modes only
    rlm_df = df[df["mode"].isin(["rlm", "rlm_tips"])]

    if len(rlm_df) == 0:
        ax.text(
            0.5,
            0.5,
            "No RLM data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    models = rlm_df["model"].unique()
    modes = [m for m in ["rlm", "rlm_tips"] if m in rlm_df["mode"].unique()]

    x = range(len(models))
    width = 0.35

    for i, mode in enumerate(modes):
        mode_data = rlm_df[rlm_df["mode"] == mode]

        # Total tokens = main model + sub-LLM tokens
        total_tokens = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                main_prompt = model_data["prompt_tokens_mean"].values[0] or 0
                main_completion = model_data["completion_tokens_mean"].values[0] or 0
                sub_prompt = model_data["sub_llm_prompt_tokens_mean"].values[0] or 0
                sub_completion = (
                    model_data["sub_llm_completion_tokens_mean"].values[0] or 0
                )
                total_tokens.append(
                    main_prompt + main_completion + sub_prompt + sub_completion
                )
            else:
                total_tokens.append(0)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        ax.bar(
            [xi + offset for xi in x],
            total_tokens,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Total Tokens")
    ax.set_title("Token Usage by Mode (RLM only)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_vs_reward(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Reward vs timing scatter (cost-benefit analysis)."""
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode]
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})

        # Filter out rows with missing data
        valid_data = data.dropna(subset=["total_ms_mean", "judge_reward_mean"])

        if len(valid_data) > 0:
            # Convert to seconds
            times = valid_data["total_ms_mean"] / 1000

            ax.scatter(
                times,
                valid_data["judge_reward_mean"],
                label=MODE_LABELS.get(mode, mode).replace("\n", " "),
                color=style["color"],
                marker=style["marker"],
                s=100,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Timing vs Reward\n(by mode)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def create_plots(df: pd.DataFrame, output_path: Path | None = None):
    """Create the 2x2 grid of plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Set style
    sns.set_style("whitegrid")

    # Main plots
    plot_reward_by_model(axes[0, 0], df)
    plot_timing(axes[0, 1], df)
    plot_rlm_metrics(axes[1, 0], df)
    plot_token_usage(axes[1, 1], df)

    plt.suptitle(
        "DeepDive Ablation Results: Mode Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


# Mapping of plot names to (function, figsize, title)
PLOT_REGISTRY = {
    "reward": (plot_reward_by_model, (10, 7), "Mode Comparison by Model"),
    "timing": (plot_timing, (10, 7), "Timing Comparison"),
    "rlm_metrics": (plot_rlm_metrics, (12, 7), "RLM Usage Metrics"),
    "tokens": (plot_token_usage, (10, 7), "Token Usage"),
    # Additional timing plots
    "timing_vs_reward": (plot_timing_vs_reward, (10, 7), "Timing vs Reward"),
}


def create_single_plot(
    plot_name: str,
    df: pd.DataFrame,
    output_path: Path | None = None,
):
    """Create a single standalone plot."""
    if plot_name not in PLOT_REGISTRY:
        raise ValueError(
            f"Unknown plot: {plot_name}. Available: {list(PLOT_REGISTRY.keys())}"
        )

    func, figsize, title = PLOT_REGISTRY[plot_name]

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")

    func(ax, df)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot deepdive ablation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show main 2x2 grid
    python plot_results.py
    
    # List available models
    python plot_results.py --list-models
    
    # Filter to a specific model
    python plot_results.py --model gpt-4.1-mini
    
    # Show individual reward plot
    python plot_results.py --image reward
    
    # Show RLM metrics comparison
    python plot_results.py --image rlm_metrics
    
    # Save plot to file
    python plot_results.py --image reward -o reward_plot.png
    
    # Timing vs reward analysis
    python plot_results.py --image timing_vs_reward
""",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(__file__).parent / "outputs" / "aggregate.csv",
        help="Path to aggregate CSV file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image file path (optional, shows interactive plot if not set)",
    )
    parser.add_argument(
        "--image",
        choices=[
            "main",
            "reward",
            "timing",
            "rlm_metrics",
            "tokens",
            "timing_vs_reward",
        ],
        default="main",
        help="Which plot to generate: 'main' for 2x2 grid, or individual plot name",
    )

    # Model filtering options
    parser.add_argument(
        "--model",
        "-M",
        type=str,
        default=None,
        help="Filter results to a specific model (supports substring matching)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models in the data and exit",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run aggregate_results.py first to generate the CSV.")
        return

    df = load_data(args.input)

    # Handle --list-models
    if args.list_models:
        if "model" not in df.columns:
            print("No model column found in data.")
            return

        models = sorted(df["model"].unique())
        print(f"Available models ({len(models)} total):")
        for model in models:
            count = len(df[df["model"] == model])
            print(f"  {model}  ({count} configurations)")
        return

    # Filter by model if specified
    if args.model:
        if "model" not in df.columns:
            print("Warning: No model column found in data, --model filter ignored.")
        else:
            original_count = len(df)
            # Try exact match first, then substring
            if args.model in df["model"].unique():
                df = df[df["model"] == args.model]
            else:
                mask = df["model"].str.contains(args.model, case=False, na=False)
                df = df[mask]

            if len(df) == 0:
                print(f"Error: No results found for model '{args.model}'")
                print("Use --list-models to see available models.")
                return

            matched_models = df["model"].unique()
            print(f"Filtered: {original_count} -> {len(df)} configurations")
            if len(matched_models) > 1:
                print(f"  Matched models: {list(matched_models)}")

    print(f"Loaded {len(df)} configurations from {args.input}")

    if args.image == "main":
        create_plots(df, args.output)
    else:
        create_single_plot(args.image, df, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot ablation results for needle-in-haystack experiments.

Creates focused plots comparing modes across different ablation dimensions:
- Mode comparison by needle type
- Mode comparison vs context size (num_lines)
- Mode comparison vs needle count
- RLM metrics comparison
- Optional timing plots: timing, timing_vs_context, timing_vs_needles, timing_efficiency

Usage:
    python plot_results.py [--input aggregate.csv] [--output plots.png]
    python plot_results.py --image needle_type
    python plot_results.py --image context_size
    python plot_results.py --image timing_vs_context
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    "standard": "Standard\n(direct)",
    "rlm": "RLM\n(no tips)",
    "rlm_tips": "RLM\n(with tips)",
}


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the aggregate results CSV."""
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
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


def plot_mode_by_needle_type(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Mode comparison across needle types (grouped bar chart)."""
    # Filter to baseline config: num_lines=10000, num_needles=1
    filtered = df[(df["num_lines"] == 10000) & (df["num_needles"] == 1)]

    # Aggregate across models
    agg_dict = {"partial_match_mean": "mean"}
    if "partial_match_count" in filtered.columns:
        agg_dict["partial_match_count"] = "sum"

    agg_df = filtered.groupby(["needle_type", "mode"]).agg(agg_dict).reset_index()

    needle_types = ["word", "numeric"]
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(needle_types))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        rewards = []
        counts = []
        for nt in needle_types:
            nt_data = mode_data[mode_data["needle_type"] == nt]
            if len(nt_data) > 0:
                rewards.append(nt_data["partial_match_mean"].values[0])
                if "partial_match_count" in nt_data.columns:
                    counts.append(int(nt_data["partial_match_count"].values[0]))
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

    ax.set_xlabel("Needle Type")
    ax.set_ylabel("Partial Match Reward")
    ax.set_title("Mode Comparison by Needle Type\n(lines=10K, needles=1)")
    ax.set_ylim(0, 1.2)  # Increased to make room for labels
    ax.set_xticks(x)
    ax.set_xticklabels([nt.capitalize() for nt in needle_types])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_mode_vs_context_size(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Mode comparison across context sizes (line plot)."""
    # Filter to baseline config: needle_type="word", num_needles=1
    filtered = df[(df["needle_type"] == "word") & (df["num_needles"] == 1)]

    # Aggregate across models
    agg_df = (
        filtered.groupby(["num_lines", "mode"])
        .agg(
            {
                "partial_match_mean": "mean",
            }
        )
        .reset_index()
    )

    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    for mode in modes:
        data = agg_df[agg_df["mode"] == mode].sort_values("num_lines")
        if len(data) > 0:
            style = MODE_STYLES.get(
                mode, {"color": "gray", "marker": "o", "linestyle": "-"}
            )
            ax.plot(
                data["num_lines"] / 1000,  # Convert to thousands
                data["partial_match_mean"],
                label=MODE_LABELS.get(mode, mode),
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Context Size (K lines)")
    ax.set_ylabel("Partial Match Reward")
    ax.set_title("Mode Comparison vs Context Size\n(type=word, needles=1)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_mode_vs_needle_count(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Mode comparison across needle counts (line plot)."""
    # Filter to baseline config: needle_type="word", num_lines=10000
    filtered = df[(df["needle_type"] == "word") & (df["num_lines"] == 10000)]

    # Aggregate across models
    agg_df = (
        filtered.groupby(["num_needles", "mode"])
        .agg(
            {
                "partial_match_mean": "mean",
                "exact_match_mean": "mean",
            }
        )
        .reset_index()
    )

    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    for mode in modes:
        data = agg_df[agg_df["mode"] == mode].sort_values("num_needles")
        if len(data) > 0:
            style = MODE_STYLES.get(
                mode, {"color": "gray", "marker": "o", "linestyle": "-"}
            )
            ax.plot(
                data["num_needles"],
                data["partial_match_mean"],
                label=f"{MODE_LABELS.get(mode, mode)} (partial)",
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Number of Needles")
    ax.set_ylabel("Partial Match Reward")
    ax.set_title("Mode Comparison vs Needle Count\n(type=word, lines=10K)")
    ax.set_ylim(0, 1.1)
    ax.set_xticks([1, 3, 5])
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

    # Aggregate across all configs
    agg_df = (
        rlm_df.groupby("mode")
        .agg(
            {
                "turns_mean": "mean",
                "sub_llm_call_count_mean": "mean",
                "sub_llm_mean_batch_size_mean": "mean",
            }
        )
        .reset_index()
    )

    modes = [m for m in ["rlm", "rlm_tips"] if m in agg_df["mode"].unique()]

    # Metrics to show
    metrics = [
        ("turns_mean", "Turns"),
        ("sub_llm_call_count_mean", "Sub-LLM Calls"),
        ("sub_llm_mean_batch_size_mean", "Avg Batch Size"),
    ]

    x = range(len(metrics))
    width = 0.35

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        if len(mode_data) == 0:
            continue

        values = [
            mode_data[m[0]].values[0]
            if m[0] in mode_data.columns and pd.notna(mode_data[m[0]].values[0])
            else 0
            for m in metrics
        ]

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        ax.bar(
            [xi + offset for xi in x],
            values,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Count / Value")
    ax.set_title("RLM Usage Metrics by Mode\n(aggregated across all configs)")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_heatmap(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Heatmap of reward by mode × needle_type."""
    # Filter to baseline config: num_lines=10000, num_needles=1
    filtered = df[(df["num_lines"] == 10000) & (df["num_needles"] == 1)]

    # Create pivot table: mode × needle_type
    pivot = filtered.pivot_table(
        values="partial_match_mean",
        index="mode",
        columns="needle_type",
        aggfunc="mean",
    )

    # Reorder rows and columns
    row_order = [m for m in MODE_ORDER if m in pivot.index]
    col_order = ["word", "numeric"]

    pivot = pivot.reindex(
        index=row_order, columns=[c for c in col_order if c in pivot.columns]
    )

    # Rename for display
    pivot.index = [MODE_LABELS.get(m, m).replace("\n", " ") for m in pivot.index]
    pivot.columns = [c.capitalize() for c in pivot.columns]

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Partial Match"},
        linewidths=0.5,
    )

    ax.set_xlabel("Needle Type")
    ax.set_ylabel("Mode")
    ax.set_title("Reward Heatmap: Mode × Needle Type\n(lines=10K, needles=1)")


def plot_partial_vs_exact(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Partial match vs exact match scatter, colored by mode."""
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode]
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})

        # Filter out NaN values
        valid_data = data.dropna(subset=["partial_match_mean", "exact_match_mean"])

        ax.scatter(
            valid_data["exact_match_mean"],
            valid_data["partial_match_mean"],
            label=MODE_LABELS.get(mode, mode).replace("\n", " "),
            color=style["color"],
            marker=style["marker"],
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    # Add diagonal line (partial >= exact always)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")

    ax.set_xlabel("Exact Match (all needles)")
    ax.set_ylabel("Partial Match (fraction)")
    ax.set_title("Partial vs Exact Match\n(all configs, by mode)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_by_mode(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing comparison across modes (bar chart)."""
    # Aggregate across all configs for each mode
    agg_df = (
        df.groupby("mode")
        .agg(
            {
                "total_ms_mean": "mean",
                "total_ms_std": "mean",
            }
        )
        .reset_index()
    )

    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]
    x = range(len(modes))

    times = []
    errors = []
    colors = []
    for mode in modes:
        mode_data = agg_df[agg_df["mode"] == mode]
        if len(mode_data) > 0 and pd.notna(mode_data["total_ms_mean"].values[0]):
            times.append(
                mode_data["total_ms_mean"].values[0] / 1000
            )  # Convert to seconds
            errors.append(
                mode_data["total_ms_std"].values[0] / 1000
                if pd.notna(mode_data["total_ms_std"].values[0])
                else 0
            )
        else:
            times.append(0)
            errors.append(0)
        colors.append(MODE_STYLES.get(mode, {"color": "gray"})["color"])

    ax.bar(
        x, times, yerr=errors, color=colors, edgecolor="black", linewidth=0.5, capsize=3
    )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Average Rollout Time by Mode\n(aggregated across all configs)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m).replace("\n", " ") for m in modes])


def plot_timing_vs_context(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing vs context size (num_lines) by mode (line plot).

    This is a critical plot for understanding how timing scales with context length.
    """
    # Filter to baseline config: needle_type="word", num_needles=1
    filtered = df[(df["needle_type"] == "word") & (df["num_needles"] == 1)]

    # Aggregate across models
    agg_df = (
        filtered.groupby(["num_lines", "mode"])
        .agg(
            {
                "total_ms_mean": "mean",
            }
        )
        .reset_index()
    )

    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    for mode in modes:
        data = agg_df[agg_df["mode"] == mode].sort_values("num_lines")
        if len(data) > 0 and "total_ms_mean" in data.columns:
            style = MODE_STYLES.get(
                mode, {"color": "gray", "marker": "o", "linestyle": "-"}
            )
            # Convert to seconds
            times = data["total_ms_mean"] / 1000
            ax.plot(
                data["num_lines"] / 1000,  # Convert to thousands
                times,
                label=MODE_LABELS.get(mode, mode).replace("\n", " "),
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Context Size (K lines)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing vs Context Size\n(type=word, needles=1)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_vs_needles(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing vs needle count by mode (line plot)."""
    # Filter to baseline config: needle_type="word", num_lines=10000
    filtered = df[(df["needle_type"] == "word") & (df["num_lines"] == 10000)]

    # Aggregate across models
    agg_df = (
        filtered.groupby(["num_needles", "mode"])
        .agg(
            {
                "total_ms_mean": "mean",
            }
        )
        .reset_index()
    )

    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    for mode in modes:
        data = agg_df[agg_df["mode"] == mode].sort_values("num_needles")
        if len(data) > 0 and "total_ms_mean" in data.columns:
            style = MODE_STYLES.get(
                mode, {"color": "gray", "marker": "o", "linestyle": "-"}
            )
            # Convert to seconds
            times = data["total_ms_mean"] / 1000
            ax.plot(
                data["num_needles"],
                times,
                label=MODE_LABELS.get(mode, mode).replace("\n", " "),
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Number of Needles")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing vs Needle Count\n(type=word, lines=10K)")
    ax.set_xticks([1, 3, 5])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_efficiency(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Reward vs timing scatter (cost-benefit analysis)."""
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode]
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})

        # Filter out rows with missing data
        valid_data = data.dropna(subset=["total_ms_mean", "partial_match_mean"])

        if len(valid_data) > 0:
            # Convert to seconds
            times = valid_data["total_ms_mean"] / 1000

            ax.scatter(
                times,
                valid_data["partial_match_mean"],
                label=MODE_LABELS.get(mode, mode).replace("\n", " "),
                color=style["color"],
                marker=style["marker"],
                s=60,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Partial Match Reward")
    ax.set_title("Timing Efficiency: Reward vs Time\n(all configs, by mode)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
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

    # Aggregate across all configs for each mode
    agg_df = (
        rlm_df.groupby("mode")
        .agg(
            {
                "prompt_tokens_mean": "mean",
                "completion_tokens_mean": "mean",
                "sub_llm_prompt_tokens_mean": "mean",
                "sub_llm_completion_tokens_mean": "mean",
            }
        )
        .reset_index()
    )

    modes = [m for m in ["rlm", "rlm_tips"] if m in agg_df["mode"].unique()]

    x = range(len(modes))
    width = 0.6

    total_tokens = []
    colors = []
    for mode in modes:
        mode_data = agg_df[agg_df["mode"] == mode]
        if len(mode_data) > 0:
            main_prompt = mode_data["prompt_tokens_mean"].values[0] or 0
            main_completion = mode_data["completion_tokens_mean"].values[0] or 0
            sub_prompt = mode_data["sub_llm_prompt_tokens_mean"].values[0] or 0
            sub_completion = mode_data["sub_llm_completion_tokens_mean"].values[0] or 0
            total_tokens.append(
                main_prompt + main_completion + sub_prompt + sub_completion
            )
        else:
            total_tokens.append(0)
        colors.append(MODE_STYLES.get(mode, {"color": "gray"})["color"])

    ax.bar(
        x,
        total_tokens,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Total Tokens")
    ax.set_title("Token Usage by Mode (RLM only)\n(aggregated across all configs)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m).replace("\n", " ") for m in modes])


def create_plots(df: pd.DataFrame, output_path: Path | None = None):
    """Create the 2x3 grid of plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Set style
    sns.set_style("whitegrid")

    # Top row
    plot_mode_by_needle_type(axes[0, 0], df)
    plot_mode_vs_context_size(axes[0, 1], df)
    plot_mode_vs_needle_count(axes[0, 2], df)

    # Bottom row
    plot_heatmap(axes[1, 0], df)
    plot_rlm_metrics(axes[1, 1], df)
    plot_partial_vs_exact(axes[1, 2], df)

    plt.suptitle(
        "Needle in Haystack Ablation Results",
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
    "needle_type": (
        plot_mode_by_needle_type,
        (10, 7),
        "Mode Comparison by Needle Type",
    ),
    "context_size": (plot_mode_vs_context_size, (10, 7), "Mode vs Context Size"),
    "needle_count": (plot_mode_vs_needle_count, (10, 7), "Mode vs Needle Count"),
    "heatmap": (plot_heatmap, (10, 7), "Reward Heatmap"),
    "rlm_metrics": (plot_rlm_metrics, (10, 7), "RLM Usage Metrics"),
    "partial_exact": (plot_partial_vs_exact, (10, 7), "Partial vs Exact Match"),
    "tokens": (plot_token_usage, (10, 7), "Token Usage"),
    # Timing plots
    "timing": (plot_timing_by_mode, (10, 7), "Timing by Mode"),
    "timing_vs_context": (plot_timing_vs_context, (10, 7), "Timing vs Context Size"),
    "timing_vs_needles": (plot_timing_vs_needles, (10, 7), "Timing vs Needle Count"),
    "timing_efficiency": (plot_timing_efficiency, (10, 7), "Timing Efficiency"),
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
        description="Plot needle-in-haystack ablation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show main 2x3 grid
    python plot_results.py
    
    # List available models
    python plot_results.py --list-models
    
    # Filter to a specific model
    python plot_results.py --model gpt-4.1-mini
    
    # Show individual plots
    python plot_results.py --image needle_type
    python plot_results.py --image context_size
    python plot_results.py --image needle_count
    
    # Save plot to file
    python plot_results.py --image needle_type -o needle_type.png
    
    # Timing plots (optional)
    python plot_results.py --image timing              # Basic timing by mode
    python plot_results.py --image timing_vs_context   # Timing scaling with context size
    python plot_results.py --image timing_vs_needles   # Timing scaling with needle count
    python plot_results.py --image timing_efficiency   # Reward vs time tradeoff
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
            "needle_type",
            "context_size",
            "needle_count",
            "heatmap",
            "rlm_metrics",
            "partial_exact",
            "tokens",
            # Timing plots
            "timing",
            "timing_vs_context",
            "timing_vs_needles",
            "timing_efficiency",
        ],
        default="main",
        help="Which plot to generate: 'main' for 2x3 grid, or individual plot name",
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
            print(f"Filtered by model: {original_count} -> {len(df)} configurations")
            if len(matched_models) > 1:
                print(f"  Matched models: {list(matched_models)}")

    print(f"Loaded {len(df)} configurations from {args.input}")

    if args.image == "main":
        create_plots(df, args.output)
    else:
        create_single_plot(args.image, df, args.output)


if __name__ == "__main__":
    main()

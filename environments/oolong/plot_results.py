#!/usr/bin/env python3
"""
Plot ablation results for oolong long-context experiments.

Creates focused plots comparing modes and subsets:
- Mode comparison by model (bar chart)
- Mode comparison by subset (bar chart)
- RLM metrics comparison
- Context length vs reward analysis
- Optional timing plots: timing_by_subset, timing_vs_context, timing_efficiency

Usage:
    python plot_results.py [--input aggregate.csv] [--output plots.png]
    python plot_results.py --image reward
    python plot_results.py --image subset
    python plot_results.py --image timing_by_subset
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

SUBSET_ORDER = ["synth", "synth_with_labels", "real"]
SUBSET_LABELS = {
    "synth": "Synthetic",
    "synth_with_labels": "Synth+Labels",
    "real": "Real",
}


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the aggregate results CSV."""
    df = pd.read_csv(csv_path)

    # Ensure mode and subset columns exist
    if "mode" not in df.columns:
        df["mode"] = "standard"
    if "subset" not in df.columns:
        df["subset"] = "synth"

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
    """Plot: Mode comparison across models (grouped bar chart), aggregated across subsets."""
    # Aggregate across subsets for each model/mode combination
    agg_df = (
        df.groupby(["model", "mode"])
        .agg(
            {
                "judge_reward_mean": "mean",
                "judge_reward_std": "mean",  # Average the stds (approximation)
            }
        )
        .reset_index()
    )

    models = agg_df["model"].unique()
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        rewards = []
        errors = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                rewards.append(model_data["judge_reward_mean"].values[0])
                errors.append(model_data["judge_reward_std"].values[0])
            else:
                rewards.append(0)
                errors.append(0)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        ax.bar(
            [xi + offset for xi in x],
            rewards,
            width,
            yerr=errors,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Mode Comparison by Model\n(aggregated across subsets)")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)


def plot_reward_by_subset(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Mode comparison across subsets (grouped bar chart), aggregated across models."""
    # Aggregate across models for each subset/mode combination
    agg_df = (
        df.groupby(["subset", "mode"])
        .agg(
            {
                "judge_reward_mean": "mean",
                "judge_reward_std": "mean",
            }
        )
        .reset_index()
    )

    subsets = [s for s in SUBSET_ORDER if s in agg_df["subset"].unique()]
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(subsets))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        rewards = []
        errors = []
        for subset in subsets:
            subset_data = mode_data[mode_data["subset"] == subset]
            if len(subset_data) > 0:
                rewards.append(subset_data["judge_reward_mean"].values[0])
                errors.append(subset_data["judge_reward_std"].values[0])
            else:
                rewards.append(0)
                errors.append(0)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        ax.bar(
            [xi + offset for xi in x],
            rewards,
            width,
            yerr=errors,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )

    ax.set_xlabel("Subset")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Mode Comparison by Subset\n(aggregated across models)")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS.get(s, s) for s in subsets])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)


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

    # Aggregate across models and subsets
    agg_df = (
        rlm_df.groupby("mode")
        .agg(
            {
                "main_rlm_turns_mean": "mean",
                "sub_llm_call_count_mean": "mean",
                "sub_llm_mean_batch_size_mean": "mean",
            }
        )
        .reset_index()
    )

    modes = [m for m in ["rlm", "rlm_tips"] if m in agg_df["mode"].unique()]

    # Metrics to show
    metrics = [
        ("main_rlm_turns_mean", "RLM Turns"),
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
    ax.set_title("RLM Usage Metrics by Mode\n(aggregated across models & subsets)")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.legend(loc="upper right", fontsize=9)


def plot_context_vs_reward(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Context length vs reward scatter, colored by mode."""
    # Need raw data for this plot - check if context_length_mean exists
    if "context_length_mean" not in df.columns:
        ax.text(
            0.5,
            0.5,
            "Context length data not available\nin aggregated results",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode]
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})

        # Convert context length to thousands for readability
        context_k = data["context_length_mean"] / 1000

        ax.scatter(
            context_k,
            data["judge_reward_mean"],
            label=MODE_LABELS.get(mode, mode).replace("\n", " "),
            color=style["color"],
            marker=style["marker"],
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add linear fit if enough points
        if len(data) >= 3:
            x_vals = context_k.values
            y_vals = data["judge_reward_mean"].values

            # Filter out NaN values
            mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            if mask.sum() >= 2:
                x_clean = x_vals[mask]
                y_clean = y_vals[mask]
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(
                    x_line,
                    y_line,
                    color=style["color"],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.5,
                )

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Reward vs Context Length\n(by mode)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)


def plot_timing(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing comparison across modes and models."""
    # Aggregate across subsets
    agg_df = (
        df.groupby(["model", "mode"])
        .agg(
            {
                "total_ms_mean": "mean",
            }
        )
        .reset_index()
    )

    models = agg_df["model"].unique()
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
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
    ax.set_title("Average Rollout Time by Mode\n(aggregated across subsets)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    ax.legend(loc="upper right", fontsize=9)


def plot_timing_by_subset(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing comparison across subsets (grouped bar chart)."""
    # Aggregate across models for each subset/mode combination
    agg_df = (
        df.groupby(["subset", "mode"])
        .agg(
            {
                "total_ms_mean": "mean",
            }
        )
        .reset_index()
    )

    subsets = [s for s in SUBSET_ORDER if s in agg_df["subset"].unique()]
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(subsets))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        times = []
        for subset in subsets:
            subset_data = mode_data[mode_data["subset"] == subset]
            if len(subset_data) > 0 and pd.notna(
                subset_data["total_ms_mean"].values[0]
            ):
                times.append(subset_data["total_ms_mean"].values[0] / 1000)
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

    ax.set_xlabel("Subset")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing by Subset\n(aggregated across models)")
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS.get(s, s) for s in subsets])
    ax.legend(loc="upper right", fontsize=9)


def plot_timing_vs_context(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing vs context length scatter, colored by mode."""
    # Need context_length_mean for this plot
    if "context_length_mean" not in df.columns:
        ax.text(
            0.5,
            0.5,
            "Context length data not available\nin aggregated results",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode]
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})

        # Filter out rows with missing data
        valid_data = data.dropna(subset=["context_length_mean", "total_ms_mean"])

        if len(valid_data) > 0:
            # Convert context length to thousands
            context_k = valid_data["context_length_mean"] / 1000
            # Convert timing to seconds
            times = valid_data["total_ms_mean"] / 1000

            ax.scatter(
                context_k,
                times,
                label=MODE_LABELS.get(mode, mode).replace("\n", " "),
                color=style["color"],
                marker=style["marker"],
                s=80,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

            # Add linear fit if enough points
            if len(valid_data) >= 3:
                x_vals = context_k.values
                y_vals = times.values
                mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                if mask.sum() >= 2:
                    x_clean = x_vals[mask]
                    y_clean = y_vals[mask]
                    slope, intercept = np.polyfit(x_clean, y_clean, 1)
                    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(
                        x_line,
                        y_line,
                        color=style["color"],
                        linestyle="--",
                        alpha=0.5,
                        linewidth=1.5,
                    )

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing vs Context Length\n(by mode)")
    ax.legend(loc="upper left", fontsize=8)


def plot_timing_efficiency(ax: plt.Axes, df: pd.DataFrame):
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
                s=60,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Timing Efficiency: Reward vs Time\n(all configs, by mode)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)


def plot_heatmap(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Heatmap of reward by mode × subset."""
    # Create pivot table: mode × subset
    pivot = df.pivot_table(
        values="judge_reward_mean",
        index="mode",
        columns="subset",
        aggfunc="mean",
    )

    # Reorder rows and columns
    row_order = [m for m in MODE_ORDER if m in pivot.index]
    col_order = [s for s in SUBSET_ORDER if s in pivot.columns]

    pivot = pivot.reindex(index=row_order, columns=col_order)

    # Rename for display
    pivot.index = [MODE_LABELS.get(m, m).replace("\n", " ") for m in pivot.index]
    pivot.columns = [SUBSET_LABELS.get(s, s) for s in pivot.columns]

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Judge Reward"},
        linewidths=0.5,
    )

    ax.set_xlabel("Subset")
    ax.set_ylabel("Mode")
    ax.set_title("Reward Heatmap: Mode × Subset")


def create_plots(df: pd.DataFrame, output_path: Path | None = None):
    """Create the 2x3 grid of plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Set style
    sns.set_style("whitegrid")

    # Top row
    plot_reward_by_model(axes[0, 0], df)
    plot_reward_by_subset(axes[0, 1], df)
    plot_heatmap(axes[0, 2], df)

    # Bottom row
    plot_rlm_metrics(axes[1, 0], df)
    plot_timing(axes[1, 1], df)
    plot_context_vs_reward(axes[1, 2], df)

    plt.suptitle(
        "Oolong Long-Context Ablation Results",
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
    "subset": (plot_reward_by_subset, (10, 7), "Mode Comparison by Subset"),
    "heatmap": (plot_heatmap, (10, 7), "Reward Heatmap"),
    "rlm_metrics": (plot_rlm_metrics, (10, 7), "RLM Usage Metrics"),
    "timing": (plot_timing, (10, 7), "Timing Comparison"),
    "context": (plot_context_vs_reward, (10, 7), "Context Length vs Reward"),
    # Additional timing plots
    "timing_by_subset": (plot_timing_by_subset, (10, 7), "Timing by Subset"),
    "timing_vs_context": (plot_timing_vs_context, (10, 7), "Timing vs Context Length"),
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
        description="Plot oolong ablation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show main 2x3 grid
    python plot_results.py
    
    # List available models
    python plot_results.py --list-models
    
    # Filter to a specific model
    python plot_results.py --model gpt-4.1-mini
    
    # Show individual reward plot
    python plot_results.py --image reward
    
    # Show subset comparison
    python plot_results.py --image subset
    
    # Show context length analysis
    python plot_results.py --image context
    
    # Save plot to file
    python plot_results.py --image reward -o reward_plot.png
    
    # Additional timing plots
    python plot_results.py --image timing_by_subset    # Timing by subset
    python plot_results.py --image timing_vs_context   # Timing vs context length
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
            "reward",
            "subset",
            "heatmap",
            "rlm_metrics",
            "timing",
            "context",
            # Additional timing plots
            "timing_by_subset",
            "timing_vs_context",
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

    # Subset filtering options
    parser.add_argument(
        "--subset",
        "-S",
        type=str,
        default=None,
        choices=["synth", "synth_with_labels", "real"],
        help="Filter results to a specific subset",
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

    # Filter by subset if specified
    if args.subset:
        original_count = len(df)
        df = df[df["subset"] == args.subset]
        if len(df) == 0:
            print(f"Error: No results found for subset '{args.subset}'")
            return
        print(f"Filtered by subset: {original_count} -> {len(df)} configurations")

    print(f"Loaded {len(df)} configurations from {args.input}")

    if args.image == "main":
        create_plots(df, args.output)
    else:
        create_single_plot(args.image, df, args.output)


if __name__ == "__main__":
    main()

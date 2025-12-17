#!/usr/bin/env python3
"""
Plot ablation results for verbatim_copy experiments.

Creates a 2x3 grid of plots or individual plots:
- Top row: Mode comparison plots (reward by mode across different dimensions)
- Bottom row: Detailed views (heatmap, distribution, scatter)
- Optional timing plots: timing, timing_by_length, timing_by_content, timing_efficiency

Usage:
    python plot_results.py [--input aggregate.csv] [--output plots.png]
    python plot_results.py --image scatter --linear-fit --exclude-perfect
    python plot_results.py --image content
    python plot_results.py --image timing_by_length
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


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the aggregate results CSV."""
    df = pd.read_csv(csv_path)

    # Convert mean_fragment_length NaN to "None" string for display
    df["frag_label"] = df["mean_fragment_length"].apply(
        lambda x: "None" if pd.isna(x) else str(int(x))
    )

    # Ensure mode column exists (backward compatibility)
    if "mode" not in df.columns:
        df["mode"] = "standard"

    return df


def plot_mode_comparison_by_content(ax: plt.Axes, df: pd.DataFrame):
    """Plot 1: Mode comparison across content types (grouped bar chart)."""
    # Filter to target_length=500, mean_fragment_length=20
    filtered = df[(df["target_length"] == 500) & (df["mean_fragment_length"] == 20)]

    content_types = ["words", "json", "csv", "codes", "mixed"]
    modes = [m for m in MODE_ORDER if m in filtered["mode"].unique()]

    x = range(len(content_types))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = filtered[filtered["mode"] == mode]
        # Reindex to ensure correct order
        rewards = []
        for ct in content_types:
            ct_data = mode_data[mode_data["content_type"] == ct]
            rewards.append(ct_data["reward_mean"].values[0] if len(ct_data) > 0 else 0)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            [xi + offset for xi in x],
            rewards,
            width,
            label=mode,
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Content Type")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Mode Comparison by Content Type\n(len=500, frag=20)")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(content_types)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)


def plot_mode_comparison_by_length(ax: plt.Axes, df: pd.DataFrame):
    """Plot 2: Mode comparison across target lengths (line plot)."""
    # Filter to mean_fragment_length=20, content_type="all"
    filtered = df[(df["mean_fragment_length"] == 20) & (df["content_type"] == "all")]

    modes = [m for m in MODE_ORDER if m in filtered["mode"].unique()]

    for mode in modes:
        data = filtered[filtered["mode"] == mode].sort_values("target_length")
        if len(data) > 0:
            style = MODE_STYLES.get(
                mode, {"color": "gray", "marker": "o", "linestyle": "-"}
            )
            ax.plot(
                data["target_length"],
                data["reward_mean"],
                label=mode,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Target Length (chars)")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Mode Comparison vs Target Length\n(content=all, frag=20)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)


def plot_mode_comparison_by_fragmentation(ax: plt.Axes, df: pd.DataFrame):
    """Plot 3: Mode comparison across fragment lengths (line plot)."""
    # Filter to target_length=500, content_type="all"
    filtered = df[(df["target_length"] == 500) & (df["content_type"] == "all")]

    modes = [m for m in MODE_ORDER if m in filtered["mode"].unique()]
    x_labels_set = False

    for mode in modes:
        data = filtered[filtered["mode"] == mode].copy()
        # Sort by fragment length, putting None first
        data["sort_key"] = data["mean_fragment_length"].fillna(-1)
        data = data.sort_values("sort_key")

        if len(data) > 0:
            style = MODE_STYLES.get(
                mode, {"color": "gray", "marker": "o", "linestyle": "-"}
            )
            x_positions = range(len(data))
            ax.plot(
                x_positions,
                data["reward_mean"],
                label=mode,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

            # Set x-tick labels only once
            if not x_labels_set:
                ax.set_xticks(list(x_positions))
                ax.set_xticklabels(data["frag_label"], rotation=45)
                x_labels_set = True

    ax.set_xlabel("Mean Fragment Length")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Mode Comparison vs Fragment Length\n(content=all, len=500)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)


def plot_heatmap(ax: plt.Axes, df: pd.DataFrame):
    """Plot 4: Heatmap of reward by mode × content_type (aggregated across all configs)."""
    # Filter to target_length=500, mean_fragment_length=20 for cleaner comparison
    filtered = df[
        (df["target_length"] == 500) & (df["mean_fragment_length"] == 20)
    ].copy()

    # Create pivot table: mode × content_type
    pivot = filtered.pivot_table(
        values="reward_mean",
        index="mode",
        columns="content_type",
        aggfunc="mean",
    )

    # Reorder rows and columns
    row_order = [m for m in MODE_ORDER if m in pivot.index]
    col_order = ["words", "json", "csv", "codes", "mixed"]

    pivot = pivot.reindex(
        index=row_order, columns=[c for c in col_order if c in pivot.columns]
    )

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        cbar_kws={"label": "Reward"},
        linewidths=0.5,
    )

    ax.set_xlabel("Content Type")
    ax.set_ylabel("Mode")
    ax.set_title("Reward Heatmap: Mode × Content\n(len=500, frag=20)")


def plot_distribution(ax: plt.Axes, df: pd.DataFrame):
    """Plot 5: Distribution of rewards by mode (across all configs)."""
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    data_for_plot = []
    colors = []
    for mode in modes:
        rewards = df[df["mode"] == mode]["reward_mean"].values
        data_for_plot.append(rewards)
        colors.append(MODE_STYLES.get(mode, {"color": "gray"})["color"])

    if not data_for_plot:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    parts = ax.violinplot(data_for_plot, positions=range(len(modes)), showmeans=True)

    # Color the violins
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Style the other parts
    for partname in ["cbars", "cmins", "cmaxes", "cmeans"]:
        parts[partname].set_color("black")

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Reward Distribution by Mode\n(all configs)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)


def plot_char_vs_exact(
    ax: plt.Axes,
    df: pd.DataFrame,
    linear_fit: bool = False,
    exclude_perfect: bool = False,
):
    """Plot 6: Char accuracy vs Exact match scatter, colored by mode.

    Args:
        ax: Matplotlib axes to plot on
        df: DataFrame with aggregated results
        linear_fit: If True, add linear regression lines for each mode
        exclude_perfect: If True, exclude exact_match=1.0 points from linear fit
    """
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode]
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})
        ax.scatter(
            data["exact_match_mean"],
            data["char_accuracy_mean"],
            label=mode,
            color=style["color"],
            marker=style["marker"],
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add linear fit if requested
        if linear_fit:
            fit_data = data
            if exclude_perfect:
                fit_data = data[data["exact_match_mean"] < 1.0]

            if len(fit_data) >= 2:
                x = fit_data["exact_match_mean"].values
                y = fit_data["char_accuracy_mean"].values

                # Linear regression using numpy
                slope, intercept = np.polyfit(x, y, 1)

                # Calculate R² value
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Plot fit line across the data range
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept

                fit_label = f"{mode} fit (R²={r_squared:.2f})"
                ax.plot(
                    x_line,
                    y_line,
                    color=style["color"],
                    linestyle="--",
                    alpha=0.6,
                    linewidth=1.5,
                    label=fit_label,
                )

    # Add diagonal line (perfect correlation)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")

    ax.set_xlabel("Exact Match (Reward)")
    ax.set_ylabel("Character Accuracy")

    # Update title based on options
    title = "Char Accuracy vs Exact Match\n(all configs, by mode)"
    if linear_fit and exclude_perfect:
        title += "\n[fit excludes perfect matches]"
    ax.set_title(title)

    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc="lower right", fontsize=8)

    # Add annotation for "near misses" region
    ax.annotate(
        "Near misses\n(high char acc,\nlow exact match)",
        xy=(0.7, 0.99),
        fontsize=8,
        ha="center",
        style="italic",
        alpha=0.7,
    )


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

    bars = ax.bar(
        x, times, yerr=errors, color=colors, edgecolor="black", linewidth=0.5, capsize=3
    )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Average Rollout Time by Mode\n(aggregated across all configs)")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)


def plot_timing_by_length(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing vs target length by mode (line plot)."""
    # Filter to mean_fragment_length=20, content_type="all"
    filtered = df[(df["mean_fragment_length"] == 20) & (df["content_type"] == "all")]

    modes = [m for m in MODE_ORDER if m in filtered["mode"].unique()]

    for mode in modes:
        data = filtered[filtered["mode"] == mode].sort_values("target_length")
        if len(data) > 0 and "total_ms_mean" in data.columns:
            style = MODE_STYLES.get(
                mode, {"color": "gray", "marker": "o", "linestyle": "-"}
            )
            # Convert to seconds
            times = data["total_ms_mean"] / 1000
            ax.plot(
                data["target_length"],
                times,
                label=mode,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Target Length (chars)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing vs Target Length\n(content=all, frag=20)")
    ax.legend(loc="upper left", fontsize=8)


def plot_timing_by_content(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing by content type across modes (grouped bar chart)."""
    # Filter to target_length=500, mean_fragment_length=20
    filtered = df[(df["target_length"] == 500) & (df["mean_fragment_length"] == 20)]

    content_types = ["words", "json", "csv", "codes", "mixed"]
    modes = [m for m in MODE_ORDER if m in filtered["mode"].unique()]

    x = range(len(content_types))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = filtered[filtered["mode"] == mode]
        times = []
        for ct in content_types:
            ct_data = mode_data[mode_data["content_type"] == ct]
            if len(ct_data) > 0 and pd.notna(ct_data["total_ms_mean"].values[0]):
                times.append(ct_data["total_ms_mean"].values[0] / 1000)
            else:
                times.append(0)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        ax.bar(
            [xi + offset for xi in x],
            times,
            width,
            label=mode,
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Content Type")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing by Content Type\n(len=500, frag=20)")
    ax.set_xticks(x)
    ax.set_xticklabels(content_types)
    ax.legend(loc="upper right", fontsize=8)


def plot_timing_efficiency(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Reward vs timing scatter (cost-benefit analysis)."""
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode]
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})

        # Filter out rows with missing timing data
        valid_data = data.dropna(subset=["total_ms_mean", "reward_mean"])

        if len(valid_data) > 0:
            # Convert to seconds
            times = valid_data["total_ms_mean"] / 1000

            ax.scatter(
                times,
                valid_data["reward_mean"],
                label=mode,
                color=style["color"],
                marker=style["marker"],
                s=60,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Timing Efficiency: Reward vs Time\n(all configs, by mode)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)


def create_plots(df: pd.DataFrame, output_path: Path | None = None):
    """Create the 2x3 grid of plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Set style
    sns.set_style("whitegrid")

    # Top row: Mode comparison plots
    plot_mode_comparison_by_content(axes[0, 0], df)
    plot_mode_comparison_by_length(axes[0, 1], df)
    plot_mode_comparison_by_fragmentation(axes[0, 2], df)

    # Bottom row: Aggregations and detailed views
    plot_heatmap(axes[1, 0], df)
    plot_distribution(axes[1, 1], df)
    plot_char_vs_exact(axes[1, 2], df)

    plt.suptitle(
        "Verbatim Copy Ablation Results: Mode Comparison",
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
    "content": (
        plot_mode_comparison_by_content,
        (10, 7),
        "Mode Comparison by Content Type",
    ),
    "length": (
        plot_mode_comparison_by_length,
        (10, 7),
        "Mode Comparison vs Target Length",
    ),
    "fragmentation": (
        plot_mode_comparison_by_fragmentation,
        (10, 7),
        "Mode Comparison vs Fragment Length",
    ),
    "heatmap": (plot_heatmap, (10, 7), "Reward Heatmap"),
    "distribution": (plot_distribution, (10, 7), "Reward Distribution by Mode"),
    "scatter": (plot_char_vs_exact, (10, 7), "Char Accuracy vs Exact Match"),
    # Timing plots
    "timing": (plot_timing_by_mode, (10, 7), "Timing by Mode"),
    "timing_by_length": (plot_timing_by_length, (10, 7), "Timing vs Target Length"),
    "timing_by_content": (plot_timing_by_content, (10, 7), "Timing by Content Type"),
    "timing_efficiency": (plot_timing_efficiency, (10, 7), "Timing Efficiency"),
}


def create_single_plot(
    plot_name: str,
    df: pd.DataFrame,
    output_path: Path | None = None,
    **kwargs,
):
    """Create a single standalone plot.

    Args:
        plot_name: Name of the plot to create (from PLOT_REGISTRY)
        df: DataFrame with aggregated results
        output_path: Optional path to save the plot
        **kwargs: Additional arguments passed to the plot function (e.g., linear_fit)
    """
    if plot_name not in PLOT_REGISTRY:
        raise ValueError(
            f"Unknown plot: {plot_name}. Available: {list(PLOT_REGISTRY.keys())}"
        )

    func, figsize, title = PLOT_REGISTRY[plot_name]

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")

    # Pass kwargs only to functions that accept them (scatter plot)
    if plot_name == "scatter":
        func(ax, df, **kwargs)
    else:
        func(ax, df)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot verbatim_copy ablation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show main 2x3 grid
    python plot_results.py
    
    # List available models
    python plot_results.py --list-models
    
    # Filter to a specific model
    python plot_results.py --model gpt-5-mini
    
    # Substring match for model (handy for OpenRouter models)
    python plot_results.py --model claude
    
    # Combine model filter with specific plot
    python plot_results.py -M gpt-5-mini --image scatter --linear-fit
    
    # Show individual scatter plot with linear fit
    python plot_results.py --image scatter --linear-fit
    
    # Scatter plot with fit excluding perfect matches
    python plot_results.py --image scatter --linear-fit --exclude-perfect
    
    # Save individual plot to file
    python plot_results.py --image content -o content_plot.png
    
    # Timing plots (optional)
    python plot_results.py --image timing              # Basic timing by mode
    python plot_results.py --image timing_by_length    # Timing scaling with text length
    python plot_results.py --image timing_by_content   # Timing by content type
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
            "content",
            "length",
            "fragmentation",
            "heatmap",
            "distribution",
            "scatter",
            # Timing plots
            "timing",
            "timing_by_length",
            "timing_by_content",
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

    # Scatter plot specific options
    scatter_group = parser.add_argument_group("scatter plot options")
    scatter_group.add_argument(
        "--linear-fit",
        action="store_true",
        help="Add linear regression lines for each mode (scatter plot only)",
    )
    scatter_group.add_argument(
        "--exclude-perfect",
        action="store_true",
        help="Exclude exact_match=1.0 points from linear fit calculation (scatter plot only)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run aggregate_results.py first to generate the CSV.")
        return

    df = load_data(args.input)

    # Handle --list-models (early exit like --help)
    if args.list_models:
        if "model" not in df.columns:
            print("No model column found in data.")
            print("Re-run aggregate_results.py to include model information.")
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
            print("Re-run aggregate_results.py to include model information.")
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
        # Build kwargs for plot-specific options
        kwargs = {}
        if args.image == "scatter":
            kwargs["linear_fit"] = args.linear_fit
            kwargs["exclude_perfect"] = args.exclude_perfect

        create_single_plot(args.image, df, args.output, **kwargs)


if __name__ == "__main__":
    main()

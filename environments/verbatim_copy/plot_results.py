#!/usr/bin/env python3
"""
Plot ablation results for verbatim_copy experiments.

Creates a 2x3 grid of plots or individual plots:
- Top row: Reward by mode, Main Model Token Usage, Rollout Time
- Bottom row: By Content Type, Scaling Behavior (2-panel), Char Accuracy vs Exact Match
- Optional timing plots: timing_by_length, timing_by_content, timing_efficiency

Usage:
    python plot_results.py [--input aggregate.csv] [--output plots.png]
    python plot_results.py --image scatter --linear-fit --exclude-perfect
    python plot_results.py --image content
    python plot_results.py --image timing_by_length
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

# Set style at module level so it applies to all plots
sns.set_style("whitegrid")

# Mode styling for consistent visualization
MODE_STYLES = {
    "standard": {"color": "#E24A33", "marker": "o", "linestyle": "-"},
    "rlm": {"color": "#348ABD", "marker": "s", "linestyle": "--"},
    "rlm_tips": {"color": "#988ED5", "marker": "^", "linestyle": ":"},
}

MODE_ORDER = ["standard", "rlm", "rlm_tips"]
MODE_LABELS = {
    "standard": "LLM",
    "rlm": "RLM",
    "rlm_tips": "RLM+tips",
}

# Hatching patterns for different models (dense patterns for thin/low bars)
HATCHES = ["///", "...", "+++", "***", "\\\\\\", "xxx", "ooo", "|||", "---"]


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


def normalize_model_name(model: str) -> str:
    """Create display-friendly model name."""
    if model.startswith("openrouter/"):
        parts = model.split("/")
        return parts[-1] if len(parts) >= 3 else model
    if "/" in model:
        return model.split("/")[-1]
    return model


def plot_reward_by_mode(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Mode comparison across all configs (grouped bar chart)."""
    # Aggregate across all configs for each mode
    agg_dict = {"reward_mean": "mean"}
    if "reward_count" in df.columns:
        agg_dict["reward_count"] = "sum"

    agg_df = df.groupby("mode").agg(agg_dict).reset_index()

    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(modes))
    width = 0.6

    rewards = []
    counts = []
    colors = []
    for mode in modes:
        mode_data = agg_df[agg_df["mode"] == mode]
        if len(mode_data) > 0:
            rewards.append(mode_data["reward_mean"].values[0])
            if "reward_count" in mode_data.columns:
                counts.append(int(mode_data["reward_count"].values[0]))
            else:
                counts.append(None)
        else:
            rewards.append(0)
            counts.append(None)
        colors.append(MODE_STYLES.get(mode, {"color": "gray"})["color"])

    bars = ax.bar(
        x,
        rewards,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add sample size labels above bars if requested
    if show_counts:
        for bar, count in zip(bars, counts):
            if count is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Reward")
    ax.set_ylim(0, 1.2 if show_counts else 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)


def plot_main_model_tokens(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Main model token usage comparison (excludes sub-LLM tokens), split by model."""
    if len(df) == 0:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]

        # Main model tokens only (fair comparison across modes)
        total_tokens = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                prompt = model_data["prompt_tokens_mean"].values[0] or 0
                completion = model_data["completion_tokens_mean"].values[0] or 0
                total_tokens.append(prompt + completion)
                if "prompt_tokens_count" in model_data.columns:
                    counts.append(int(model_data["prompt_tokens_count"].values[0]))
                else:
                    counts.append(None)
            else:
                total_tokens.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            [xi + offset for xi in x],
            total_tokens,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add sample size labels above bars if requested
        if show_counts:
            for bar, count in zip(bars, counts):
                if count is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + bar.get_height() * 0.02,
                        f"n={count}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="gray",
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Tokens (Main Model Only)")
    ax.set_title("Main Model Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_by_mode(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Timing comparison across modes (bar chart)."""
    # Aggregate across all configs for each mode
    agg_dict = {"total_ms_mean": "mean"}
    if "total_ms_count" in df.columns:
        agg_dict["total_ms_count"] = "sum"

    agg_df = df.groupby("mode").agg(agg_dict).reset_index()

    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]
    x = range(len(modes))

    times = []
    counts = []
    colors = []
    for mode in modes:
        mode_data = agg_df[agg_df["mode"] == mode]
        if len(mode_data) > 0 and pd.notna(mode_data["total_ms_mean"].values[0]):
            times.append(
                mode_data["total_ms_mean"].values[0] / 1000
            )  # Convert to seconds
            if "total_ms_count" in mode_data.columns:
                counts.append(int(mode_data["total_ms_count"].values[0]))
            else:
                counts.append(None)
        else:
            times.append(0)
            counts.append(None)
        colors.append(MODE_STYLES.get(mode, {"color": "gray"})["color"])

    bars = ax.bar(
        x,
        times,
        0.6,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add sample size labels above bars if requested
    if show_counts:
        for bar, count in zip(bars, counts):
            if count is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Average Rollout Time")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes])


def plot_mode_comparison_by_content(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Mode comparison across content types (grouped bar chart)."""
    # Filter to target_length=500, mean_fragment_length=20
    filtered = df[(df["target_length"] == 500) & (df["mean_fragment_length"] == 20)]

    content_types = ["words", "json", "csv", "codes", "mixed"]
    modes = [m for m in MODE_ORDER if m in filtered["mode"].unique()]

    x = range(len(content_types))
    width = 0.25

    all_bars = []
    all_counts = []

    for i, mode in enumerate(modes):
        mode_data = filtered[filtered["mode"] == mode]
        rewards = []
        counts = []
        for ct in content_types:
            ct_data = mode_data[mode_data["content_type"] == ct]
            if len(ct_data) > 0:
                rewards.append(ct_data["reward_mean"].values[0])
                if "reward_count" in ct_data.columns:
                    counts.append(int(ct_data["reward_count"].values[0]))
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
        all_bars.append(bars)
        all_counts.append(counts)

    # Add sample size labels above bars if requested (rotated to avoid overlap)
    if show_counts:
        for bars, counts in zip(all_bars, all_counts):
            for bar, count in zip(bars, counts):
                if count is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"n={count}",
                        ha="center",
                        va="bottom",
                        fontsize=5,
                        color="gray",
                        rotation=90,
                    )

    ax.set_xlabel("Content Type")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Reward by Content Type\n(total length=500, fragment length=20)")
    ax.set_ylim(0, 1.3 if show_counts else 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(content_types)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_scaling_behavior(ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True):
    """Plot: Combined scaling plot with twin x-axes for length and fragmentation.

    Bottom x-axis: Target Length (solid lines)
    Top x-axis: Fragment Length (dotted lines)
    """
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    # Primary axis (bottom): Target Length - solid lines
    filtered_length = df[
        (df["mean_fragment_length"] == 20) & (df["content_type"] == "all")
    ]

    for mode in modes:
        data = filtered_length[filtered_length["mode"] == mode].sort_values(
            "target_length"
        )
        if len(data) > 0:
            style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})
            ax.plot(
                data["target_length"],
                data["reward_mean"],
                marker=style["marker"],
                color=style["color"],
                linestyle="-",  # Solid for length
                linewidth=2,
                markersize=5,
            )

    ax.set_xlabel("Target Length (solid lines)")
    ax.set_ylabel("Reward")
    ax.set_ylim(0.6, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    # Secondary axis (top): Fragment Length - dotted lines
    ax2 = ax.twiny()
    filtered_frag = df[(df["target_length"] == 500) & (df["content_type"] == "all")]

    frag_labels = None
    for mode in modes:
        data = filtered_frag[filtered_frag["mode"] == mode].copy()
        # Sort by fragment length, putting None (no fragmentation = largest) last
        data["sort_key"] = data["mean_fragment_length"].fillna(float("inf"))
        data = data.sort_values("sort_key")

        if len(data) > 0:
            style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})
            x_positions = list(range(len(data)))
            ax2.plot(
                x_positions,
                data["reward_mean"],
                marker=style["marker"],
                color=style["color"],
                linestyle=":",  # Dotted for fragmentation
                linewidth=2,
                markersize=5,
            )
            if frag_labels is None:
                frag_labels = data["frag_label"].tolist()

    if frag_labels:
        ax2.set_xticks(list(range(len(frag_labels))))
        ax2.set_xticklabels(frag_labels)
    ax2.set_xlabel("Fragment Length (dotted lines)")

    ax.set_title("Scaling Behavior")


def plot_mode_comparison_by_length(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True
):
    """Plot: Mode comparison across target lengths (line plot)."""
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
                label=MODE_LABELS.get(mode, mode),
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Target Length (chars)")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Reward vs Total Length\n(content type=all, fragment length=20)")
    # Let matplotlib auto-scale y-axis
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_mode_comparison_by_fragmentation(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True
):
    """Plot: Mode comparison across fragment lengths (line plot)."""
    # Filter to target_length=500, content_type="all"
    target_length = 500
    filtered = df[
        (df["target_length"] == target_length) & (df["content_type"] == "all")
    ]

    modes = [m for m in MODE_ORDER if m in filtered["mode"].unique()]
    x_labels_set = False

    for mode in modes:
        data = filtered[filtered["mode"] == mode].copy()
        # Sort by fragment length, putting None (no fragmentation = largest) last
        data["sort_key"] = data["mean_fragment_length"].fillna(float("inf"))
        data = data.sort_values("sort_key")

        if len(data) > 0:
            style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})
            x_positions = range(len(data))
            ax.plot(
                x_positions,
                data["reward_mean"],
                label=MODE_LABELS.get(mode, mode),
                marker=style["marker"],
                color=style["color"],
                linestyle="-",  # Solid lines for individual plot
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
    ax.set_title(
        f"Reward vs Fragment Length\n(total length={target_length}, content type=all)"
    )
    # Let matplotlib auto-scale y-axis
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_char_vs_exact(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    linear_fit: bool = False,
    exclude_perfect: bool = False,
):
    """Plot: Char accuracy vs Exact match scatter, colored by mode.

    Args:
        ax: Matplotlib axes to plot on
        df: DataFrame with aggregated results
        show_legend: Whether to show legend
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
            label=MODE_LABELS.get(mode, mode),
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

                fit_label = f"{MODE_LABELS.get(mode, mode)} fit (R²={r_squared:.2f})"
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
    title = "Char Accuracy vs Exact Match"
    if linear_fit and exclude_perfect:
        title += "\n[fit excludes perfect]"
    ax.set_title(title)

    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.85, 1.02)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)


def plot_heatmap(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Heatmap of reward by mode × content_type."""
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

    # Rename for display
    pivot.index = [MODE_LABELS.get(m, m) for m in pivot.index]

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
    """Plot: Distribution of rewards by mode (across all configs)."""
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
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes])
    ax.set_xlabel("Mode")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Reward Distribution by Mode")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)


def plot_token_usage(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Total token usage comparison across all modes, split by model."""
    if len(df) == 0:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]

        # Total tokens = main model + sub-LLM tokens (sub-LLM will be 0 for standard)
        total_tokens = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                main_prompt = model_data["prompt_tokens_mean"].values[0] or 0
                main_completion = model_data["completion_tokens_mean"].values[0] or 0
                # Handle missing/NaN sub-LLM columns for standard mode
                sub_prompt = 0
                sub_completion = 0
                if "sub_llm_prompt_tokens_mean" in model_data.columns:
                    val = model_data["sub_llm_prompt_tokens_mean"].values[0]
                    sub_prompt = 0 if pd.isna(val) else val
                if "sub_llm_completion_tokens_mean" in model_data.columns:
                    val = model_data["sub_llm_completion_tokens_mean"].values[0]
                    sub_completion = 0 if pd.isna(val) else val
                total_tokens.append(
                    main_prompt + main_completion + sub_prompt + sub_completion
                )
                if "prompt_tokens_count" in model_data.columns:
                    counts.append(int(model_data["prompt_tokens_count"].values[0]))
                else:
                    counts.append(None)
            else:
                total_tokens.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            [xi + offset for xi in x],
            total_tokens,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add sample size labels above bars if requested
        if show_counts:
            for bar, count in zip(bars, counts):
                if count is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + bar.get_height() * 0.02,
                        f"n={count}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="gray",
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Total Tokens")
    ax.set_title("Total Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_by_length(ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True):
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
                label=MODE_LABELS.get(mode, mode),
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Target Length (chars)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing vs Target Length\n(content=all, frag=20)")
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_by_content(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
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
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Content Type")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing by Content Type\n(len=500, frag=20)")
    ax.set_xticks(x)
    ax.set_xticklabels(content_types)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_efficiency(ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True):
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
                label=MODE_LABELS.get(mode, mode),
                color=style["color"],
                marker=style["marker"],
                s=60,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Timing Efficiency: Reward vs Time")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def create_plots(
    df: pd.DataFrame, output_path: Path | None = None, show_counts: bool = False
):
    """Create the 2x3 grid of plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Top row: Standard metrics (like other environments)
    plot_reward_by_mode(axes[0, 0], df, show_legend=False, show_counts=show_counts)
    plot_main_model_tokens(axes[0, 1], df, show_legend=False, show_counts=show_counts)
    plot_timing_by_mode(axes[0, 2], df, show_legend=False, show_counts=show_counts)

    # Bottom row: Verbatim-copy specific
    plot_mode_comparison_by_content(
        axes[1, 0], df, show_legend=False, show_counts=show_counts
    )
    plot_scaling_behavior(axes[1, 1], df, show_legend=False)
    plot_char_vs_exact(axes[1, 2], df, show_legend=False, linear_fit=True)

    # Create central legend for modes (using markers for consistency)
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]
    legend_handles = []
    for mode in modes:
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})
        mode_label = MODE_LABELS.get(mode, mode).replace("\n", " ")
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="w",
                markerfacecolor=style["color"],
                markeredgecolor="black",
                markersize=10,
                label=mode_label,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(modes),
        fontsize=10,
    )

    plt.suptitle(
        "Verbatim Copy",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    # Make room for the central legend at the bottom
    plt.subplots_adjust(bottom=0.08)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


# Mapping of plot names to (function, figsize, title)
PLOT_REGISTRY = {
    "reward": (plot_reward_by_mode, (10, 7), "Reward"),
    "main_tokens": (plot_main_model_tokens, (10, 7), "Main Model Token Usage"),
    "tokens": (plot_token_usage, (10, 7), "Total Token Usage"),
    "timing": (plot_timing_by_mode, (10, 7), "Average Rollout Time"),
    "content": (plot_mode_comparison_by_content, (10, 7), ""),  # No suptitle
    "scaling": (plot_scaling_behavior, (10, 7), "Scaling Behavior"),
    "length": (plot_mode_comparison_by_length, (10, 7), ""),  # No suptitle
    "fragmentation": (
        plot_mode_comparison_by_fragmentation,
        (10, 7),
        "",  # No suptitle - the axis title is sufficient
    ),
    "heatmap": (plot_heatmap, (10, 7), "Reward Heatmap"),
    "distribution": (plot_distribution, (10, 7), "Reward Distribution"),
    "scatter": (plot_char_vs_exact, (10, 7), "Char Accuracy vs Exact Match"),
    # Timing plots
    "timing_by_length": (plot_timing_by_length, (10, 7), "Timing vs Target Length"),
    "timing_by_content": (plot_timing_by_content, (10, 7), "Timing by Content Type"),
    "timing_efficiency": (plot_timing_efficiency, (10, 7), "Timing Efficiency"),
}


def create_single_plot(
    plot_name: str,
    df: pd.DataFrame,
    output_path: Path | None = None,
    show_counts: bool = False,
    **kwargs,
):
    """Create a single standalone plot.

    Args:
        plot_name: Name of the plot to create (from PLOT_REGISTRY)
        df: DataFrame with aggregated results
        output_path: Optional path to save the plot
        show_counts: Whether to show sample counts
        **kwargs: Additional arguments passed to the plot function (e.g., linear_fit)
    """
    if plot_name not in PLOT_REGISTRY:
        raise ValueError(
            f"Unknown plot: {plot_name}. Available: {list(PLOT_REGISTRY.keys())}"
        )

    func, figsize, title = PLOT_REGISTRY[plot_name]

    fig, ax = plt.subplots(figsize=figsize)

    # Pass appropriate arguments based on plot type
    if plot_name == "scatter":
        func(ax, df, **kwargs)
    elif plot_name in (
        "reward",
        "main_tokens",
        "tokens",
        "timing",
        "content",
        "timing_by_content",
    ):
        func(ax, df, show_counts=show_counts)
    else:
        func(ax, df)

    if title:
        plt.suptitle(f"Verbatim Copy: {title}", fontsize=14, fontweight="bold", y=1.02)
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
    
    # Show sample counts on bars
    python plot_results.py --show-counts
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
            "main_tokens",
            "tokens",
            "timing",
            "content",
            "scaling",
            "length",
            "fragmentation",
            "heatmap",
            "distribution",
            "scatter",
            # Timing plots
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
        nargs="*",
        type=str,
        default=None,
        help="Filter results to specific models (supports substring matching, multiple models allowed)",
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

    parser.add_argument(
        "--show-counts",
        "-c",
        action="store_true",
        help="Show sample counts (n=X) above bars in bar chart plots",
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
            # Build combined mask for all model patterns
            combined_mask = None
            for model_pattern in args.model:
                # Try exact match first, then substring
                if model_pattern in df["model"].unique():
                    mask = df["model"] == model_pattern
                else:
                    mask = df["model"].str.contains(model_pattern, case=False, na=False)
                combined_mask = (
                    mask if combined_mask is None else (combined_mask | mask)
                )

            if combined_mask is not None:
                df = df[combined_mask]

            if len(df) == 0:
                print(f"Error: No results found for models '{args.model}'")
                print("Use --list-models to see available models.")
                return

            matched_models = df["model"].unique()
            print(f"Filtered: {original_count} -> {len(df)} configurations")
            if len(matched_models) > 1:
                print(f"  Matched models: {list(matched_models)}")

    print(f"Loaded {len(df)} configurations from {args.input}")

    if args.image == "main":
        create_plots(df, args.output, show_counts=args.show_counts)
    else:
        # Build kwargs for plot-specific options
        kwargs = {}
        if args.image == "scatter":
            kwargs["linear_fit"] = args.linear_fit
            kwargs["exclude_perfect"] = args.exclude_perfect

        create_single_plot(
            args.image, df, args.output, show_counts=args.show_counts, **kwargs
        )


if __name__ == "__main__":
    main()

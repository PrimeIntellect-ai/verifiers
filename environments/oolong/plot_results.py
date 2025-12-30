#!/usr/bin/env python3
"""
Plot ablation results for oolong long-context experiments.

Creates focused plots comparing modes and subsets:
- Mode comparison by model (bar chart)
- Mode comparison by subset (bar chart)
- RLM metrics comparison
- Context length vs reward analysis
- Optional timing plots: timing_by_subset, timing_vs_context, timing_efficiency
- Raw data context plots: context_scatter, context_binned, context_rolling
  (These show per-example reward vs context length)

Usage:
    python plot_results.py [--input aggregate.csv] [--output plots.png]
    python plot_results.py --image reward
    python plot_results.py --image subset
    python plot_results.py --image timing_by_subset

    # Raw data plots (require running aggregate_results.py --raw-output first):
    python plot_results.py --image context_scatter
    python plot_results.py --image context_binned
    python plot_results.py --image context_rolling
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DARK_FIGURE_BG = "#0f111a"
DARK_AXES_BG = "#0f111a"
DARK_GRID_COLOR = "#2a2f38"
DARK_TEXT_COLOR = "#e6e6e6"
DARK_MUTED_TEXT_COLOR = "#b3b3b3"
DARK_EDGE_COLOR = "#d0d0d0"

LIGHT_FIGURE_BG = "#ffffff"
LIGHT_AXES_BG = "#ffffff"
LIGHT_GRID_COLOR = "#e0e0e0"
LIGHT_TEXT_COLOR = "#222222"
LIGHT_MUTED_TEXT_COLOR = "#555555"
LIGHT_EDGE_COLOR = "#333333"

GRID_COLOR = DARK_GRID_COLOR
TEXT_COLOR = DARK_TEXT_COLOR
MUTED_TEXT_COLOR = DARK_MUTED_TEXT_COLOR
EDGE_COLOR = DARK_EDGE_COLOR


def apply_dark_style() -> None:
    global GRID_COLOR, TEXT_COLOR, MUTED_TEXT_COLOR, EDGE_COLOR
    GRID_COLOR = DARK_GRID_COLOR
    TEXT_COLOR = DARK_TEXT_COLOR
    MUTED_TEXT_COLOR = DARK_MUTED_TEXT_COLOR
    EDGE_COLOR = DARK_EDGE_COLOR
    sns.set_theme(
        style="darkgrid",
        rc={
            "figure.facecolor": DARK_FIGURE_BG,
            "axes.facecolor": DARK_AXES_BG,
            "savefig.facecolor": DARK_FIGURE_BG,
            "savefig.edgecolor": DARK_FIGURE_BG,
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "legend.frameon": True,
            "legend.facecolor": DARK_AXES_BG,
            "legend.edgecolor": GRID_COLOR,
        },
    )


def apply_light_style() -> None:
    global GRID_COLOR, TEXT_COLOR, MUTED_TEXT_COLOR, EDGE_COLOR
    GRID_COLOR = LIGHT_GRID_COLOR
    TEXT_COLOR = LIGHT_TEXT_COLOR
    MUTED_TEXT_COLOR = LIGHT_MUTED_TEXT_COLOR
    EDGE_COLOR = LIGHT_EDGE_COLOR
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": LIGHT_FIGURE_BG,
            "axes.facecolor": LIGHT_AXES_BG,
            "savefig.facecolor": LIGHT_FIGURE_BG,
            "savefig.edgecolor": LIGHT_FIGURE_BG,
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "legend.frameon": True,
            "legend.facecolor": LIGHT_AXES_BG,
            "legend.edgecolor": GRID_COLOR,
        },
    )


apply_dark_style()

# Mode styling for consistent visualization
MODE_STYLES = {
    "standard": {"color": "#E24A33", "marker": "o", "linestyle": "-"},
    "rlm": {"color": "#348ABD", "marker": "s", "linestyle": "--"},
    "rlm_tips": {"color": "#988ED5", "marker": "^", "linestyle": ":"},
    "rlm_tips_v2": {"color": "#2CA02C", "marker": "D", "linestyle": "-."},
}

MODE_ORDER = ["standard", "rlm", "rlm_tips", "rlm_tips_v2"]
MODE_LABELS = {
    "standard": "LLM",
    "rlm": "RLM",
    "rlm_tips": "RLM+tips",
    "rlm_tips_v2": "RLM+tips v2",
}

SUBSET_ORDER = ["synth", "synth_with_labels", "real"]
SUBSET_LABELS = {
    "synth": "Synthetic",
    "synth_with_labels": "Synth+Labels",
    "real": "Real",
}

# Models to exclude from multi-subplot grid plots by default (partial match)
EXCLUDED_MODELS = {"mimo", "deepseek"}

# Hatching patterns for different models (dense patterns for thin/low bars)
HATCHES = ["///", "...", "+++", "***", "\\\\\\", "xxx", "ooo", "|||", "---"]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the aggregate results CSV."""
    df = pd.read_csv(csv_path)

    # Ensure mode and subset columns exist
    if "mode" not in df.columns:
        df["mode"] = "standard"
    if "subset" not in df.columns:
        df["subset"] = "synth"

    return df


def load_raw_data(raw_csv_path: Path) -> pd.DataFrame:
    """Load raw per-example results CSV."""
    df = pd.read_csv(raw_csv_path)

    # Ensure required columns exist
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


def plot_reward_by_model(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Plot: Mode comparison across models (grouped bar chart), aggregated across subsets."""
    # Handle both raw data (judge_reward) and aggregated data (judge_reward_mean)
    if "judge_reward" in df.columns:
        # Raw data - aggregate on the fly
        agg_df = (
            df.groupby(["model", "mode"])
            .agg(
                judge_reward_mean=("judge_reward", "mean"),
                judge_reward_count=("judge_reward", "count"),
            )
            .reset_index()
        )
    else:
        # Already aggregated data
        agg_dict = {"judge_reward_mean": "mean"}
        if "judge_reward_count" in df.columns:
            agg_dict["judge_reward_count"] = "sum"
        agg_df = df.groupby(["model", "mode"]).agg(agg_dict).reset_index()

    models = agg_df["model"].unique()
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        rewards = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                rewards.append(model_data["judge_reward_mean"].values[0])
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
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, reward, count in zip(bars, rewards, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{reward:.2f}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=MUTED_TEXT_COLOR,
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Reward")
    if absolute:
        extra = 0.1 + (0.05 if show_counts else 0) + (0.05 if show_values else 0)
        ax.set_ylim(0, 1.0 + extra)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    if absolute:
        ax.axhline(y=1.0, color=MUTED_TEXT_COLOR, linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_reward_by_subset(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Plot: Mode comparison across subsets (grouped bar chart), aggregated across models."""
    # Handle both raw data (judge_reward) and aggregated data (judge_reward_mean)
    if "judge_reward" in df.columns:
        # Raw data - aggregate on the fly
        agg_df = (
            df.groupby(["subset", "mode"])
            .agg(
                judge_reward_mean=("judge_reward", "mean"),
                judge_reward_count=("judge_reward", "count"),
            )
            .reset_index()
        )
    else:
        # Already aggregated data
        agg_dict = {"judge_reward_mean": "mean"}
        if "judge_reward_count" in df.columns:
            agg_dict["judge_reward_count"] = "sum"
        agg_df = df.groupby(["subset", "mode"]).agg(agg_dict).reset_index()

    subsets = [s for s in SUBSET_ORDER if s in agg_df["subset"].unique()]
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(subsets))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        rewards = []
        counts = []
        for subset in subsets:
            subset_data = mode_data[mode_data["subset"] == subset]
            if len(subset_data) > 0:
                rewards.append(subset_data["judge_reward_mean"].values[0])
                if "judge_reward_count" in subset_data.columns:
                    counts.append(int(subset_data["judge_reward_count"].values[0]))
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
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, reward, count in zip(bars, rewards, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{reward:.2f}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=MUTED_TEXT_COLOR,
                    )

    ax.set_xlabel("Subset")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Reward by Subset")
    if absolute:
        extra = 0.1 + (0.05 if show_counts else 0) + (0.05 if show_values else 0)
        ax.set_ylim(0, 1.0 + extra)
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS.get(s, s) for s in subsets])
    if absolute:
        ax.axhline(y=1.0, color=MUTED_TEXT_COLOR, linestyle="--", alpha=0.5)
    if show_legend:
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

    # Handle raw data vs aggregated data
    if "turns" in df.columns:
        # Raw data - aggregate on the fly
        agg_df = (
            rlm_df.groupby("mode")
            .agg(
                turns_mean=("turns", "mean"),
                sub_llm_call_count_mean=("sub_llm_call_count", "mean"),
                sub_llm_mean_batch_size_mean=("sub_llm_mean_batch_size", "mean"),
            )
            .reset_index()
        )
    else:
        # Already aggregated data
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
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Count / Value")
    ax.set_title("RLM Usage Metrics by Mode")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_rlm_metrics_by_model(ax: plt.Axes, df: pd.DataFrame):
    """Plot: RLM-specific metrics comparison with model differentiation."""
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

    # Handle raw data vs aggregated data
    if "turns" in df.columns:
        # Raw data - aggregate on the fly
        rlm_df = (
            rlm_df.groupby(["model", "mode"])
            .agg(
                turns_mean=("turns", "mean"),
                sub_llm_call_count_mean=("sub_llm_call_count", "mean"),
                sub_llm_mean_batch_size_mean=("sub_llm_mean_batch_size", "mean"),
            )
            .reset_index()
        )

    models = list(rlm_df["model"].unique())
    modes = [m for m in ["rlm", "rlm_tips"] if m in rlm_df["mode"].unique()]

    # Metrics to show
    metrics = [
        ("turns_mean", "Turns"),
        ("sub_llm_call_count_mean", "Sub-LLM Calls"),
        ("sub_llm_mean_batch_size_mean", "Avg Batch Size"),
    ]

    n_models = len(models)
    n_modes = len(modes)
    n_metrics = len(metrics)

    hatches = HATCHES[:n_models]

    # Bar sizing
    bar_width = 0.1
    mode_gap = 0.05
    metric_gap = 0.4

    mode_group_width = n_models * bar_width
    metric_group_width = n_modes * mode_group_width + mode_gap

    for metric_idx, (metric_col, metric_label) in enumerate(metrics):
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
                        and pd.notna(model_data[metric_col].values[0])
                        else 0
                    )

                bar_x = (
                    metric_center
                    + mode_idx * (mode_group_width + mode_gap)
                    + model_idx * bar_width
                    - (metric_group_width - bar_width) / 2
                )

                ax.bar(
                    bar_x,
                    value,
                    bar_width,
                    color=style["color"],
                    edgecolor=EDGE_COLOR,
                    linewidth=0.5,
                    hatch=hatches[model_idx],
                )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Count / Value")
    ax.set_title("RLM Usage Metrics by Mode")

    tick_positions = [i * (metric_group_width + metric_gap) for i in range(n_metrics)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([m[1] for m in metrics])

    # Create custom legend
    legend_handles = []

    for mode in modes:
        style = MODE_STYLES.get(mode, {"color": "gray"})
        mode_label = MODE_LABELS.get(mode, mode).replace("\n", " ")
        legend_handles.append(
            Patch(
                facecolor=style["color"],
                edgecolor=EDGE_COLOR,
                linewidth=0.5,
                label=mode_label,
            )
        )

    for model_idx, model in enumerate(models):
        legend_handles.append(
            Patch(
                facecolor="white",
                edgecolor=EDGE_COLOR,
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


def plot_context_vs_reward(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, absolute: bool = False
):
    """Plot: Context length vs reward scatter, colored by mode."""
    # Determine column names based on data type (raw vs aggregated)
    if "context_length" in df.columns:
        context_col = "context_length"
        reward_col = "judge_reward"
    elif "context_length_mean" in df.columns:
        context_col = "context_length_mean"
        reward_col = "judge_reward_mean"
    else:
        ax.text(
            0.5,
            0.5,
            "Context length data not available",
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
        context_k = data[context_col] / 1000

        ax.scatter(
            context_k,
            data[reward_col],
            label=MODE_LABELS.get(mode, mode).replace("\n", " "),
            color=style["color"],
            marker=style["marker"],
            s=80,
            alpha=0.7,
            edgecolors=EDGE_COLOR,
            linewidth=0.5,
        )

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Reward vs Context Length")
    if absolute:
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color=MUTED_TEXT_COLOR, linestyle="--", alpha=0.3)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_reward_vs_context_scatter(
    ax: plt.Axes, df: pd.DataFrame, absolute: bool = False
):
    """Plot: Reward vs context length scatter for individual examples (raw data)."""
    if "context_length" not in df.columns:
        ax.text(
            0.5,
            0.5,
            "Context length not available\nin raw data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode].dropna(subset=["context_length", "judge_reward"])
        if len(data) == 0:
            continue

        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})

        # Convert to thousands of characters
        context_k = data["context_length"] / 1000

        ax.scatter(
            context_k,
            data["judge_reward"],
            label=MODE_LABELS.get(mode, mode).replace("\n", " "),
            color=style["color"],
            marker=style["marker"],
            s=40,
            alpha=0.5,
            edgecolors="none",
        )

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Judge Reward")
    ax.set_title("Reward vs Context Length (per-example)")
    if absolute:
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=1.0, color=MUTED_TEXT_COLOR, linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_reward_vs_context_binned(
    ax: plt.Axes,
    df: pd.DataFrame,
    num_bins: int = 10,
    absolute: bool = False,
    show_legend: bool = True,
):
    """Plot: Reward vs context length with binned means (raw data)."""
    if "context_length" not in df.columns:
        ax.text(
            0.5,
            0.5,
            "Context length not available\nin raw data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    # Get global bin edges based on all data
    all_context = df["context_length"].dropna()
    if len(all_context) == 0:
        ax.text(
            0.5,
            0.5,
            "No context length data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    bin_edges = np.linspace(all_context.min(), all_context.max(), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / 1000  # Convert to K

    for mode in modes:
        data = df[df["mode"] == mode].dropna(subset=["context_length", "judge_reward"])
        if len(data) == 0:
            continue

        style = MODE_STYLES.get(
            mode, {"color": "gray", "marker": "o", "linestyle": "-"}
        )

        # Compute binned means
        bin_means, _, _ = stats.binned_statistic(
            data["context_length"],
            data["judge_reward"],
            statistic="mean",
            bins=bin_edges,
        )

        # Plot binned means
        valid = ~np.isnan(bin_means)
        ax.plot(
            bin_centers[valid],
            bin_means[valid],
            label=MODE_LABELS.get(mode, mode).replace("\n", " "),
            color=style["color"],
            marker=style["marker"],
            linestyle="-",  # Use solid lines; colors and markers distinguish modes
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Judge Reward (mean)")
    ax.set_title(f"Reward vs Context Length (binned into {num_bins} ranges)")
    if absolute:
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color=MUTED_TEXT_COLOR, linestyle="--", alpha=0.3)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_reward_vs_context_binned_by_subset(
    df: pd.DataFrame,
    output_path: Path | None = None,
    num_bins: int = 10,
    absolute: bool = False,
):
    """Plot: Reward vs context length binned, faceted by subset."""
    if "context_length" not in df.columns:
        print("Context length not available in raw data")
        return

    subsets = [s for s in SUBSET_ORDER if s in df["subset"].unique()]
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    if len(subsets) == 0:
        print("No subsets found")
        return

    # Create subplot grid
    fig, axes = plt.subplots(
        1, len(subsets), figsize=(6 * len(subsets), 5), sharey=True
    )
    if len(subsets) == 1:
        axes = [axes]

    # Get global bin edges based on all data
    all_context = df["context_length"].dropna()
    if len(all_context) == 0:
        print("No context length data available")
        plt.close()
        return

    bin_edges = np.linspace(all_context.min(), all_context.max(), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / 1000  # Convert to K

    for ax, subset in zip(axes, subsets):
        subset_df = df[df["subset"] == subset]

        for mode in modes:
            data = subset_df[subset_df["mode"] == mode].dropna(
                subset=["context_length", "judge_reward"]
            )
            if len(data) == 0:
                continue

            style = MODE_STYLES.get(
                mode, {"color": "gray", "marker": "o", "linestyle": "-"}
            )

            # Compute binned means
            bin_means, _, _ = stats.binned_statistic(
                data["context_length"],
                data["judge_reward"],
                statistic="mean",
                bins=bin_edges,
            )

            valid = ~np.isnan(bin_means)
            ax.plot(
                bin_centers[valid],
                bin_means[valid],
                label=MODE_LABELS.get(mode, mode).replace("\n", " "),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Context Length (K chars)")
        ax.set_title(SUBSET_LABELS.get(subset, subset))
        if absolute:
            ax.set_ylim(0, 1.1)
            ax.axhline(y=1.0, color=MUTED_TEXT_COLOR, linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Judge Reward (mean)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    plt.suptitle(
        f"Oolong: Reward vs Context Length by Subset (binned into {num_bins} ranges)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()
    plt.close()


def plot_reward_vs_context_rolling(
    ax: plt.Axes, df: pd.DataFrame, window_frac: float = 0.1, absolute: bool = False
):
    """Plot: Reward vs context length with rolling mean (raw data)."""
    if "context_length" not in df.columns:
        ax.text(
            0.5,
            0.5,
            "Context length not available\nin raw data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode].dropna(subset=["context_length", "judge_reward"])
        if len(data) < 5:
            continue

        style = MODE_STYLES.get(mode, {"color": "gray", "linestyle": "-"})

        # Sort by context length
        data = data.sort_values("context_length")
        context_k = data["context_length"].values / 1000
        rewards = data["judge_reward"].values

        # Compute rolling mean
        window = max(5, int(len(data) * window_frac))
        rolling_mean = (
            pd.Series(rewards).rolling(window, center=True, min_periods=3).mean()
        )

        # Plot scatter (faded) and rolling mean
        ax.scatter(
            context_k, rewards, color=style["color"], alpha=0.2, s=20, edgecolors="none"
        )
        ax.plot(
            context_k,
            rolling_mean,
            label=MODE_LABELS.get(mode, mode).replace("\n", " "),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.5,
        )

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Judge Reward (rolling mean)")
    ax.set_title("Reward vs Context Length (rolling average)")
    if absolute:
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color=MUTED_TEXT_COLOR, linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Timing comparison across modes and models."""
    # Handle both raw data (total_ms) and aggregated data (total_ms_mean)
    if "total_ms" in df.columns:
        # Raw data - aggregate on the fly
        agg_df = (
            df.groupby(["model", "mode"])
            .agg(
                total_ms_mean=("total_ms", "mean"),
                total_ms_count=("total_ms", "count"),
            )
            .reset_index()
        )
    else:
        # Already aggregated data
        agg_dict = {"total_ms_mean": "mean"}
        if "total_ms_count" in df.columns:
            agg_dict["total_ms_count"] = "sum"
        agg_df = df.groupby(["model", "mode"]).agg(agg_dict).reset_index()

    models = agg_df["model"].unique()
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        times = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0 and pd.notna(model_data["total_ms_mean"].values[0]):
                times.append(model_data["total_ms_mean"].values[0] / 1000)
                if "total_ms_count" in model_data.columns:
                    counts.append(int(model_data["total_ms_count"].values[0]))
                else:
                    counts.append(None)
            else:
                times.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            [xi + offset for xi in x],
            times,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, time_val, count in zip(bars, times, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{time_val:.1f}s")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=MUTED_TEXT_COLOR,
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Average Rollout Time by Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_turns(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Number of turns/iterations by mode and model."""
    # Handle both raw data (turns) and aggregated data (turns_mean)
    if "turns" in df.columns:
        # Raw data - aggregate on the fly
        agg_df = (
            df.groupby(["model", "mode"])
            .agg(
                turns_mean=("turns", "mean"),
                turns_count=("turns", "count"),
            )
            .reset_index()
        )
    else:
        # Already aggregated data
        agg_dict = {"turns_mean": "mean"}
        if "turns_count" in df.columns:
            agg_dict["turns_count"] = "sum"
        agg_df = df.groupby(["model", "mode"]).agg(agg_dict).reset_index()

    models = agg_df["model"].unique()
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        turns = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0 and pd.notna(model_data["turns_mean"].values[0]):
                turns.append(model_data["turns_mean"].values[0])
                if "turns_count" in model_data.columns:
                    counts.append(int(model_data["turns_count"].values[0]))
                else:
                    counts.append(None)
            else:
                turns.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            [xi + offset for xi in x],
            turns,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, turn_val, count in zip(bars, turns, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{turn_val:.1f}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=MUTED_TEXT_COLOR,
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Turns")
    ax.set_title("Average Turns by Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_by_subset(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Timing comparison across subsets (grouped bar chart)."""
    # Handle both raw data (total_ms) and aggregated data (total_ms_mean)
    if "total_ms" in df.columns:
        # Raw data - aggregate on the fly
        agg_df = (
            df.groupby(["subset", "mode"])
            .agg(
                total_ms_mean=("total_ms", "mean"),
                total_ms_count=("total_ms", "count"),
            )
            .reset_index()
        )
    else:
        # Already aggregated data
        agg_dict = {"total_ms_mean": "mean"}
        if "total_ms_count" in df.columns:
            agg_dict["total_ms_count"] = "sum"
        agg_df = df.groupby(["subset", "mode"]).agg(agg_dict).reset_index()

    subsets = [s for s in SUBSET_ORDER if s in agg_df["subset"].unique()]
    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(subsets))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = agg_df[agg_df["mode"] == mode]
        times = []
        counts = []
        for subset in subsets:
            subset_data = mode_data[mode_data["subset"] == subset]
            if len(subset_data) > 0 and pd.notna(
                subset_data["total_ms_mean"].values[0]
            ):
                times.append(subset_data["total_ms_mean"].values[0] / 1000)
                if "total_ms_count" in subset_data.columns:
                    counts.append(int(subset_data["total_ms_count"].values[0]))
                else:
                    counts.append(None)
            else:
                times.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            [xi + offset for xi in x],
            times,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, time_val, count in zip(bars, times, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{time_val:.1f}s")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=MUTED_TEXT_COLOR,
                    )

    ax.set_xlabel("Subset")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing by Subset")
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS.get(s, s) for s in subsets])
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_vs_context(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Timing vs context length scatter, colored by mode."""
    # Determine column names based on data type (raw vs aggregated)
    if "context_length" in df.columns:
        context_col = "context_length"
        time_col = "total_ms"
    elif "context_length_mean" in df.columns:
        context_col = "context_length_mean"
        time_col = "total_ms_mean"
    else:
        ax.text(
            0.5,
            0.5,
            "Context length data not available",
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
        valid_data = data.dropna(subset=[context_col, time_col])

        if len(valid_data) > 0:
            # Convert context length to thousands
            context_k = valid_data[context_col] / 1000
            # Convert timing to seconds
            times = valid_data[time_col] / 1000

            ax.scatter(
                context_k,
                times,
                label=MODE_LABELS.get(mode, mode).replace("\n", " "),
                color=style["color"],
                marker=style["marker"],
                s=80,
                alpha=0.7,
                edgecolors=EDGE_COLOR,
                linewidth=0.5,
            )

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing vs Context Length")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_efficiency(ax: plt.Axes, df: pd.DataFrame, absolute: bool = False):
    """Plot: Reward vs timing scatter (cost-benefit analysis)."""
    # Determine column names based on data type (raw vs aggregated)
    if "total_ms" in df.columns:
        time_col = "total_ms"
        reward_col = "judge_reward"
    else:
        time_col = "total_ms_mean"
        reward_col = "judge_reward_mean"

    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode]
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})

        # Filter out rows with missing data
        valid_data = data.dropna(subset=[time_col, reward_col])

        if len(valid_data) > 0:
            # Convert to seconds
            times = valid_data[time_col] / 1000

            ax.scatter(
                times,
                valid_data[reward_col],
                label=MODE_LABELS.get(mode, mode).replace("\n", " "),
                color=style["color"],
                marker=style["marker"],
                s=60,
                alpha=0.7,
                edgecolors=EDGE_COLOR,
                linewidth=0.5,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Timing Efficiency: Reward vs Time")
    if absolute:
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color=MUTED_TEXT_COLOR, linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_token_usage(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
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

    # Handle raw data vs aggregated data
    if "prompt_tokens" in df.columns:
        # Raw data - aggregate on the fly
        agg_cols = {
            "prompt_tokens_mean": ("prompt_tokens", "mean"),
            "completion_tokens_mean": ("completion_tokens", "mean"),
            "prompt_tokens_count": ("prompt_tokens", "count"),
        }
        if "sub_llm_prompt_tokens" in df.columns:
            agg_cols["sub_llm_prompt_tokens_mean"] = ("sub_llm_prompt_tokens", "mean")
        if "sub_llm_completion_tokens" in df.columns:
            agg_cols["sub_llm_completion_tokens_mean"] = (
                "sub_llm_completion_tokens",
                "mean",
            )
        df = df.groupby(["model", "mode"]).agg(**agg_cols).reset_index()

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
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, tokens, count in zip(bars, total_tokens, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{int(tokens):,}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + bar.get_height() * 0.02,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=MUTED_TEXT_COLOR,
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


def plot_main_model_tokens(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
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

    # Handle raw data vs aggregated data
    if "prompt_tokens" in df.columns:
        # Raw data - aggregate on the fly
        df = (
            df.groupby(["model", "mode"])
            .agg(
                prompt_tokens_mean=("prompt_tokens", "mean"),
                completion_tokens_mean=("completion_tokens", "mean"),
                prompt_tokens_count=("prompt_tokens", "count"),
            )
            .reset_index()
        )

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
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, tokens, count in zip(bars, total_tokens, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{int(tokens):,}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + bar.get_height() * 0.02,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=MUTED_TEXT_COLOR,
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


def plot_prompt_tokens(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Prompt token usage (main + sub-LLM stacked), split by model."""
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

    # Handle raw data vs aggregated data
    if "prompt_tokens" in df.columns:
        agg_cols = {
            "prompt_tokens_mean": ("prompt_tokens", "mean"),
            "prompt_tokens_count": ("prompt_tokens", "count"),
        }
        if "sub_llm_prompt_tokens" in df.columns:
            agg_cols["sub_llm_prompt_tokens_mean"] = ("sub_llm_prompt_tokens", "mean")
        df = df.groupby(["model", "mode"]).agg(**agg_cols).reset_index()

    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        main_tokens = []
        sub_tokens = []
        counts = []

        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                main_tokens.append(model_data["prompt_tokens_mean"].values[0] or 0)
                sub_val = 0
                if "sub_llm_prompt_tokens_mean" in model_data.columns:
                    val = model_data["sub_llm_prompt_tokens_mean"].values[0]
                    sub_val = 0 if pd.isna(val) else val
                sub_tokens.append(sub_val)
                if "prompt_tokens_count" in model_data.columns:
                    counts.append(int(model_data["prompt_tokens_count"].values[0]))
                else:
                    counts.append(None)
            else:
                main_tokens.append(0)
                sub_tokens.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})

        bars = ax.bar(
            [xi + offset for xi in x],
            main_tokens,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )
        ax.bar(
            [xi + offset for xi in x],
            sub_tokens,
            width,
            bottom=main_tokens,
            color=style["color"],
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
        )

        if show_values or show_counts:
            for bar, main, sub, count in zip(bars, main_tokens, sub_tokens, counts):
                total = main + sub
                annotations = []
                if show_values:
                    annotations.append(f"{int(total):,}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        total + 100,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Prompt Tokens")
    ax.set_title("Prompt Token Usage")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right", fontsize=8
    )

    # Add in-subplot legend for Main Model / Sub-LLM distinction
    pattern_handles = [
        Patch(
            facecolor="#CCCCCC", edgecolor=EDGE_COLOR, linewidth=0.5, label="Main Model"
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
            label="Sub-LLM",
        ),
    ]
    ax.legend(handles=pattern_handles, loc="upper left", fontsize=8)


def plot_completion_tokens(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Completion token usage (main + sub-LLM stacked), split by model."""
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

    # Handle raw data vs aggregated data
    if "completion_tokens" in df.columns:
        agg_cols = {
            "completion_tokens_mean": ("completion_tokens", "mean"),
            "completion_tokens_count": ("completion_tokens", "count"),
        }
        if "sub_llm_completion_tokens" in df.columns:
            agg_cols["sub_llm_completion_tokens_mean"] = (
                "sub_llm_completion_tokens",
                "mean",
            )
        df = df.groupby(["model", "mode"]).agg(**agg_cols).reset_index()

    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        main_tokens = []
        sub_tokens = []
        counts = []

        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                main_tokens.append(model_data["completion_tokens_mean"].values[0] or 0)
                sub_val = 0
                if "sub_llm_completion_tokens_mean" in model_data.columns:
                    val = model_data["sub_llm_completion_tokens_mean"].values[0]
                    sub_val = 0 if pd.isna(val) else val
                sub_tokens.append(sub_val)
                if "completion_tokens_count" in model_data.columns:
                    counts.append(int(model_data["completion_tokens_count"].values[0]))
                else:
                    counts.append(None)
            else:
                main_tokens.append(0)
                sub_tokens.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})

        bars = ax.bar(
            [xi + offset for xi in x],
            main_tokens,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
        )
        ax.bar(
            [xi + offset for xi in x],
            sub_tokens,
            width,
            bottom=main_tokens,
            color=style["color"],
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
        )

        if show_values or show_counts:
            for bar, main, sub, count in zip(bars, main_tokens, sub_tokens, counts):
                total = main + sub
                annotations = []
                if show_values:
                    annotations.append(f"{int(total):,}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        total + 100,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Completion Tokens")
    ax.set_title("Completion Token Usage")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right", fontsize=8
    )

    # Add in-subplot legend for Main Model / Sub-LLM distinction
    pattern_handles = [
        Patch(
            facecolor="#CCCCCC", edgecolor=EDGE_COLOR, linewidth=0.5, label="Main Model"
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
            label="Sub-LLM",
        ),
    ]
    ax.legend(handles=pattern_handles, loc="upper left", fontsize=8)


def plot_heatmap(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Heatmap of reward by mode × subset."""
    # Determine column name based on data type (raw vs aggregated)
    reward_col = "judge_reward" if "judge_reward" in df.columns else "judge_reward_mean"

    # Create pivot table: mode × subset
    pivot = df.pivot_table(
        values=reward_col,
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


def create_tokens_by_subset_grid(
    df: pd.DataFrame,
    output_path: Path | None = None,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Create a 2x3 grid showing token usage by subset.

    Top row: Prompt tokens for each subset (synth, synth_with_labels, real)
    Bottom row: Completion tokens for each subset
    """
    subsets = [s for s in SUBSET_ORDER if s in df["subset"].unique()]

    if len(subsets) == 0:
        print("No subsets found in data")
        return

    fig, axes = plt.subplots(2, len(subsets), figsize=(6 * len(subsets), 10))
    if len(subsets) == 1:
        axes = axes.reshape(2, 1)

    # Handle raw data vs aggregated data
    if "prompt_tokens" in df.columns:
        agg_cols = {
            "prompt_tokens_mean": ("prompt_tokens", "mean"),
            "completion_tokens_mean": ("completion_tokens", "mean"),
            "prompt_tokens_count": ("prompt_tokens", "count"),
        }
        if "sub_llm_prompt_tokens" in df.columns:
            agg_cols["sub_llm_prompt_tokens_mean"] = ("sub_llm_prompt_tokens", "mean")
        if "sub_llm_completion_tokens" in df.columns:
            agg_cols["sub_llm_completion_tokens_mean"] = (
                "sub_llm_completion_tokens",
                "mean",
            )
        df = df.groupby(["model", "mode", "subset"]).agg(**agg_cols).reset_index()

    # Filter out excluded models (partial match)
    all_models = df["model"].unique()
    models = [
        m
        for m in all_models
        if not any(excl.lower() in m.lower() for excl in EXCLUDED_MODELS)
    ]
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for col_idx, subset in enumerate(subsets):
        subset_df = df[df["subset"] == subset]

        # Top row: Prompt tokens
        ax_prompt = axes[0, col_idx]
        for i, mode in enumerate(modes):
            mode_data = subset_df[subset_df["mode"] == mode]
            main_tokens = []
            sub_tokens = []

            for model in models:
                model_data = mode_data[mode_data["model"] == model]
                if len(model_data) > 0:
                    main_tokens.append(model_data["prompt_tokens_mean"].values[0] or 0)
                    sub_val = 0
                    if "sub_llm_prompt_tokens_mean" in model_data.columns:
                        val = model_data["sub_llm_prompt_tokens_mean"].values[0]
                        sub_val = 0 if pd.isna(val) else val
                    sub_tokens.append(sub_val)
                else:
                    main_tokens.append(0)
                    sub_tokens.append(0)

            offset = (i - len(modes) / 2 + 0.5) * width
            style = MODE_STYLES.get(mode, {"color": "gray"})

            ax_prompt.bar(
                [xi + offset for xi in x],
                main_tokens,
                width,
                color=style["color"],
                edgecolor=EDGE_COLOR,
                linewidth=0.5,
            )
            ax_prompt.bar(
                [xi + offset for xi in x],
                sub_tokens,
                width,
                bottom=main_tokens,
                color=style["color"],
                edgecolor=EDGE_COLOR,
                linewidth=0.5,
                hatch="///",
                alpha=0.6,
            )

        ax_prompt.set_xlabel("Model")
        ax_prompt.set_ylabel("Prompt Tokens")
        ax_prompt.set_title(f"Prompt Tokens - {SUBSET_LABELS.get(subset, subset)}")
        ax_prompt.set_xticks(list(x))
        ax_prompt.set_xticklabels(
            [normalize_model_name(m) for m in models],
            rotation=15,
            ha="right",
            fontsize=8,
        )

        # Bottom row: Completion tokens
        ax_completion = axes[1, col_idx]
        for i, mode in enumerate(modes):
            mode_data = subset_df[subset_df["mode"] == mode]
            main_tokens = []
            sub_tokens = []

            for model in models:
                model_data = mode_data[mode_data["model"] == model]
                if len(model_data) > 0:
                    main_tokens.append(
                        model_data["completion_tokens_mean"].values[0] or 0
                    )
                    sub_val = 0
                    if "sub_llm_completion_tokens_mean" in model_data.columns:
                        val = model_data["sub_llm_completion_tokens_mean"].values[0]
                        sub_val = 0 if pd.isna(val) else val
                    sub_tokens.append(sub_val)
                else:
                    main_tokens.append(0)
                    sub_tokens.append(0)

            offset = (i - len(modes) / 2 + 0.5) * width
            style = MODE_STYLES.get(mode, {"color": "gray"})

            ax_completion.bar(
                [xi + offset for xi in x],
                main_tokens,
                width,
                color=style["color"],
                edgecolor=EDGE_COLOR,
                linewidth=0.5,
            )
            ax_completion.bar(
                [xi + offset for xi in x],
                sub_tokens,
                width,
                bottom=main_tokens,
                color=style["color"],
                edgecolor=EDGE_COLOR,
                linewidth=0.5,
                hatch="///",
                alpha=0.6,
            )

        ax_completion.set_xlabel("Model")
        ax_completion.set_ylabel("Completion Tokens")
        ax_completion.set_title(
            f"Completion Tokens - {SUBSET_LABELS.get(subset, subset)}"
        )
        ax_completion.set_xticks(list(x))
        ax_completion.set_xticklabels(
            [normalize_model_name(m) for m in models],
            rotation=15,
            ha="right",
            fontsize=8,
        )

    # Add in-subplot legend for Main Model / Sub-LLM distinction (top-left subplot only)
    pattern_handles = [
        Patch(
            facecolor="#CCCCCC", edgecolor=EDGE_COLOR, linewidth=0.5, label="Main Model"
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor=EDGE_COLOR,
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
            label="Sub-LLM",
        ),
    ]
    axes[0, 0].legend(handles=pattern_handles, loc="upper left", fontsize=8)

    # Create central legend for modes
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
                markeredgecolor=EDGE_COLOR,
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
        "Oolong: Token Usage by Subset",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def create_plots(
    df: pd.DataFrame,
    output_path: Path | None = None,
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Create the 2x3 grid of plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: Reward-related plots
    plot_reward_by_model(
        axes[0, 0],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
        absolute=absolute,
    )
    plot_reward_by_subset(
        axes[0, 1],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
        absolute=absolute,
    )
    plot_reward_vs_context_binned(axes[0, 2], df, absolute=absolute, show_legend=False)

    # Bottom row: Usage-related plots
    plot_turns(
        axes[1, 0],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )
    plot_prompt_tokens(
        axes[1, 1],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )
    plot_completion_tokens(
        axes[1, 2],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )

    # Create central legend for modes (using markers to match scatter plots)
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
                markeredgecolor=EDGE_COLOR,
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
        "Oolong",
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
    "reward": (plot_reward_by_model, (10, 7), "Reward"),
    "subset": (plot_reward_by_subset, (10, 7), "Reward by Subset"),
    "heatmap": (plot_heatmap, (10, 7), "Reward Heatmap"),
    "rlm_metrics": (plot_rlm_metrics, (10, 7), "RLM Usage Metrics"),
    "rlm_metrics_by_model": (
        plot_rlm_metrics_by_model,
        (12, 7),
        "RLM Usage Metrics (by model)",
    ),
    "turns": (plot_turns, (10, 7), "Turns"),
    "timing": (plot_timing, (10, 7), "Timing Comparison"),
    "context": (plot_context_vs_reward, (10, 7), "Reward vs Context Length"),
    "main_tokens": (plot_main_model_tokens, (10, 7), "Main Model Token Usage"),
    "tokens": (plot_token_usage, (10, 7), "Total Token Usage"),
    "prompt_tokens": (plot_prompt_tokens, (10, 7), "Prompt Token Usage"),
    "completion_tokens": (plot_completion_tokens, (10, 7), "Completion Token Usage"),
    # Additional timing plots
    "timing_by_subset": (plot_timing_by_subset, (10, 7), "Timing by Subset"),
    "timing_vs_context": (plot_timing_vs_context, (10, 7), "Timing vs Context Length"),
    "timing_efficiency": (plot_timing_efficiency, (10, 7), "Timing Efficiency"),
    # Raw data plots (require --raw-input)
    "context_scatter": (
        plot_reward_vs_context_scatter,
        (10, 7),
        "Reward vs Context (scatter)",
    ),
    "context_binned": (
        plot_reward_vs_context_binned,
        (10, 7),
        "Reward vs Context (binned)",
    ),
    "context_rolling": (
        plot_reward_vs_context_rolling,
        (10, 7),
        "Reward vs Context (rolling)",
    ),
}

# Plot types that require raw (per-example) data instead of aggregated data
RAW_DATA_PLOTS = {
    "context_scatter",
    "context_binned",
    "context_binned_by_subset",
    "context_rolling",
}

# Plot types that manage their own figure (not single-axis plots)
MULTI_AXIS_PLOTS = {"context_binned_by_subset", "tokens_by_subset"}


def create_single_plot(
    plot_name: str,
    df: pd.DataFrame,
    output_path: Path | None = None,
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Create a single standalone plot."""
    if plot_name not in PLOT_REGISTRY:
        raise ValueError(
            f"Unknown plot: {plot_name}. Available: {list(PLOT_REGISTRY.keys())}"
        )

    func, figsize, title = PLOT_REGISTRY[plot_name]

    fig, ax = plt.subplots(figsize=figsize)

    # Plots that support show_counts, show_values, and absolute
    if plot_name in ("reward", "subset"):
        func(
            ax,
            df,
            show_counts=show_counts,
            show_values=show_values,
            absolute=absolute,
        )
    # Plots that support show_counts and show_values
    elif plot_name in (
        "timing",
        "timing_by_subset",
        "main_tokens",
        "tokens",
        "prompt_tokens",
        "completion_tokens",
    ):
        func(ax, df, show_counts=show_counts, show_values=show_values)
    # Plots that support only absolute
    elif plot_name in (
        "context",
        "context_scatter",
        "context_binned",
        "context_rolling",
        "timing_efficiency",
    ):
        func(ax, df, absolute=absolute)
    else:
        func(ax, df)

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
    
    # Show context length analysis (aggregated)
    python plot_results.py --image context
    
    # Save plot to file
    python plot_results.py --image reward -o reward_plot.png
    
    # Additional timing plots
    python plot_results.py --image timing_by_subset    # Timing by subset
    python plot_results.py --image timing_vs_context   # Timing vs context length
    python plot_results.py --image timing_efficiency   # Reward vs time tradeoff
    
    # Raw data context plots (per-example granularity)
    # First generate raw data: python aggregate_results.py --raw-output outputs/raw_results.csv
    python plot_results.py --image context_scatter            # Individual scatter points
    python plot_results.py --image context_binned             # Binned means
    python plot_results.py --image context_binned_by_subset   # Binned means, faceted by subset
    python plot_results.py --image context_rolling            # Rolling average curves
    
    # Use custom raw data file
    python plot_results.py --image context_binned -r my_raw_data.csv
    
    # Show sample counts on bars
    python plot_results.py --show-counts
    
    # Show bar values (e.g., 0.85, 12.3s, 1,234)
    python plot_results.py --show-values
    python plot_results.py -v --image reward
    
    # Show both values and counts
    python plot_results.py -v -c
    
    # Use fixed 0-1 y-axis for reward plots
    python plot_results.py --absolute
    python plot_results.py --image reward -a
""",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(__file__).parent / "outputs" / "raw_results.csv",
        help="Path to raw results CSV file (use raw_results.csv for context plots)",
    )
    parser.add_argument(
        "--raw-input",
        "-r",
        type=Path,
        default=None,
        help="Path to raw results CSV (for context_scatter/binned/rolling plots). "
        "If not specified, defaults to outputs/raw_results.csv",
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
        "-I",
        choices=[
            "main",
            "reward",
            "subset",
            "heatmap",
            "rlm_metrics",
            "rlm_metrics_by_model",
            "turns",
            "timing",
            "context",
            "main_tokens",
            "tokens",
            "prompt_tokens",
            "completion_tokens",
            # Additional timing plots
            "timing_by_subset",
            "timing_vs_context",
            "timing_efficiency",
            # Raw data plots (require --raw-input)
            "context_scatter",
            "context_binned",
            "context_binned_by_subset",
            "context_rolling",
            # Multi-subplot grid plots
            "tokens_by_subset",
        ],
        default="main",
        help="Which plot to generate: 'main' for 2x3 grid, or individual plot name. "
        "context_scatter/binned/rolling require raw data (--raw-input or default path)",
    )

    # Model filtering options
    parser.add_argument(
        "--model",
        "-m",
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

    # Subset filtering options
    parser.add_argument(
        "--subset",
        "-S",
        type=str,
        default=None,
        choices=["synth", "synth_with_labels", "real"],
        help="Filter results to a specific subset",
    )

    parser.add_argument(
        "--show-counts",
        "-c",
        action="store_true",
        help="Show sample counts (n=X) above bars in bar chart plots",
    )
    parser.add_argument(
        "--show-values",
        "-v",
        action="store_true",
        help="Show bar values above bars in bar chart plots",
    )
    parser.add_argument(
        "--absolute",
        "-a",
        action="store_true",
        help="Use fixed 0-1 y-axis range for reward plots (default: auto-scale)",
    )
    parser.add_argument(
        "--include-updated-tips",
        "-u",
        action="store_true",
        help="Include updated tips (rlm_tips_v2) mode in plots",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Use light theme instead of dark (default)",
    )

    args = parser.parse_args()

    if args.light:
        apply_light_style()
    else:
        apply_dark_style()

    # Check if we need raw data for this plot type
    needs_raw_data = args.image in RAW_DATA_PLOTS

    if needs_raw_data:
        # Use raw data file
        if args.raw_input is None:
            args.raw_input = Path(__file__).parent / "outputs" / "raw_results.csv"

        if not args.raw_input.exists():
            print(f"Error: Raw data file not found: {args.raw_input}")
            print(
                "Run: python aggregate_results.py --raw-output outputs/raw_results.csv"
            )
            return

        df = load_raw_data(args.raw_input)
        print(f"Loaded {len(df)} raw results from {args.raw_input}")
    else:
        # Use aggregated data
        if not args.input.exists():
            print(f"Error: Input file not found: {args.input}")
            print("Run aggregate_results.py first to generate the CSV.")
            return

        df = load_data(args.input)

    # Filter out rlm_tips_v2 unless explicitly requested
    if not args.include_updated_tips and "mode" in df.columns:
        original_count = len(df)
        df = df[df["mode"] != "rlm_tips_v2"]
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            print(
                f"Filtered out {filtered_count} rlm_tips_v2 configurations (use -u to include)"
            )

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

    if not needs_raw_data:
        print(f"Loaded {len(df)} configurations from {args.input}")

    if args.image == "main":
        create_plots(
            df,
            args.output,
            show_counts=args.show_counts,
            show_values=args.show_values,
            absolute=args.absolute,
        )
    elif args.image in MULTI_AXIS_PLOTS:
        # These plots manage their own figure
        if args.image == "context_binned_by_subset":
            plot_reward_vs_context_binned_by_subset(
                df, args.output, absolute=args.absolute
            )
        elif args.image == "tokens_by_subset":
            create_tokens_by_subset_grid(
                df,
                args.output,
                show_counts=args.show_counts,
                show_values=args.show_values,
            )
    else:
        create_single_plot(
            args.image,
            df,
            args.output,
            show_counts=args.show_counts,
            show_values=args.show_values,
            absolute=args.absolute,
        )


if __name__ == "__main__":
    main()

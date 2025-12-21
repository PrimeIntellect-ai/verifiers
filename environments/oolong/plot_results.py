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

SUBSET_ORDER = ["synth", "synth_with_labels", "real"]
SUBSET_LABELS = {
    "synth": "Synthetic",
    "synth_with_labels": "Synth+Labels",
    "real": "Real",
}

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
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Mode comparison across models (grouped bar chart), aggregated across subsets."""
    # Aggregate across subsets for each model/mode combination
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

    ax.set_xlabel("Model")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Reward")
    ax.set_ylim(0, 1.2 if show_counts else 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_reward_by_subset(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Mode comparison across subsets (grouped bar chart), aggregated across models."""
    # Aggregate across models for each subset/mode combination
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

    ax.set_xlabel("Subset")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Reward by Subset")
    ax.set_ylim(0, 1.2 if show_counts else 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS.get(s, s) for s in subsets])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
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

    # Aggregate across models and subsets
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
                    edgecolor="black",
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
                edgecolor="black",
                linewidth=0.5,
                label=mode_label,
            )
        )

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


def plot_context_vs_reward(ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True):
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

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Judge Reward (Accuracy)")
    ax.set_title("Reward vs Context Length")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_reward_vs_context_scatter(ax: plt.Axes, df: pd.DataFrame):
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
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_reward_vs_context_binned(ax: plt.Axes, df: pd.DataFrame, num_bins: int = 10):
    """Plot: Reward vs context length with binned means and error bars (raw data)."""
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

        # Compute binned statistics
        bin_means, _, _ = stats.binned_statistic(
            data["context_length"],
            data["judge_reward"],
            statistic="mean",
            bins=bin_edges,
        )
        bin_stds, _, _ = stats.binned_statistic(
            data["context_length"],
            data["judge_reward"],
            statistic="std",
            bins=bin_edges,
        )
        bin_counts, _, _ = stats.binned_statistic(
            data["context_length"],
            data["judge_reward"],
            statistic="count",
            bins=bin_edges,
        )

        # Compute standard error
        bin_sems = bin_stds / np.sqrt(bin_counts)
        bin_sems = np.nan_to_num(bin_sems, nan=0)

        # Plot with error bars
        valid = ~np.isnan(bin_means)
        ax.errorbar(
            bin_centers[valid],
            bin_means[valid],
            yerr=bin_sems[valid],
            label=MODE_LABELS.get(mode, mode).replace("\n", " "),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=8,
            capsize=3,
        )

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Judge Reward (mean ± SEM)")
    ax.set_title(f"Reward vs Context Length (binned into {num_bins} ranges)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_reward_vs_context_rolling(
    ax: plt.Axes, df: pd.DataFrame, window_frac: float = 0.1
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
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Timing comparison across modes and models."""
    # Aggregate across subsets
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

    ax.set_xlabel("Model")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Average Rollout Time by Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_by_subset(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Timing comparison across subsets (grouped bar chart)."""
    # Aggregate across models for each subset/mode combination
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

    ax.set_xlabel("Subset")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing by Subset")
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS.get(s, s) for s in subsets])
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


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

    ax.set_xlabel("Context Length (K chars)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing vs Context Length")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


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
    ax.set_title("Timing Efficiency: Reward vs Time")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_token_usage(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Total token usage comparison across all modes."""
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

    # Aggregate across models and subsets
    agg_dict = {
        "prompt_tokens_mean": "mean",
        "completion_tokens_mean": "mean",
    }
    if "prompt_tokens_count" in df.columns:
        agg_dict["prompt_tokens_count"] = "sum"

    agg_df = df.groupby("mode").agg(agg_dict).reset_index()

    # Add sub-LLM tokens if available (for RLM modes)
    if "sub_llm_prompt_tokens_mean" in df.columns:
        sub_agg = (
            df.groupby("mode")
            .agg(
                {
                    "sub_llm_prompt_tokens_mean": "mean",
                    "sub_llm_completion_tokens_mean": "mean",
                }
            )
            .reset_index()
        )
        agg_df = agg_df.merge(sub_agg, on="mode", how="left")

    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(modes))
    width = 0.6

    total_tokens = []
    counts = []
    colors = []
    for mode in modes:
        mode_data = agg_df[agg_df["mode"] == mode]
        if len(mode_data) > 0:
            main_prompt = mode_data["prompt_tokens_mean"].values[0] or 0
            main_completion = mode_data["completion_tokens_mean"].values[0] or 0
            sub_prompt = 0
            sub_completion = 0
            if "sub_llm_prompt_tokens_mean" in mode_data.columns:
                val = mode_data["sub_llm_prompt_tokens_mean"].values[0]
                sub_prompt = 0 if pd.isna(val) else val
            if "sub_llm_completion_tokens_mean" in mode_data.columns:
                val = mode_data["sub_llm_completion_tokens_mean"].values[0]
                sub_completion = 0 if pd.isna(val) else val
            total_tokens.append(
                main_prompt + main_completion + sub_prompt + sub_completion
            )
            if "prompt_tokens_count" in mode_data.columns:
                counts.append(int(mode_data["prompt_tokens_count"].values[0]))
            else:
                counts.append(None)
        else:
            total_tokens.append(0)
            counts.append(None)
        colors.append(MODE_STYLES.get(mode, {"color": "gray"})["color"])

    bars = ax.bar(
        x,
        total_tokens,
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
                    bar.get_height() + bar.get_height() * 0.02,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Total Tokens")
    ax.set_title("Total Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes])


def plot_main_model_tokens(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, show_counts: bool = False
):
    """Plot: Main model token usage comparison (excludes sub-LLM tokens)."""
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

    # Aggregate across models and subsets
    agg_dict = {
        "prompt_tokens_mean": "mean",
        "completion_tokens_mean": "mean",
    }
    if "prompt_tokens_count" in df.columns:
        agg_dict["prompt_tokens_count"] = "sum"

    agg_df = df.groupby("mode").agg(agg_dict).reset_index()

    modes = [m for m in MODE_ORDER if m in agg_df["mode"].unique()]

    x = range(len(modes))
    width = 0.6

    total_tokens = []
    counts = []
    colors = []
    for mode in modes:
        mode_data = agg_df[agg_df["mode"] == mode]
        if len(mode_data) > 0:
            main_prompt = mode_data["prompt_tokens_mean"].values[0] or 0
            main_completion = mode_data["completion_tokens_mean"].values[0] or 0
            total_tokens.append(main_prompt + main_completion)
            if "prompt_tokens_count" in mode_data.columns:
                counts.append(int(mode_data["prompt_tokens_count"].values[0]))
            else:
                counts.append(None)
        else:
            total_tokens.append(0)
            counts.append(None)
        colors.append(MODE_STYLES.get(mode, {"color": "gray"})["color"])

    bars = ax.bar(
        x,
        total_tokens,
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
                    bar.get_height() + bar.get_height() * 0.02,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Tokens (Main Model Only)")
    ax.set_title("Main Model Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes])


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


def create_plots(
    df: pd.DataFrame, output_path: Path | None = None, show_counts: bool = False
):
    """Create the 2x3 grid of plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Main plots (suppress individual legends)
    # Left column: same as deepdive
    plot_reward_by_model(axes[0, 0], df, show_legend=False, show_counts=show_counts)
    plot_timing(axes[0, 1], df, show_legend=False, show_counts=show_counts)
    plot_main_model_tokens(axes[1, 0], df, show_legend=False, show_counts=show_counts)
    plot_token_usage(axes[1, 1], df, show_legend=False, show_counts=show_counts)

    # Right column: oolong-specific
    plot_reward_by_subset(axes[0, 2], df, show_legend=False, show_counts=show_counts)
    plot_context_vs_reward(axes[1, 2], df, show_legend=False)

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
    "timing": (plot_timing, (10, 7), "Timing Comparison"),
    "context": (plot_context_vs_reward, (10, 7), "Reward vs Context Length"),
    "main_tokens": (plot_main_model_tokens, (10, 7), "Main Model Token Usage"),
    "tokens": (plot_token_usage, (10, 7), "Total Token Usage"),
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
RAW_DATA_PLOTS = {"context_scatter", "context_binned", "context_rolling"}


def create_single_plot(
    plot_name: str,
    df: pd.DataFrame,
    output_path: Path | None = None,
    show_counts: bool = False,
):
    """Create a single standalone plot."""
    if plot_name not in PLOT_REGISTRY:
        raise ValueError(
            f"Unknown plot: {plot_name}. Available: {list(PLOT_REGISTRY.keys())}"
        )

    func, figsize, title = PLOT_REGISTRY[plot_name]

    fig, ax = plt.subplots(figsize=figsize)

    # Pass show_counts to functions that support it
    if plot_name in (
        "reward",
        "subset",
        "timing",
        "timing_by_subset",
        "main_tokens",
        "tokens",
    ):
        func(ax, df, show_counts=show_counts)
    else:
        func(ax, df)

    plt.suptitle(f"Oolong: {title}", fontsize=14, fontweight="bold", y=1.02)
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
    python plot_results.py --image context_scatter     # Individual scatter points
    python plot_results.py --image context_binned      # Binned means with error bars
    python plot_results.py --image context_rolling     # Rolling average curves
    
    # Use custom raw data file
    python plot_results.py --image context_binned -r my_raw_data.csv
    
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
        choices=[
            "main",
            "reward",
            "subset",
            "heatmap",
            "rlm_metrics",
            "rlm_metrics_by_model",
            "timing",
            "context",
            "main_tokens",
            "tokens",
            # Additional timing plots
            "timing_by_subset",
            "timing_vs_context",
            "timing_efficiency",
            # Raw data plots (require --raw-input)
            "context_scatter",
            "context_binned",
            "context_rolling",
        ],
        default="main",
        help="Which plot to generate: 'main' for 2x3 grid, or individual plot name. "
        "context_scatter/binned/rolling require raw data (--raw-input or default path)",
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

    parser.add_argument(
        "--show-counts",
        "-c",
        action="store_true",
        help="Show sample counts (n=X) above bars in bar chart plots",
    )

    args = parser.parse_args()

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

    if not needs_raw_data:
        print(f"Loaded {len(df)} configurations from {args.input}")

    if args.image == "main":
        create_plots(df, args.output, show_counts=args.show_counts)
    else:
        create_single_plot(args.image, df, args.output, show_counts=args.show_counts)


if __name__ == "__main__":
    main()

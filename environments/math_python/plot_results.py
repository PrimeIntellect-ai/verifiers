#!/usr/bin/env python3
"""
Plot ablation results for math-python experiments.

Creates focused plots comparing modes across models:
- Mode comparison by model (bar chart)
- RLM metrics comparison
- Timing comparison
- Standard mode tool usage
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

# Extended mode styling for ablation plots (includes sub-LLM tip variants)
ABLATION_MODE_COLORS = {
    "standard": "#E24A33",
    "rlm": "#348ABD",
    "rlm_tips": "#988ED5",
    "rlm_tips_subllm": "#2ECC71",  # Green base for sub-LLM variants
}

# Color palette for sub-LLM timeout variants (greens of different shades)
SUBLLM_TIMEOUT_COLORS = ["#27AE60", "#2ECC71", "#58D68D", "#82E0AA", "#ABEBC6"]


def get_mode_style(mode: str) -> dict:
    """Get style for any mode, including dynamic sub-LLM timeout variants."""
    if mode in MODE_STYLES:
        return MODE_STYLES[mode]
    if mode in ABLATION_MODE_COLORS:
        return {"color": ABLATION_MODE_COLORS[mode], "marker": "D", "linestyle": "-."}
    if mode.startswith("rlm_tips_subllm_"):
        # Extract timeout and assign color based on index
        return {"color": "#2ECC71", "marker": "D", "linestyle": "-."}
    return {"color": "gray", "marker": "x", "linestyle": "-"}


def get_mode_label(mode: str) -> str:
    """Get display label for any mode, including sub-LLM timeout variants."""
    if mode in MODE_LABELS:
        return MODE_LABELS[mode]
    if mode == "rlm_tips_subllm":
        return "RLM+sub-LLM"
    if mode.startswith("rlm_tips_subllm_"):
        # Extract timeout: "rlm_tips_subllm_120s" -> "RLM+sub-LLM (120s)"
        timeout = mode.replace("rlm_tips_subllm_", "").replace("s", "")
        return f"RLM+sub-LLM ({timeout}s)"
    return mode


def get_ablation_mode_order(mode: str) -> int:
    """Get sort order for ablation plot modes."""
    if mode == "standard":
        return 0
    elif mode == "rlm":
        return 1
    elif mode == "rlm_tips":
        return 2
    elif mode == "rlm_tips_subllm":
        return 3
    elif mode.startswith("rlm_tips_subllm_"):
        try:
            timeout = int(mode.replace("rlm_tips_subllm_", "").replace("s", ""))
            return 4 + timeout  # Higher timeouts come later
        except ValueError:
            return 100
    return 100


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


def plot_reward_by_model(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Plot: Mode comparison across models (grouped bar chart)."""
    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        accuracies = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                accuracies.append(model_data["correct_answer_mean"].values[0])
                if "correct_answer_count" in model_data.columns:
                    counts.append(int(model_data["correct_answer_count"].values[0]))
                else:
                    counts.append(None)
            else:
                accuracies.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            [xi + offset for xi in x],
            accuracies,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add annotations above each bar if requested
        if show_values or show_counts:
            for bar, accuracy, count in zip(bars, accuracies, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{accuracy:.2f}")
                if show_counts and count is not None and count > 0:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="gray",
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (Correct Answer)")
    ax.set_title("Reward")
    if absolute:
        extra = 0.1 + (0.05 if show_counts else 0) + (0.05 if show_values else 0)
        ax.set_ylim(0, 1.0 + extra)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
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
                        and pd.notna(model_data[metric_col].values[0])
                        else 0
                    )

                # Calculate bar position within metric group
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
    ax.set_ylabel("Count / Value")
    ax.set_title("RLM Usage Metrics by Mode")

    # Set x-ticks at the center of each metric group
    tick_positions = [i * (metric_group_width + metric_gap) for i in range(n_metrics)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([m[1] for m in metrics])

    # Create custom legend with separate entries for models (pattern) and modes (color)
    legend_handles = []

    # Mode entries (color, no hatch)
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


def plot_timing(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Timing comparison across modes and models."""
    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        times = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0 and pd.notna(model_data["total_ms_mean"].values[0]):
                times.append(
                    model_data["total_ms_mean"].values[0] / 1000
                )  # Convert to seconds
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

        # Add annotations above each bar if requested
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


def plot_turns(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Number of turns/iterations by mode and model."""
    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        turns = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            # Try both turns_mean (from aggregate) and num_turns_mean (from some datasets)
            turns_col = (
                "turns_mean" if "turns_mean" in model_data.columns else "num_turns_mean"
            )
            if (
                len(model_data) > 0
                and turns_col in model_data.columns
                and pd.notna(model_data[turns_col].values[0])
            ):
                turns.append(model_data[turns_col].values[0])
                count_col = (
                    "turns_count"
                    if "turns_count" in model_data.columns
                    else "num_turns_count"
                )
                if count_col in model_data.columns:
                    counts.append(int(model_data[count_col].values[0]))
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
            edgecolor="black",
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
                        color="gray",
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


def plot_standard_tool_usage(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Tool usage metrics for standard mode."""
    # Filter to standard mode only
    std_df = df[df["mode"] == "standard"]

    if len(std_df) == 0:
        ax.text(
            0.5,
            0.5,
            "No standard mode data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    models = list(std_df["model"].unique())
    n_models = len(models)
    hatches = HATCHES[:n_models]

    # Metrics to show
    metrics = [
        ("num_turns_mean", "Turns"),
        ("num_tool_calls_mean", "Tool Calls"),
        ("num_errors_mean", "Errors"),
    ]

    n_metrics = len(metrics)
    bar_width = 0.15
    metric_gap = 0.3

    for metric_idx, (metric_col, metric_label) in enumerate(metrics):
        metric_center = metric_idx * (n_models * bar_width + metric_gap)

        for model_idx, model in enumerate(models):
            model_data = std_df[std_df["model"] == model]
            if len(model_data) == 0:
                value = 0
                count = None
            else:
                value = (
                    model_data[metric_col].values[0]
                    if metric_col in model_data.columns
                    and pd.notna(model_data[metric_col].values[0])
                    else 0
                )
                if f"{metric_col.replace('_mean', '_count')}" in model_data.columns:
                    count = int(
                        model_data[metric_col.replace("_mean", "_count")].values[0]
                    )
                else:
                    count = None

            bar_x = metric_center + model_idx * bar_width - (n_models * bar_width) / 2

            bar = ax.bar(
                bar_x,
                value,
                bar_width,
                edgecolor="black",
                linewidth=0.5,
                hatch=hatches[model_idx],
                color=MODE_STYLES["standard"]["color"],
            )

            if show_values or show_counts:
                annotations = []
                if show_values:
                    annotations.append(f"{value:.1f}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar_x,
                        value + 0.1,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="gray",
                    )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Count")
    ax.set_title("Standard Mode Tool Usage")

    # Set x-ticks at the center of each metric group
    tick_positions = [i * (n_models * bar_width + metric_gap) for i in range(n_metrics)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([m[1] for m in metrics])

    if show_legend:
        # Create legend with model hatches
        legend_handles = []
        for model_idx, model in enumerate(models):
            legend_handles.append(
                Patch(
                    facecolor=MODE_STYLES["standard"]["color"],
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
            ncol=n_models,
            fontsize=8,
        )


def plot_token_usage(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Total token usage comparison (main model + sub-LLM for all modes)."""
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

        # Add annotations above each bar if requested
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


def plot_main_model_tokens(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Main model token usage comparison (excludes sub-LLM tokens for fair comparison)."""
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

        # Add annotations above each bar if requested
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
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            [xi + offset for xi in x],
            sub_tokens,
            width,
            bottom=main_tokens,
            color=style["color"],
            edgecolor="black",
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
                        color="gray",
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Prompt Tokens")
    ax.set_title("Prompt Token Usage")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )

    # Add in-subplot legend for Main Model / Sub-LLM distinction
    pattern_handles = [
        Patch(
            facecolor="#CCCCCC", edgecolor="black", linewidth=0.5, label="Main Model"
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor="black",
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
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            [xi + offset for xi in x],
            sub_tokens,
            width,
            bottom=main_tokens,
            color=style["color"],
            edgecolor="black",
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
                        color="gray",
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Completion Tokens")
    ax.set_title("Completion Token Usage")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )

    # Add in-subplot legend for Main Model / Sub-LLM distinction
    pattern_handles = [
        Patch(
            facecolor="#CCCCCC", edgecolor="black", linewidth=0.5, label="Main Model"
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
            label="Sub-LLM",
        ),
    ]
    ax.legend(handles=pattern_handles, loc="upper left", fontsize=8)


def plot_sub_llm_tokens(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Sub-LLM token usage comparison (RLM modes only)."""
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

        # Sub-LLM tokens only
        total_tokens = []
        counts = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                sub_prompt = model_data["sub_llm_prompt_tokens_mean"].values[0] or 0
                sub_completion = (
                    model_data["sub_llm_completion_tokens_mean"].values[0] or 0
                )
                total_tokens.append(sub_prompt + sub_completion)
                if "sub_llm_prompt_tokens_count" in model_data.columns:
                    counts.append(
                        int(model_data["sub_llm_prompt_tokens_count"].values[0])
                    )
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

        # Add annotations above each bar if requested
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
                        color="gray",
                    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Sub-LLM Tokens")
    ax.set_title("Sub-LLM Token Usage (RLM only)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def plot_timing_vs_reward(ax: plt.Axes, df: pd.DataFrame, absolute: bool = False):
    """Plot: Accuracy vs timing scatter (cost-benefit analysis)."""
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    for mode in modes:
        data = df[df["mode"] == mode]
        style = MODE_STYLES.get(mode, {"color": "gray", "marker": "o"})

        # Filter out rows with missing data
        valid_data = data.dropna(subset=["total_ms_mean", "correct_answer_mean"])

        if len(valid_data) > 0:
            # Convert to seconds
            times = valid_data["total_ms_mean"] / 1000

            ax.scatter(
                times,
                valid_data["correct_answer_mean"],
                label=MODE_LABELS.get(mode, mode).replace("\n", " "),
                color=style["color"],
                marker=style["marker"],
                s=100,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Accuracy (Correct Answer)")
    ax.set_title("Timing vs Accuracy\n(by mode)")
    if absolute:
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)


def _get_ablation_common_data(
    df: pd.DataFrame,
) -> tuple[str, list[str], dict[str, str], list[str]]:
    """Extract common data needed for all ablation subplots.

    Returns: (model_display, all_modes, subllm_color_map, labels)
    """
    models = df["model"].unique()
    model = models[0]
    model_display = normalize_model_name(model)

    # Get all modes and sort them
    all_modes = list(df["mode"].unique())
    all_modes.sort(key=get_ablation_mode_order)

    # Assign colors to sub-LLM timeout variants dynamically
    subllm_modes = [m for m in all_modes if m.startswith("rlm_tips_subllm_")]
    subllm_color_map = {}
    for i, mode in enumerate(subllm_modes):
        color_idx = i % len(SUBLLM_TIMEOUT_COLORS)
        subllm_color_map[mode] = SUBLLM_TIMEOUT_COLORS[color_idx]

    labels = [get_mode_label(mode) for mode in all_modes]

    return model_display, all_modes, subllm_color_map, labels


def _get_mode_color(mode: str, subllm_color_map: dict[str, str]) -> str:
    """Get color for a mode."""
    if mode in subllm_color_map:
        return subllm_color_map[mode]
    elif mode in ABLATION_MODE_COLORS:
        return ABLATION_MODE_COLORS[mode]
    else:
        style = get_mode_style(mode)
        return style["color"]


def _plot_ablation_reward(
    ax: plt.Axes,
    df: pd.DataFrame,
    all_modes: list[str],
    subllm_color_map: dict[str, str],
    labels: list[str],
    show_counts: bool = False,
    show_values: bool = False,
):
    """Subplot: Reward/accuracy comparison."""
    x = range(len(all_modes))
    width = 0.6

    accuracies = []
    counts = []
    colors = []

    for mode in all_modes:
        mode_data = df[df["mode"] == mode]
        if len(mode_data) > 0:
            accuracies.append(mode_data["correct_answer_mean"].values[0])
            if "correct_answer_count" in mode_data.columns:
                counts.append(int(mode_data["correct_answer_count"].values[0]))
            else:
                counts.append(None)
        else:
            accuracies.append(0)
            counts.append(None)

        colors.append(_get_mode_color(mode, subllm_color_map))

    bars = ax.bar(
        x,
        accuracies,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    if show_values or show_counts:
        for bar, acc, count in zip(bars, accuracies, counts):
            annotations = []
            if show_values:
                annotations.append(f"{acc:.2f}")
            if show_counts and count is not None and count > 0:
                annotations.append(f"n={count}")
            if annotations:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    "\n".join(annotations),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reward")
    max_acc = max(accuracies) if accuracies else 1.0
    headroom = 0.15 if show_counts else 0.1
    headroom += 0.05 if show_values else 0
    ax.set_ylim(0, max_acc * (1 + headroom))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    if max_acc >= 0.9:
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)


def _plot_ablation_tokens_stacked(
    ax: plt.Axes,
    df: pd.DataFrame,
    all_modes: list[str],
    subllm_color_map: dict[str, str],
    labels: list[str],
    show_counts: bool = False,
    show_values: bool = False,
):
    """Subplot: Token usage with stacked bars (main model + sub-LLM)."""
    x = range(len(all_modes))
    width = 0.6

    main_tokens = []
    sub_tokens = []
    colors = []
    counts = []

    for mode in all_modes:
        mode_data = df[df["mode"] == mode]
        if len(mode_data) > 0:
            # Main model tokens
            prompt = mode_data["prompt_tokens_mean"].values[0] or 0
            completion = mode_data["completion_tokens_mean"].values[0] or 0
            main_tokens.append(prompt + completion)

            # Sub-LLM tokens (0 for standard mode)
            sub_prompt = 0
            sub_completion = 0
            if "sub_llm_prompt_tokens_mean" in mode_data.columns:
                val = mode_data["sub_llm_prompt_tokens_mean"].values[0]
                sub_prompt = 0 if pd.isna(val) else val
            if "sub_llm_completion_tokens_mean" in mode_data.columns:
                val = mode_data["sub_llm_completion_tokens_mean"].values[0]
                sub_completion = 0 if pd.isna(val) else val
            sub_tokens.append(sub_prompt + sub_completion)

            # Get count
            if "prompt_tokens_count" in mode_data.columns:
                counts.append(int(mode_data["prompt_tokens_count"].values[0]))
            else:
                counts.append(None)
        else:
            main_tokens.append(0)
            sub_tokens.append(0)
            counts.append(None)

        colors.append(_get_mode_color(mode, subllm_color_map))

    # Stacked bar chart
    bars_main = ax.bar(
        x,
        main_tokens,
        width,
        label="Main Model",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Sub-LLM tokens stacked on top (use hatching to distinguish)
    bars_sub = ax.bar(
        x,
        sub_tokens,
        width,
        bottom=main_tokens,
        label="Sub-LLM",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.6,
    )

    # Add annotations above each stacked bar if requested
    if show_values or show_counts:
        for i, (main, sub, count) in enumerate(zip(main_tokens, sub_tokens, counts)):
            total_height = main + sub
            annotations = []
            if show_values:
                annotations.append(f"{int(total_height):,}")
            if show_counts and count is not None and count > 0:
                annotations.append(f"n={count}")
            if annotations:
                ax.text(
                    i,
                    total_height + total_height * 0.02,
                    "\n".join(annotations),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Tokens")
    ax.set_title("Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    # Add legend for stacked components (neutral gray color)
    legend_handles = [
        Patch(
            facecolor="#CCCCCC", edgecolor="black", linewidth=0.5, label="Main Model"
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
            label="Sub-LLM",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
    )


def _plot_ablation_timing(
    ax: plt.Axes,
    df: pd.DataFrame,
    all_modes: list[str],
    subllm_color_map: dict[str, str],
    labels: list[str],
    show_counts: bool = False,
    show_values: bool = False,
):
    """Subplot: Timing comparison."""
    x = range(len(all_modes))
    width = 0.6

    times = []
    colors = []
    counts = []

    for mode in all_modes:
        mode_data = df[df["mode"] == mode]
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

        colors.append(_get_mode_color(mode, subllm_color_map))

    bars = ax.bar(
        x,
        times,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add annotations above each bar if requested
    if show_values or show_counts:
        for bar, time_val, count in zip(bars, times, counts):
            annotations = []
            if show_values:
                annotations.append(f"{time_val:.1f}s")
            if show_counts and count is not None and count > 0:
                annotations.append(f"n={count}")
            if annotations:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.02,
                    "\n".join(annotations),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)


def _plot_ablation_prompt_tokens(
    ax: plt.Axes,
    df: pd.DataFrame,
    all_modes: list[str],
    subllm_color_map: dict[str, str],
    labels: list[str],
    show_counts: bool = False,
    show_values: bool = False,
):
    """Subplot: Prompt token usage (main model + sub-LLM stacked)."""
    x = range(len(all_modes))
    width = 0.6

    main_prompt = []
    sub_prompt = []
    colors = []
    counts = []

    for mode in all_modes:
        mode_data = df[df["mode"] == mode]
        if len(mode_data) > 0:
            # Main model prompt tokens
            val = mode_data["prompt_tokens_mean"].values[0]
            main_prompt.append(0 if pd.isna(val) else val)

            # Sub-LLM prompt tokens
            sub_val = 0
            if "sub_llm_prompt_tokens_mean" in mode_data.columns:
                v = mode_data["sub_llm_prompt_tokens_mean"].values[0]
                sub_val = 0 if pd.isna(v) else v
            sub_prompt.append(sub_val)

            # Get count
            if "prompt_tokens_count" in mode_data.columns:
                counts.append(int(mode_data["prompt_tokens_count"].values[0]))
            else:
                counts.append(None)
        else:
            main_prompt.append(0)
            sub_prompt.append(0)
            counts.append(None)

        colors.append(_get_mode_color(mode, subllm_color_map))

    # Stacked bar chart
    ax.bar(
        x,
        main_prompt,
        width,
        label="Main Model",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.bar(
        x,
        sub_prompt,
        width,
        bottom=main_prompt,
        label="Sub-LLM",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.6,
    )

    # Add annotations
    if show_values or show_counts:
        for i, (main, sub, count) in enumerate(zip(main_prompt, sub_prompt, counts)):
            total_height = main + sub
            annotations = []
            if show_values:
                annotations.append(f"{int(total_height):,}")
            if show_counts and count is not None and count > 0:
                annotations.append(f"n={count}")
            if annotations:
                ax.text(
                    i,
                    total_height + total_height * 0.02,
                    "\n".join(annotations),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Prompt Tokens")
    ax.set_title("Prompt Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    # Add legend
    legend_handles = [
        Patch(
            facecolor="#CCCCCC", edgecolor="black", linewidth=0.5, label="Main Model"
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
            label="Sub-LLM",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)


def _plot_ablation_completion_tokens(
    ax: plt.Axes,
    df: pd.DataFrame,
    all_modes: list[str],
    subllm_color_map: dict[str, str],
    labels: list[str],
    show_counts: bool = False,
    show_values: bool = False,
):
    """Subplot: Completion token usage (main model + sub-LLM stacked)."""
    x = range(len(all_modes))
    width = 0.6

    main_completion = []
    sub_completion = []
    colors = []
    counts = []

    for mode in all_modes:
        mode_data = df[df["mode"] == mode]
        if len(mode_data) > 0:
            # Main model completion tokens
            val = mode_data["completion_tokens_mean"].values[0]
            main_completion.append(0 if pd.isna(val) else val)

            # Sub-LLM completion tokens
            sub_val = 0
            if "sub_llm_completion_tokens_mean" in mode_data.columns:
                v = mode_data["sub_llm_completion_tokens_mean"].values[0]
                sub_val = 0 if pd.isna(v) else v
            sub_completion.append(sub_val)

            # Get count
            if "completion_tokens_count" in mode_data.columns:
                counts.append(int(mode_data["completion_tokens_count"].values[0]))
            else:
                counts.append(None)
        else:
            main_completion.append(0)
            sub_completion.append(0)
            counts.append(None)

        colors.append(_get_mode_color(mode, subllm_color_map))

    # Stacked bar chart
    ax.bar(
        x,
        main_completion,
        width,
        label="Main Model",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.bar(
        x,
        sub_completion,
        width,
        bottom=main_completion,
        label="Sub-LLM",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.6,
    )

    # Add annotations
    if show_values or show_counts:
        for i, (main, sub, count) in enumerate(
            zip(main_completion, sub_completion, counts)
        ):
            total_height = main + sub
            annotations = []
            if show_values:
                annotations.append(f"{int(total_height):,}")
            if show_counts and count is not None and count > 0:
                annotations.append(f"n={count}")
            if annotations:
                ax.text(
                    i,
                    total_height + total_height * 0.02,
                    "\n".join(annotations),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                )

    ax.set_xlabel("Mode")
    ax.set_ylabel("Completion Tokens")
    ax.set_title("Completion Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    # Add legend
    legend_handles = [
        Patch(
            facecolor="#CCCCCC", edgecolor="black", linewidth=0.5, label="Main Model"
        ),
        Patch(
            facecolor="#CCCCCC",
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
            label="Sub-LLM",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)


def create_ablation_timing_plots(
    df: pd.DataFrame,
    output_path: Path | None = None,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Create the 1x3 token/timing breakdown plot (includes timing).

    Shows prompt/completion token breakdown and timing for a single model:
    - Left: Prompt tokens (stacked: main model + sub-LLM)
    - Center: Completion tokens (stacked: main model + sub-LLM)
    - Right: Timing (total time per mode)

    Note: Timing data may be unreliable due to network variability.
    """
    models = df["model"].unique()
    if len(models) != 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"Token breakdown plot requires exactly 1 model\n(found {len(models)})\n\nUse -M to select a model",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        plt.show()
        return

    # Get common data
    model_display, all_modes, subllm_color_map, labels = _get_ablation_common_data(df)

    if len(all_modes) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No mode data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        plt.show()
        return

    # Create 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Plot each subplot
    _plot_ablation_prompt_tokens(
        axes[0], df, all_modes, subllm_color_map, labels, show_counts, show_values
    )
    _plot_ablation_completion_tokens(
        axes[1], df, all_modes, subllm_color_map, labels, show_counts, show_values
    )
    _plot_ablation_timing(
        axes[2], df, all_modes, subllm_color_map, labels, show_counts, show_values
    )

    # Create central legend for mode colors
    legend_handles = []
    for mode in ["standard", "rlm", "rlm_tips"]:
        if mode in all_modes:
            legend_handles.append(
                Patch(
                    facecolor=ABLATION_MODE_COLORS.get(
                        mode, MODE_STYLES.get(mode, {}).get("color", "gray")
                    ),
                    edgecolor="black",
                    linewidth=0.5,
                    label=get_mode_label(mode),
                )
            )
    subllm_in_data = [m for m in all_modes if m.startswith("rlm_tips_subllm")]
    if subllm_in_data:
        legend_handles.append(
            Patch(
                facecolor="#2ECC71",
                edgecolor="black",
                linewidth=0.5,
                label="RLM+sub-LLM tips",
            )
        )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_handles),
            fontsize=10,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def create_ablation_plots(
    df: pd.DataFrame,
    output_path: Path | None = None,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Create the 1x3 ablation comparison plot.

    Shows all mode variants for a single model:
    - Left: Reward/accuracy
    - Center: Prompt tokens (stacked: main model + sub-LLM)
    - Right: Completion tokens (stacked: main model + sub-LLM)
    """
    models = df["model"].unique()
    if len(models) != 1:
        # Create a simple figure with error message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"Ablation plot requires exactly 1 model\n(found {len(models)})\n\nUse -M to select a model",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        plt.show()
        return

    # Get common data
    model_display, all_modes, subllm_color_map, labels = _get_ablation_common_data(df)

    if len(all_modes) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No mode data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        plt.show()
        return

    # Create 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Plot each subplot
    _plot_ablation_reward(
        axes[0], df, all_modes, subllm_color_map, labels, show_counts, show_values
    )
    _plot_ablation_prompt_tokens(
        axes[1], df, all_modes, subllm_color_map, labels, show_counts, show_values
    )
    _plot_ablation_completion_tokens(
        axes[2], df, all_modes, subllm_color_map, labels, show_counts, show_values
    )

    # Create central legend for mode colors
    legend_handles = []
    for mode in ["standard", "rlm", "rlm_tips"]:
        if mode in all_modes:
            legend_handles.append(
                Patch(
                    facecolor=ABLATION_MODE_COLORS.get(
                        mode, MODE_STYLES.get(mode, {}).get("color", "gray")
                    ),
                    edgecolor="black",
                    linewidth=0.5,
                    label=get_mode_label(mode),
                )
            )
    subllm_in_data = [m for m in all_modes if m.startswith("rlm_tips_subllm")]
    if subllm_in_data:
        legend_handles.append(
            Patch(
                facecolor="#2ECC71",
                edgecolor="black",
                linewidth=0.5,
                label="RLM+sub-LLM tips",
            )
        )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_handles),
            fontsize=10,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def plot_ablation(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Legacy single-axis ablation plot (now redirects to create_ablation_plots).

    Note: This function is kept for API compatibility but the ablation plot
    is now handled specially in create_single_plot to create a 1x3 grid.
    """
    # This shouldn't be called directly anymore, but keep it for safety
    models = df["model"].unique()
    if len(models) != 1:
        ax.text(
            0.5,
            0.5,
            f"Ablation plot requires exactly 1 model\n(found {len(models)})\n\nUse -M to select a model",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        return

    model_display, all_modes, subllm_color_map, labels = _get_ablation_common_data(df)
    _plot_ablation_reward(
        ax, df, all_modes, subllm_color_map, labels, show_counts, show_values
    )


def create_plots(
    df: pd.DataFrame,
    output_path: Path | None = None,
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Create the 2x2 grid of plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Main plots (suppress individual legends)
    plot_reward_by_model(
        axes[0, 0],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
        absolute=absolute,
    )
    plot_turns(
        axes[0, 1],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )
    plot_prompt_tokens(
        axes[1, 0],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )
    plot_completion_tokens(
        axes[1, 1],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )

    # Create central legend for modes (using markers for consistency with scatter plots)
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
        "Math Python",
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
    "turns": (plot_turns, (10, 7), "Turns"),
    "timing": (plot_timing, (10, 7), "Timing Comparison"),
    "main_tokens": (plot_main_model_tokens, (10, 7), "Main Model Token Usage"),
    "tokens": (plot_token_usage, (10, 7), "Total Token Usage"),
    "prompt_tokens": (plot_prompt_tokens, (10, 7), "Prompt Token Usage"),
    "completion_tokens": (plot_completion_tokens, (10, 7), "Completion Token Usage"),
    "rlm_metrics": (plot_rlm_metrics, (12, 7), "RLM Usage Metrics"),
    "tool_usage": (plot_standard_tool_usage, (10, 7), "Standard Mode Tool Usage"),
    "sub_llm_tokens": (plot_sub_llm_tokens, (10, 7), "Sub-LLM Token Usage (RLM only)"),
    "timing_vs_reward": (plot_timing_vs_reward, (10, 7), "Timing vs Accuracy"),
    "ablation": (plot_ablation, (12, 7), "Full Ablation"),
}


def create_single_plot(
    plot_name: str,
    df: pd.DataFrame,
    output_path: Path | None = None,
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Create a single standalone plot."""
    # Handle ablation plots specially (they're 1x3 grids, not single axis)
    if plot_name == "ablation":
        create_ablation_plots(df, output_path, show_counts, show_values)
        return
    if plot_name == "ablation_timing":
        create_ablation_timing_plots(df, output_path, show_counts, show_values)
        return

    if plot_name not in PLOT_REGISTRY:
        raise ValueError(
            f"Unknown plot: {plot_name}. Available: {list(PLOT_REGISTRY.keys())}"
        )

    func, figsize, title = PLOT_REGISTRY[plot_name]

    fig, ax = plt.subplots(figsize=figsize)

    # Pass show_counts, show_values and absolute to functions that support them
    if plot_name == "reward":
        func(
            ax,
            df,
            show_counts=show_counts,
            show_values=show_values,
            absolute=absolute,
        )
    elif plot_name == "timing_vs_reward":
        func(ax, df, absolute=absolute)
    elif plot_name in (
        "turns",
        "timing",
        "main_tokens",
        "tokens",
        "prompt_tokens",
        "completion_tokens",
        "tool_usage",
        "sub_llm_tokens",
    ):
        func(ax, df, show_counts=show_counts, show_values=show_values)
    else:
        func(ax, df)

    plt.suptitle(f"Math Python: {title}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot math-python ablation results",
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
    python plot_results.py --image reward -o accuracy_plot.png
    
    # Timing vs reward analysis
    python plot_results.py --image timing_vs_reward
    
    # Show sample counts on bars
    python plot_results.py --show-counts
    
    # Full ablation comparison (requires single model)
    python plot_results.py --image ablation -M gpt-4.1-mini
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
        "-I",
        choices=[
            "main",
            "reward",
            "turns",
            "timing",
            "main_tokens",
            "tokens",
            "prompt_tokens",
            "completion_tokens",
            "rlm_metrics",
            "tool_usage",
            "sub_llm_tokens",
            "timing_vs_reward",
            "ablation",
            "ablation_timing",
        ],
        default="main",
        help="Which plot to generate: 'main' for 2x2 grid, 'ablation' for mode comparison (reward + tokens, requires -M), 'ablation_timing' for tokens + timing, or individual plot name",
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

    # Warn if ablation plot is selected with multiple models
    if args.image == "ablation" and "model" in df.columns:
        n_models = df["model"].nunique()
        if n_models != 1:
            print(
                f"\nWarning: Ablation plot works best with exactly 1 model (found {n_models})."
            )
            print("Use -M <model> to select a specific model.")
            print("Available models:")
            for model in df["model"].unique():
                print(f"  {model}")
            print()

    if args.image == "main":
        create_plots(
            df,
            args.output,
            show_counts=args.show_counts,
            show_values=args.show_values,
            absolute=args.absolute,
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

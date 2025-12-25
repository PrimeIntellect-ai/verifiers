#!/usr/bin/env python3
"""
Cross-environment plotting script for comparing RLM vs standard mode performance.

Aggregates results from multiple environments and creates comparison plots.

Usage:
    # Plot all environments and models (main grid)
    python environments/plot_results.py

    # Filter to specific environments
    python environments/plot_results.py -e oolong needle_in_haystack

    # Filter to specific models (aggregates across them)
    python environments/plot_results.py -M gpt-4.1-mini gpt-5-mini

    # Generate specific plot
    python environments/plot_results.py --image lift

    # List available environments
    python environments/plot_results.py --list-environments
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Set style at module level so it applies to all plots
sns.set_style("whitegrid")

# Mode styling (consistent with individual environment plots)
MODE_ORDER = ["standard", "rlm", "rlm_tips"]
MODE_LABELS = {
    "standard": "LLM",
    "rlm": "RLM",
    "rlm_tips": "RLM+tips",
}
MODE_STYLES = {
    "standard": {"color": "#E24A33", "marker": "o", "linestyle": "-"},
    "rlm": {"color": "#348ABD", "marker": "s", "linestyle": "--"},
    "rlm_tips": {"color": "#988ED5", "marker": "^", "linestyle": ":"},
}

# Environment display names (auto-generated from folder names if not specified)
ENV_LABELS = {
    "oolong": "Oolong",
    "needle_in_haystack": "Needle in\nHaystack",
    "verbatim_copy": "Verbatim\nCopy",
    "math_python": "Math\nPython",
    "deepdive": "DeepDive",
    "gsm8k": "GSM8K",
}

# Environment markers for scatter plots (shapes identify environments)
ENV_MARKERS = {
    "deepdive": "o",  # circle
    "math_python": "s",  # square
    "needle_in_haystack": "^",  # triangle up
    "oolong": "D",  # diamond
    "verbatim_copy": "p",  # pentagon
    "gsm8k": "h",  # hexagon
}


def normalize_model_name(model: str) -> str:
    """Normalize model name for display."""
    # Remove common prefixes
    for prefix in ["openai/", "anthropic/", "meta-llama/", "google/"]:
        if model.startswith(prefix):
            model = model[len(prefix) :]
    # Truncate if too long
    if len(model) > 25:
        model = model[:22] + "..."
    return model


def get_env_label(env_name: str) -> str:
    """Get display label for environment."""
    if env_name in ENV_LABELS:
        return ENV_LABELS[env_name]
    # Auto-generate from folder name
    return env_name.replace("_", "\n").title()


def discover_environments(base_path: Path) -> list[str]:
    """Discover environments that have outputs/aggregate.csv files."""
    environments = []
    for env_dir in sorted(base_path.iterdir()):
        if env_dir.is_dir():
            csv_path = env_dir / "outputs" / "aggregate.csv"
            if csv_path.exists():
                environments.append(env_dir.name)
    return environments


def load_environment_data(base_path: Path, env_name: str) -> pd.DataFrame | None:
    """Load outputs/aggregate.csv for an environment."""
    csv_path = base_path / env_name / "outputs" / "aggregate.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df["environment"] = env_name

    # Normalize reward column to a standard name for cross-environment comparison
    reward_columns = [
        "judge_reward_mean",
        "reward_mean",
        "correct_answer_mean",
        "accuracy_mean",
        "partial_match_mean",
        "exact_match_mean",
    ]
    for col in reward_columns:
        if col in df.columns and "reward_normalized" not in df.columns:
            df["reward_normalized"] = df[col]
            break

    return df


def load_all_environments(
    base_path: Path,
    environments: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Load and combine data from multiple environments."""
    if environments is None:
        environments = discover_environments(base_path)

    if not environments:
        raise ValueError("No environments found with outputs/aggregate.csv")

    dfs = []
    for env_name in environments:
        df = load_environment_data(base_path, env_name)
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise ValueError("No data loaded from any environment")

    combined = pd.concat(dfs, ignore_index=True)

    # Filter by models if specified
    if models and "model" in combined.columns:
        # Build combined mask for all model patterns
        combined_mask = None
        for model_pattern in models:
            if model_pattern in combined["model"].unique():
                mask = combined["model"] == model_pattern
            else:
                mask = combined["model"].str.contains(
                    model_pattern, case=False, na=False
                )
            combined_mask = mask if combined_mask is None else (combined_mask | mask)

        if combined_mask is not None:
            combined = combined[combined_mask]

    return combined


def aggregate_across_models(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across models for each environment/mode combination.

    Only aggregates by environment and mode (ignoring environment-specific
    grouping columns like 'subset', 'needle_type', etc.) to enable
    cross-environment comparison.
    """
    if "model" not in df.columns:
        return df

    # Identify numeric columns to aggregate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Only group by environment and mode for cross-environment comparison
    # Environment-specific columns (subset, needle_type, etc.) are aggregated over
    group_cols = ["environment", "mode"]

    agg_dict = {}
    for col in numeric_cols:
        if col not in group_cols:
            if "_count" in col:
                agg_dict[col] = "sum"
            else:
                agg_dict[col] = "mean"

    if not agg_dict:
        return df

    return df.groupby(group_cols, as_index=False).agg(agg_dict)


def get_reward_column(df: pd.DataFrame) -> str:
    """Find the reward column in the dataframe."""
    # Prefer normalized column if it exists (for cross-environment comparison)
    if "reward_normalized" in df.columns:
        return "reward_normalized"

    # Priority order for reward columns (fallback for single-environment data)
    for col in [
        "judge_reward_mean",
        "reward_mean",
        "correct_answer_mean",
        "accuracy_mean",
        "partial_match_mean",
        "exact_match_mean",
    ]:
        if col in df.columns:
            return col
    raise ValueError("No reward column found in data")


# =============================================================================
# Plot Functions
# =============================================================================


def get_count_for_env_data(env_data: pd.DataFrame) -> int | None:
    """Get the sample count from environment-specific data.

    Different environments have different count column names, so we check
    for non-zero values in known count columns.
    """
    count_columns = [
        "judge_reward_count",
        "reward_count",
        "correct_answer_count",
        "accuracy_count",
        "partial_match_count",
        "exact_match_count",
    ]
    for col in count_columns:
        if col in env_data.columns:
            val = env_data[col].sum()
            if not pd.isna(val) and val > 0:
                return int(val)
    return None


def plot_reward_by_environment(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Plot: Reward comparison across environments, grouped by mode."""
    reward_col = get_reward_column(df)

    environments = df["environment"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = np.arange(len(environments))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        rewards = []
        counts = []
        for env in environments:
            env_data = mode_data[mode_data["environment"] == env]
            if len(env_data) > 0:
                rewards.append(env_data[reward_col].mean())
                counts.append(get_count_for_env_data(env_data))
            else:
                rewards.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            x + offset,
            rewards,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
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
                        rotation=90,
                    )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Reward (Accuracy)")
    ax.set_title("Reward by Environment")
    if absolute:
        extra = 0.1 + (0.05 if show_counts else 0) + (0.05 if show_values else 0)
        ax.set_ylim(0, 1.0 + extra)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([get_env_label(e) for e in environments], fontsize=9)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)


def plot_rlm_lift(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: RLM lift (Δ reward) over standard baseline per environment."""
    reward_col = get_reward_column(df)

    environments = df["environment"].unique()
    rlm_modes = [m for m in ["rlm", "rlm_tips"] if m in df["mode"].unique()]

    if "standard" not in df["mode"].unique():
        ax.text(
            0.5,
            0.5,
            "No standard baseline data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    x = np.arange(len(environments))
    width = 0.35

    # Get standard baseline per environment (None if missing)
    standard_rewards = {}
    for env in environments:
        std_data = df[(df["environment"] == env) & (df["mode"] == "standard")]
        if len(std_data) > 0:
            standard_rewards[env] = std_data[reward_col].mean()
        else:
            standard_rewards[env] = None  # Mark as missing

    for i, mode in enumerate(rlm_modes):
        mode_data = df[df["mode"] == mode]
        lifts = []
        counts = []
        for env in environments:
            env_data = mode_data[mode_data["environment"] == env]
            baseline = standard_rewards.get(env)
            # Only show lift if both RLM data AND standard baseline exist
            if len(env_data) > 0 and baseline is not None:
                lift = env_data[reward_col].mean() - baseline
                lifts.append(lift)
                counts.append(get_count_for_env_data(env_data))
            else:
                lifts.append(0)  # No bar (will be at 0)
                counts.append(None)

        offset = (i - len(rlm_modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            x + offset,
            lifts,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Color bars by positive/negative, hide bars where baseline is missing
        for j, (bar, lift) in enumerate(zip(bars, lifts)):
            env = environments[j]
            if standard_rewards.get(env) is None:
                bar.set_alpha(0)  # Hide bar entirely
            elif lift < 0:
                bar.set_alpha(0.6)

        # Add annotations above bars if requested
        if show_values or show_counts:
            for j, (bar, lift, count) in enumerate(zip(bars, lifts, counts)):
                env = environments[j]
                if standard_rewards.get(env) is not None:
                    annotations = []
                    if show_values:
                        annotations.append(f"{lift:+.2f}")
                    if show_counts and count is not None:
                        annotations.append(f"n={count}")
                    if annotations:
                        y_pos = (
                            bar.get_height() + 0.01
                            if bar.get_height() >= 0
                            else bar.get_height() - 0.02
                        )
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            y_pos,
                            "\n".join(annotations),
                            ha="center",
                            va="bottom" if bar.get_height() >= 0 else "top",
                            fontsize=6,
                            rotation=90,
                        )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Δ Reward (RLM - Standard)")
    ax.set_title("RLM Lift over Standard Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([get_env_label(e) for e in environments], fontsize=9)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)


def plot_token_usage(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Total token usage by environment and mode."""
    environments = df["environment"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = np.arange(len(environments))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        tokens = []
        counts = []
        for env in environments:
            env_data = mode_data[mode_data["environment"] == env]
            if len(env_data) > 0:
                # Sum main + sub-LLM tokens
                main_prompt = (
                    env_data["prompt_tokens_mean"].values[0]
                    if "prompt_tokens_mean" in env_data.columns
                    else 0
                )
                main_completion = (
                    env_data["completion_tokens_mean"].values[0]
                    if "completion_tokens_mean" in env_data.columns
                    else 0
                )
                sub_prompt = 0
                sub_completion = 0
                if "sub_llm_prompt_tokens_mean" in env_data.columns:
                    val = env_data["sub_llm_prompt_tokens_mean"].values[0]
                    sub_prompt = 0 if pd.isna(val) else val
                if "sub_llm_completion_tokens_mean" in env_data.columns:
                    val = env_data["sub_llm_completion_tokens_mean"].values[0]
                    sub_completion = 0 if pd.isna(val) else val
                tokens.append(
                    (main_prompt or 0)
                    + (main_completion or 0)
                    + sub_prompt
                    + sub_completion
                )
                counts.append(get_count_for_env_data(env_data))
            else:
                tokens.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            x + offset,
            tokens,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, token_count, count in zip(bars, tokens, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{int(token_count):,}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 100,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                    )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Total Tokens")
    ax.set_title("Total Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels([get_env_label(e) for e in environments], fontsize=9)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)


def plot_main_model_tokens(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Main model token usage only (no sub-LLM tokens)."""
    environments = df["environment"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = np.arange(len(environments))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        tokens = []
        counts = []
        for env in environments:
            env_data = mode_data[mode_data["environment"] == env]
            if len(env_data) > 0:
                # Main model tokens only (no sub-LLM)
                main_prompt = (
                    env_data["prompt_tokens_mean"].values[0]
                    if "prompt_tokens_mean" in env_data.columns
                    else 0
                )
                main_completion = (
                    env_data["completion_tokens_mean"].values[0]
                    if "completion_tokens_mean" in env_data.columns
                    else 0
                )
                tokens.append((main_prompt or 0) + (main_completion or 0))
                counts.append(get_count_for_env_data(env_data))
            else:
                tokens.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            x + offset,
            tokens,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, token_count, count in zip(bars, tokens, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{int(token_count):,}")
                if show_counts and count is not None:
                    annotations.append(f"n={count}")
                if annotations:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 100,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                    )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Main Model Tokens")
    ax.set_title("Main Model Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels([get_env_label(e) for e in environments], fontsize=9)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)


def plot_token_efficiency(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Plot: Token efficiency relative to standard baseline (normalized per environment).

    Uses only main model tokens to show the context compression benefit of RLM.
    Standard mode = 1.0, other modes shown as multipliers.
    """
    reward_col = get_reward_column(df)

    environments = df["environment"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    # Calculate raw efficiency and counts for each (env, mode)
    efficiency_map = {}
    count_map = {}
    for env in environments:
        for mode in modes:
            env_mode_data = df[(df["environment"] == env) & (df["mode"] == mode)]
            if len(env_mode_data) > 0:
                reward = env_mode_data[reward_col].mean()
                main_prompt = (
                    env_mode_data["prompt_tokens_mean"].values[0]
                    if "prompt_tokens_mean" in env_mode_data.columns
                    else 0
                )
                main_completion = (
                    env_mode_data["completion_tokens_mean"].values[0]
                    if "completion_tokens_mean" in env_mode_data.columns
                    else 0
                )
                total_tokens = (main_prompt or 0) + (main_completion or 0)
                if total_tokens > 0:
                    efficiency_map[(env, mode)] = reward / total_tokens
                else:
                    efficiency_map[(env, mode)] = 0
                count_val = get_count_for_env_data(env_mode_data)
                if count_val is not None:
                    count_map[(env, mode)] = count_val

    x = np.arange(len(environments))
    width = 0.25

    for i, mode in enumerate(modes):
        normalized_efficiencies = []
        counts = []
        for env in environments:
            raw_eff = efficiency_map.get((env, mode), 0)
            baseline_eff = efficiency_map.get((env, "standard"), 0)
            # Normalize: standard = 1.0
            if baseline_eff > 0:
                normalized_efficiencies.append(raw_eff / baseline_eff)
            else:
                normalized_efficiencies.append(0)
            counts.append(count_map.get((env, mode)))

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            x + offset,
            normalized_efficiencies,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, eff, count in zip(bars, normalized_efficiencies, counts):
                if bar.get_height() > 0:
                    annotations = []
                    if show_values:
                        annotations.append(f"{eff:.2f}")
                    if show_counts and count is not None:
                        annotations.append(f"n={count}")
                    if annotations:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.1,
                            "\n".join(annotations),
                            ha="center",
                            va="bottom",
                            fontsize=6,
                            rotation=90,
                        )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Reward per Token (Relative to LLM)")
    ax.set_title("Main Model Token Efficiency")
    ax.set_xticks(x)
    ax.set_xticklabels([get_env_label(e) for e in environments], fontsize=9)
    if absolute:
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)


def plot_timing(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Timing comparison by environment and mode."""
    environments = df["environment"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    # Find timing column
    timing_col = None
    for col in ["total_time_mean", "rollout_time_mean", "generation_ms_mean"]:
        if col in df.columns:
            timing_col = col
            break

    if timing_col is None:
        ax.text(
            0.5,
            0.5,
            "No timing data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    x = np.arange(len(environments))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        times = []
        counts = []
        for env in environments:
            env_data = mode_data[mode_data["environment"] == env]
            if len(env_data) > 0:
                time_val = env_data[timing_col].mean()
                # Convert to seconds if in ms
                if "ms" in timing_col:
                    time_val = time_val / 1000
                times.append(time_val)
                counts.append(get_count_for_env_data(env_data))
            else:
                times.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            x + offset,
            times,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
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
                        bar.get_height() + 5,
                        "\n".join(annotations),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                    )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing by Environment")
    ax.set_xticks(x)
    ax.set_xticklabels([get_env_label(e) for e in environments], fontsize=9)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)


def plot_reward_vs_tokens_scatter(
    ax: plt.Axes, df: pd.DataFrame, show_legend: bool = True, absolute: bool = False
):
    """Plot: Reward vs main model tokens scatter.

    Colors = modes (consistent with other plots)
    Shapes = environments (for identification)
    Lines connect same-environment points across modes.
    """
    reward_col = get_reward_column(df)

    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]
    environments = list(df["environment"].unique())

    # First, compute all data points: {env: {mode: (tokens_k, reward)}}
    env_data_map: dict[str, dict[str, tuple[float, float]]] = {}
    for env in environments:
        env_data_map[env] = {}
        for mode in modes:
            env_mode_data = df[(df["environment"] == env) & (df["mode"] == mode)]
            if len(env_mode_data) == 0:
                continue

            reward = env_mode_data[reward_col].mean()
            main_prompt = (
                env_mode_data["prompt_tokens_mean"].values[0]
                if "prompt_tokens_mean" in env_mode_data.columns
                else 0
            )
            main_completion = (
                env_mode_data["completion_tokens_mean"].values[0]
                if "completion_tokens_mean" in env_mode_data.columns
                else 0
            )
            total_tokens = (main_prompt or 0) + (main_completion or 0)
            env_data_map[env][mode] = (total_tokens / 1000, reward)  # tokens in K

    # Draw scatter points: color=mode, shape=environment
    for env in environments:
        env_marker = ENV_MARKERS.get(env, "o")  # default to circle
        for mode in modes:
            if mode not in env_data_map.get(env, {}):
                continue
            tokens_k, reward = env_data_map[env][mode]
            mode_color = MODE_STYLES.get(mode, {"color": "gray"})["color"]
            ax.scatter(
                tokens_k,
                reward,
                color=mode_color,
                marker=env_marker,
                s=120,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.5,
                zorder=2,
            )

    ax.set_xlabel("Main Model Tokens (K)")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Tokens")
    if absolute:
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    if show_legend:
        # Create two-part legend: modes (colors) in left column, environments (shapes) in right columns
        # Mode legend handles (colored circles)
        mode_handles = [
            plt.scatter(
                [],
                [],
                color=MODE_STYLES[m]["color"],
                marker="o",
                s=80,
                edgecolors="black",
                linewidth=0.5,
                label=MODE_LABELS.get(m, m),
            )
            for m in modes
        ]
        # Environment legend handles (gray shapes)
        env_list = list(environments)
        env_handles = [
            plt.scatter(
                [],
                [],
                color="gray",
                marker=ENV_MARKERS.get(env, "o"),
                s=80,
                edgecolors="black",
                linewidth=0.5,
                label=get_env_label(env).replace("\n", " "),
            )
            for env in env_list
        ]

        # Interleave: mode, env, env pattern for 3-column layout
        # Row 1: mode0, env0, env3
        # Row 2: mode1, env1, env4
        # Row 3: mode2, env2, (empty)
        interleaved_handles = []
        n_envs = len(env_handles)
        for i in range(3):  # 3 rows
            if i < len(mode_handles):
                interleaved_handles.append(mode_handles[i])
            if i < n_envs:
                interleaved_handles.append(env_handles[i])
            if i + 3 < n_envs:
                interleaved_handles.append(env_handles[i + 3])

        ax.legend(
            handles=interleaved_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            fontsize=8,
            columnspacing=1.5,
        )


def plot_rlm_overhead(
    ax: plt.Axes,
    df: pd.DataFrame,
    show_legend: bool = True,
    show_counts: bool = False,
    show_values: bool = False,
):
    """Plot: Sub-LLM call count by environment (RLM modes only)."""
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

    # Check for the metric column
    metric_col = "sub_llm_call_count_mean"
    if metric_col not in rlm_df.columns:
        ax.text(
            0.5,
            0.5,
            "No sub-LLM call data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    environments = rlm_df["environment"].unique()
    modes = [m for m in ["rlm", "rlm_tips"] if m in rlm_df["mode"].unique()]

    x = np.arange(len(environments))
    width = 0.35

    for i, mode in enumerate(modes):
        mode_data = rlm_df[rlm_df["mode"] == mode]
        values = []
        counts = []
        for env in environments:
            env_data = mode_data[mode_data["environment"] == env]
            if len(env_data) > 0 and metric_col in env_data.columns:
                val = env_data[metric_col].values[0]
                values.append(0 if pd.isna(val) else val)
                counts.append(get_count_for_env_data(env_data))
            else:
                values.append(0)
                counts.append(None)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add annotations above bars if requested
        if show_values or show_counts:
            for bar, val, count in zip(bars, values, counts):
                annotations = []
                if show_values:
                    annotations.append(f"{val:.1f}")
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
                        rotation=90,
                    )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Sub-LLM Calls")
    ax.set_title("RLM Sub-LLM Call Count")
    ax.set_xticks(x)
    ax.set_xticklabels([get_env_label(e) for e in environments], fontsize=9)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)


def plot_heatmap(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Heatmap of reward by mode × environment."""
    reward_col = get_reward_column(df)

    # Create pivot table
    pivot = df.pivot_table(
        values=reward_col,
        index="mode",
        columns="environment",
        aggfunc="mean",
    )

    # Reorder
    row_order = [m for m in MODE_ORDER if m in pivot.index]
    pivot = pivot.reindex(index=row_order)

    # Rename for display
    pivot.index = [MODE_LABELS.get(m, m).replace("\n", " ") for m in pivot.index]
    pivot.columns = [get_env_label(e).replace("\n", " ") for e in pivot.columns]

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Reward"},
    )
    ax.set_title("Reward Heatmap")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Mode")


# =============================================================================
# Main Plot Creation
# =============================================================================


def create_main_plots(
    df: pd.DataFrame,
    output_path: Path | None,
    model_info: str = "",
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Create the main 2x3 grid of plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Top row: Performance metrics
    plot_reward_by_environment(
        axes[0, 0],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
        absolute=absolute,
    )
    plot_rlm_lift(
        axes[0, 1],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )
    plot_rlm_overhead(
        axes[0, 2],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )

    # Bottom row: Token costs
    plot_main_model_tokens(
        axes[1, 0],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )
    plot_token_usage(
        axes[1, 1],
        df,
        show_legend=False,
        show_counts=show_counts,
        show_values=show_values,
    )
    plot_reward_vs_tokens_scatter(axes[1, 2], df, show_legend=False, absolute=absolute)

    # Create central legend with modes (colors) and environments (shapes)
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]
    environments = list(df["environment"].unique())

    # Mode legend handles (colored squares)
    mode_handles = [
        Patch(
            facecolor=MODE_STYLES[m]["color"],
            edgecolor="black",
            linewidth=0.5,
            label=MODE_LABELS.get(m, m).replace("\n", " "),
        )
        for m in modes
    ]
    # Environment legend handles (gray shapes)
    env_handles = [
        plt.scatter(
            [],
            [],
            color="gray",
            marker=ENV_MARKERS.get(env, "o"),
            s=60,
            edgecolors="black",
            linewidth=0.5,
            label=get_env_label(env).replace("\n", " "),
        )
        for env in environments
    ]

    # Combined legend
    all_handles = mode_handles + env_handles
    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=len(all_handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.0,
    )

    if model_info:
        plt.suptitle(model_info, fontsize=14, y=0.98)
    else:
        plt.suptitle("Cross-Environment RLM Analysis", fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.93)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()


def create_single_plot(
    plot_name: str,
    df: pd.DataFrame,
    output_path: Path | None,
    model_info: str = "",
    show_counts: bool = False,
    show_values: bool = False,
    absolute: bool = False,
):
    """Create a single plot by name."""
    plot_registry = {
        "reward": (plot_reward_by_environment, (10, 7)),
        "lift": (plot_rlm_lift, (10, 7)),
        "tokens": (plot_token_usage, (10, 7)),
        "main_tokens": (plot_main_model_tokens, (10, 7)),
        "efficiency": (plot_token_efficiency, (10, 7)),
        "timing": (plot_timing, (10, 7)),
        "scatter": (plot_reward_vs_tokens_scatter, (10, 7)),
        "overhead": (plot_rlm_overhead, (12, 7)),
        "heatmap": (plot_heatmap, (10, 6)),
    }

    if plot_name not in plot_registry:
        print(f"Unknown plot: {plot_name}")
        print(f"Available plots: {list(plot_registry.keys())}")
        return

    plot_func, figsize = plot_registry[plot_name]

    fig, ax = plt.subplots(figsize=figsize)

    # Pass show_counts, show_values and absolute to functions that support them
    if plot_name in ("reward", "efficiency"):
        plot_func(
            ax,
            df,
            show_counts=show_counts,
            show_values=show_values,
            absolute=absolute,
        )
    elif plot_name == "scatter":
        plot_func(ax, df, absolute=absolute)
    elif plot_name in ("lift", "tokens", "main_tokens", "timing", "overhead"):
        plot_func(ax, df, show_counts=show_counts, show_values=show_values)
    else:
        plot_func(ax, df)

    if model_info:
        ax.set_title(ax.get_title() + f"\n({model_info})", fontsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Cross-environment plotting for RLM comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plot all environments (main grid)
    python environments/plot_results.py
    
    # Filter to specific environments
    python environments/plot_results.py -e oolong needle_in_haystack
    
    # Filter to specific models (aggregates across them)
    python environments/plot_results.py -M gpt-4.1-mini gpt-5-mini
    
    # Generate specific plot
    python environments/plot_results.py --image lift
    
    # List available environments
    python environments/plot_results.py --list-environments
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(__file__).parent,
        help="Base path to environments directory (default: script directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for plot image (default: display)",
    )
    parser.add_argument(
        "--image",
        "-I",
        type=str,
        choices=[
            "main",
            "reward",
            "lift",
            "tokens",
            "main_tokens",
            "efficiency",
            "timing",
            "scatter",
            "overhead",
            "heatmap",
        ],
        default="main",
        help="Which plot to generate (default: main grid)",
    )

    # Environment filtering
    parser.add_argument(
        "--environment",
        "-e",
        nargs="*",
        type=str,
        default=None,
        help="Filter to specific environments (default: all with outputs/aggregate.csv)",
    )
    parser.add_argument(
        "--list-environments",
        action="store_true",
        help="List available environments and exit",
    )

    # Model filtering
    parser.add_argument(
        "--model",
        "-M",
        nargs="*",
        type=str,
        default=None,
        help="Filter to specific models (supports substring matching, aggregates across multiple)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models in the data and exit",
    )

    # Aggregation
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Don't aggregate across models (show per-model data)",
    )

    # Display options
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

    # List environments
    if args.list_environments:
        environments = discover_environments(args.input)
        if environments:
            print(f"Available environments ({len(environments)} total):")
            for env in environments:
                print(f"  {env}")
        else:
            print("No environments found with outputs/aggregate.csv")
        return

    # Load data
    try:
        df = load_all_environments(args.input, args.environment, args.model)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # List models
    if args.list_models:
        if "model" not in df.columns:
            print("No model column in data")
        else:
            models = sorted(df["model"].unique())
            print(f"Available models ({len(models)} total):")
            for model in models:
                count = len(df[df["model"] == model])
                print(f"  {model}  ({count} rows)")
        return

    # Build model info string for plot titles
    model_info = ""
    if args.model:
        if len(args.model) == 1:
            model_info = args.model[0]
        else:
            model_info = ", ".join(args.model[:3]) + (
                "..." if len(args.model) > 3 else ""
            )
    elif "model" in df.columns:
        n_models = df["model"].nunique()
        if n_models > 1:
            model_info = f"Aggregated across {n_models} models"

    # Aggregate across models if requested
    if not args.no_aggregate and "model" in df.columns and df["model"].nunique() > 1:
        df = aggregate_across_models(df)

    # Report what we loaded
    environments = df["environment"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]
    print(f"Loaded data: {len(environments)} environments, {len(modes)} modes")
    print(f"  Environments: {list(environments)}")
    print(f"  Modes: {modes}")

    # Generate plots
    if args.image == "main":
        create_main_plots(
            df,
            args.output,
            model_info,
            show_counts=args.show_counts,
            show_values=args.show_values,
            absolute=args.absolute,
        )
    else:
        create_single_plot(
            args.image,
            df,
            args.output,
            model_info,
            show_counts=args.show_counts,
            show_values=args.show_values,
            absolute=args.absolute,
        )


if __name__ == "__main__":
    main()

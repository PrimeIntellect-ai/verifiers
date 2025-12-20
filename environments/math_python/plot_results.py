#!/usr/bin/env python3
"""
Plot ablation results for math-python experiments.

Creates focused plots comparing modes across models:
- Mode comparison by model (bar chart)
- RLM metrics comparison
- Timing comparison
- Standard mode tool usage
- Optional: timing_vs_accuracy (cost-benefit analysis)

Usage:
    python plot_results.py [--input aggregate.csv] [--output plots.png]
    python plot_results.py --image accuracy
    python plot_results.py --image rlm_metrics
    python plot_results.py --image timing_vs_accuracy
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
    "standard": "Standard\n(PythonEnv)",
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


def plot_accuracy_by_model(ax: plt.Axes, df: pd.DataFrame):
    """Plot: Mode comparison across models (grouped bar chart)."""
    models = df["model"].unique()
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    x = range(len(models))
    width = 0.25

    for i, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]
        accuracies = []
        errors = []
        for model in models:
            model_data = mode_data[mode_data["model"] == model]
            if len(model_data) > 0:
                accuracies.append(model_data["correct_answer_mean"].values[0])
                errors.append(model_data["correct_answer_std"].values[0])
            else:
                accuracies.append(0)
                errors.append(0)

        offset = (i - len(modes) / 2 + 0.5) * width
        style = MODE_STYLES.get(mode, {"color": "gray"})
        ax.bar(
            [xi + offset for xi in x],
            accuracies,
            width,
            yerr=errors,
            label=MODE_LABELS.get(mode, mode),
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (Correct Answer)")
    ax.set_title("Mode Comparison by Model")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [normalize_model_name(m) for m in models], rotation=15, ha="right"
    )
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

    models = rlm_df["model"].unique()
    modes = [m for m in ["rlm", "rlm_tips"] if m in rlm_df["mode"].unique()]

    # Metrics to show
    metrics = [
        ("turns_mean", "Turns"),
        ("sub_llm_call_count_mean", "Sub-LLM Calls"),
        ("sub_llm_mean_batch_size_mean", "Avg Batch Size"),
    ]

    x = range(len(metrics))
    width = 0.35

    # For each model, create a subplot grouping
    for model_idx, model in enumerate(models):
        model_data = rlm_df[rlm_df["model"] == model]

        for i, mode in enumerate(modes):
            mode_data = model_data[model_data["mode"] == mode]
            if len(mode_data) == 0:
                continue

            values = [
                mode_data[m[0]].values[0]
                if m[0] in mode_data.columns and pd.notna(mode_data[m[0]].values[0])
                else 0
                for m in metrics
            ]

            offset = (i - len(modes) / 2 + 0.5) * width + model_idx * (
                len(modes) * width + 0.3
            )
            style = MODE_STYLES.get(mode, {"color": "gray"})

            ax.bar(
                [xi + offset for xi in x],
                values,
                width,
                label=f"{normalize_model_name(model)}\n{mode}",
                color=style["color"],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7 + 0.3 * (model_idx / max(1, len(models) - 1)),
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Count / Value")
    ax.set_title("RLM Usage Metrics by Mode")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.legend(loc="upper right", fontsize=8, ncol=2)


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
    ax.legend(loc="upper right", fontsize=9)


def plot_standard_tool_usage(ax: plt.Axes, df: pd.DataFrame):
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

    models = std_df["model"].unique()

    # Metrics to show
    metrics = [
        ("num_turns_mean", "Turns"),
        ("num_tool_calls_mean", "Tool Calls"),
        ("num_errors_mean", "Errors"),
    ]

    x = range(len(metrics))
    width = 0.35

    for model_idx, model in enumerate(models):
        model_data = std_df[std_df["model"] == model]
        if len(model_data) == 0:
            continue

        values = [
            model_data[m[0]].values[0]
            if m[0] in model_data.columns and pd.notna(model_data[m[0]].values[0])
            else 0
            for m in metrics
        ]

        offset = (model_idx - len(models) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            values,
            width,
            label=normalize_model_name(model),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Count")
    ax.set_title("Standard Mode Tool Usage")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.legend(loc="upper right", fontsize=9)


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
    ax.legend(loc="upper right", fontsize=9)


def plot_timing_vs_accuracy(ax: plt.Axes, df: pd.DataFrame):
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
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)


def create_plots(df: pd.DataFrame, output_path: Path | None = None):
    """Create the 2x2 grid of plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Set style
    sns.set_style("whitegrid")

    # Main plots
    plot_accuracy_by_model(axes[0, 0], df)
    plot_timing(axes[0, 1], df)
    plot_rlm_metrics(axes[1, 0], df)
    plot_standard_tool_usage(axes[1, 1], df)

    plt.suptitle(
        "Math Python Ablation Results: Mode Comparison",
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
    "accuracy": (plot_accuracy_by_model, (10, 7), "Mode Comparison by Model"),
    "timing": (plot_timing, (10, 7), "Timing Comparison"),
    "rlm_metrics": (plot_rlm_metrics, (12, 7), "RLM Usage Metrics"),
    "tool_usage": (plot_standard_tool_usage, (10, 7), "Standard Mode Tool Usage"),
    "tokens": (plot_token_usage, (10, 7), "Token Usage"),
    # Additional timing plots
    "timing_vs_accuracy": (plot_timing_vs_accuracy, (10, 7), "Timing vs Accuracy"),
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
    
    # Show individual accuracy plot
    python plot_results.py --image accuracy
    
    # Show RLM metrics comparison
    python plot_results.py --image rlm_metrics
    
    # Save plot to file
    python plot_results.py --image accuracy -o accuracy_plot.png
    
    # Timing vs accuracy analysis
    python plot_results.py --image timing_vs_accuracy
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
            "accuracy",
            "timing",
            "rlm_metrics",
            "tool_usage",
            "tokens",
            "timing_vs_accuracy",
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

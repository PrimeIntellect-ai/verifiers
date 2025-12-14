#!/usr/bin/env python3
"""
Plot ablation results for verbatim_copy experiments.

Creates a 2x3 grid of plots:
- Top row: Individual metrics (reward vs each ablation variable)
- Bottom row: Aggregation views (heatmap, distribution, scatter)

Usage:
    python plot_results.py [--input aggregate.csv] [--output plots.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the aggregate results CSV."""
    df = pd.read_csv(csv_path)
    
    # Convert mean_fragment_length NaN to "None" string for display
    df["frag_label"] = df["mean_fragment_length"].apply(
        lambda x: "None" if pd.isna(x) else str(int(x))
    )
    
    return df


def plot_reward_vs_content_type(ax: plt.Axes, df: pd.DataFrame):
    """Plot 1: Reward vs content_type (bar chart)."""
    # Filter to target_length=500, mean_fragment_length=20
    filtered = df[(df["target_length"] == 500) & (df["mean_fragment_length"] == 20)]
    
    # Order content types
    order = ["words", "json", "csv", "codes", "mixed"]
    filtered = filtered.set_index("content_type").loc[order].reset_index()
    
    colors = sns.color_palette("husl", len(order))
    bars = ax.bar(
        filtered["content_type"],
        filtered["reward_mean"],
        color=colors,
        edgecolor="black",
        linewidth=1,
    )
    
    ax.set_xlabel("Content Type")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Reward vs Content Type\n(len=500, frag=20)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, filtered["reward_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_reward_vs_length(ax: plt.Axes, df: pd.DataFrame):
    """Plot 2: Reward vs target_length (line plot)."""
    # Filter to mean_fragment_length=20, show all content types
    filtered = df[df["mean_fragment_length"] == 20]
    
    content_types = ["words", "json", "csv", "codes", "mixed"]
    colors = sns.color_palette("husl", len(content_types))
    
    for content_type, color in zip(content_types, colors):
        data = filtered[filtered["content_type"] == content_type].sort_values("target_length")
        if len(data) > 0:
            ax.plot(
                data["target_length"],
                data["reward_mean"],
                label=content_type,
                marker="o",
                color=color,
                linewidth=2,
                markersize=8,
            )
    
    ax.set_xlabel("Target Length (chars)")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Reward vs Target Length\n(frag=20, by content type)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)


def plot_reward_vs_fragmentation(ax: plt.Axes, df: pd.DataFrame):
    """Plot 3: Reward vs mean_fragment_length (line plot)."""
    # Filter to target_length=500, show all content types
    filtered = df[df["target_length"] == 500]
    
    content_types = ["words", "json", "csv", "codes", "mixed"]
    colors = sns.color_palette("husl", len(content_types))
    
    for content_type, color in zip(content_types, colors):
        data = filtered[filtered["content_type"] == content_type].copy()
        # Sort by fragment length, putting None first
        data["sort_key"] = data["mean_fragment_length"].fillna(-1)
        data = data.sort_values("sort_key")
        
        if len(data) > 0:
            # Use frag_label for x-axis
            x_positions = range(len(data))
            ax.plot(
                x_positions,
                data["reward_mean"],
                label=content_type,
                marker="o",
                color=color,
                linewidth=2,
                markersize=8,
            )
            
            # Set x-tick labels only once
            if content_type == content_types[0]:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(data["frag_label"], rotation=45)
    
    ax.set_xlabel("Mean Fragment Length")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Reward vs Fragment Length\n(len=500, by content type)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)


def plot_heatmap(ax: plt.Axes, df: pd.DataFrame):
    """Plot 4: Heatmap of reward by content_type × fragment_length."""
    # Filter to target_length=500
    filtered = df[df["target_length"] == 500].copy()
    
    # Create pivot table
    pivot = filtered.pivot_table(
        values="reward_mean",
        index="content_type",
        columns="frag_label",
        aggfunc="mean",
    )
    
    # Reorder rows and columns
    row_order = ["words", "json", "csv", "codes", "mixed"]
    col_order = ["None"] + [str(x) for x in sorted([int(c) for c in pivot.columns if c != "None"])]
    
    pivot = pivot.reindex(index=row_order, columns=[c for c in col_order if c in pivot.columns])
    
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
    
    ax.set_xlabel("Mean Fragment Length")
    ax.set_ylabel("Content Type")
    ax.set_title("Reward Heatmap\n(len=500)")


def plot_distribution(ax: plt.Axes, df: pd.DataFrame):
    """Plot 5: Distribution of rewards across all configs."""
    # Create violin plot by content type
    content_types = ["words", "json", "csv", "codes", "mixed"]
    colors = sns.color_palette("husl", len(content_types))
    
    data_for_plot = []
    for content_type in content_types:
        rewards = df[df["content_type"] == content_type]["reward_mean"].values
        data_for_plot.append(rewards)
    
    parts = ax.violinplot(data_for_plot, positions=range(len(content_types)), showmeans=True)
    
    # Color the violins
    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Style the other parts
    for partname in ["cbars", "cmins", "cmaxes", "cmeans"]:
        parts[partname].set_color("black")
    
    ax.set_xticks(range(len(content_types)))
    ax.set_xticklabels(content_types)
    ax.set_xlabel("Content Type")
    ax.set_ylabel("Reward (Exact Match)")
    ax.set_title("Reward Distribution by Content Type\n(all configs)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)


def plot_char_vs_exact(ax: plt.Axes, df: pd.DataFrame):
    """Plot 6: Char accuracy vs Exact match scatter."""
    content_types = ["words", "json", "csv", "codes", "mixed"]
    colors = sns.color_palette("husl", len(content_types))
    
    for content_type, color in zip(content_types, colors):
        data = df[df["content_type"] == content_type]
        ax.scatter(
            data["exact_match_mean"],
            data["char_accuracy_mean"],
            label=content_type,
            color=color,
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )
    
    # Add diagonal line (perfect correlation)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    
    ax.set_xlabel("Exact Match (Reward)")
    ax.set_ylabel("Character Accuracy")
    ax.set_title("Char Accuracy vs Exact Match\n(all configs)")
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


def create_plots(df: pd.DataFrame, output_path: Path | None = None):
    """Create the 2x3 grid of plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Top row: Individual metrics
    plot_reward_vs_content_type(axes[0, 0], df)
    plot_reward_vs_length(axes[0, 1], df)
    plot_reward_vs_fragmentation(axes[0, 2], df)
    
    # Bottom row: Aggregations
    plot_heatmap(axes[1, 0], df)
    plot_distribution(axes[1, 1], df)
    plot_char_vs_exact(axes[1, 2], df)
    
    plt.suptitle("Verbatim Copy Ablation Results", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot verbatim_copy ablation results")
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
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run aggregate_results.py first to generate the CSV.")
        return
    
    df = load_data(args.input)
    print(f"Loaded {len(df)} configurations from {args.input}")
    
    create_plots(df, args.output)


if __name__ == "__main__":
    main()

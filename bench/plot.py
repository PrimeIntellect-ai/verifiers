"""Plot a runtime benchmark (per-cell bench_aggregate.py output).

    uv run --with matplotlib python bench/plot.py [results_dir] [out.png]

Reads one JSON per cell from `<results_dir>` (default bench/results/single_turn). A 2x2 of
the rollout stages — setup (provisioning), generation, scoring, total — each a per-runtime
grouped bar at p50 with a p10..p90 whisker, grouped by rollout count. The distribution (not
the straggler-gated e2e) is the honest comparator; the panels show where each runtime spends
its time and how that scales with concurrency. PNG is committed (the gitignore excepts it).
"""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pct(xs: list[float], p: float) -> float:
    if not xs:  # absent metric (e.g. a cell not re-run for this stage) -> skip the bar
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


src = sys.argv[1] if len(sys.argv) > 1 else "bench/results/single_turn"
out = sys.argv[2] if len(sys.argv) > 2 else src.rstrip("/") + ".png"
cells = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(src, "*.json")))]
if not cells:
    sys.exit(f"no cell JSON files in {src}")

runtimes = sorted({c["runtime"] for c in cells})
rollouts = sorted({c["rollouts"] for c in cells})
by_key = {(c["runtime"], c["rollouts"]): c for c in cells}
stages = [
    ("setup", "setup_durations", "#d98c5f"),
    ("generation", "gen_durations", "#5f8cd9"),
    ("scoring", "scoring_durations", "#6fae6f"),
    ("total", "total_durations", "#8a6fae"),
]
shades = {
    r: a for r, a in zip(rollouts, [0.55, 1.0])
}  # lighter bar for the smaller group

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
x = list(range(len(runtimes)))
width = 0.8 / len(rollouts)
for ax, (title, field, color) in zip(axes.flat, stages):
    for j, r in enumerate(rollouts):
        offs = [i + (j - (len(rollouts) - 1) / 2) * width for i in x]
        p50 = [pct(by_key.get((rt, r), {}).get(field, []), 50) for rt in runtimes]
        p10 = [pct(by_key.get((rt, r), {}).get(field, []), 10) for rt in runtimes]
        p90 = [pct(by_key.get((rt, r), {}).get(field, []), 90) for rt in runtimes]
        yerr = [[a - b for a, b in zip(p50, p10)], [a - b for a, b in zip(p90, p50)]]
        ax.bar(
            offs,
            p50,
            width,
            yerr=yerr,
            capsize=3,
            label=f"r={r}",
            color=color,
            alpha=shades[r],
            edgecolor="black",
            linewidth=0.4,
        )
        for o, v, hi in zip(offs, p50, p90):
            if v != v:  # NaN -> no data for this cell/stage
                continue
            ax.annotate(
                f"{v:.0f}" if v >= 1 else f"{v:.1f}",
                (o, hi),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7,
                color="#333",
            )
    ax.set_title(f"{title}  (p50 bar, p10–p90 whisker)")
    ax.set_ylabel("seconds / rollout")
    ax.set_xticks(x)
    ax.set_xticklabels(runtimes)
    ax.margins(y=0.15)
    ax.legend(title="rollouts", fontsize=8)

fig.suptitle(f"runtime benchmark — {os.path.basename(src.rstrip('/'))}", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(out, dpi=120)
print(f"wrote {out}")

"""Plot a runtime benchmark (bench_aggregate.py output): setup vs generation per cell.

    uv run --with matplotlib python bench/plot.py [benchmark.json] [out.png]

Per (runtime, rollouts) cell, a stacked bar of setup p50 (runtime provisioning + serving)
and generation p50, with a whisker to the generation p90 (the straggler tail) and the error
count annotated. e2e is straggler-gated, so the per-rollout distribution is the honest
comparator; the setup vs generation split shows how much of each runtime's cost is just
provisioning. Output PNG is gitignored (regenerate from the committed JSON).
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vbench/single-turn/benchmark.json"
data = json.load(open(src))
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".png"

runs = sorted(data["runs"], key=lambda r: (r["runtime"], r["rollouts"]))
labels = [f"{r['runtime']}\nr={r['rollouts']}" for r in runs]
setup50 = [pct(r.get("setup_durations", []), 50) for r in runs]
gen50 = [pct(r["gen_durations"], 50) for r in runs]
gen90 = [pct(r["gen_durations"], 90) for r in runs]
x = list(range(len(runs)))

fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(runs)), 5))
ax.bar(x, setup50, color="#d98c5f", label="setup p50")
ax.bar(x, gen50, bottom=setup50, color="#5f8cd9", label="generation p50")
# whisker: generation p90 (straggler tail), stacked on top of setup
tops50 = [s + g for s, g in zip(setup50, gen50)]
tops90 = [s + g for s, g in zip(setup50, gen90)]
ax.vlines(x, tops50, tops90, color="black", linewidth=1)
ax.scatter(x, tops90, marker="_", s=200, color="black", label="generation p90")
for i, r in enumerate(runs):
    note = f"e2e {r['e2e_s']}s"
    if r.get("errors"):
        note += f"\n{r['errors']} err"
    ax.annotate(
        note, (i, tops90[i]), textcoords="offset points", xytext=(0, 6),
        ha="center", fontsize=8, color="#444",
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("seconds (per rollout)")
ax.set_title(f"single-turn runtime benchmark — {data['label']}")
ax.legend()
ax.margins(y=0.15)
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"wrote {out}")

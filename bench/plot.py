"""Visualize bench/benchmark.json — per-rollout generation-duration distribution (p10/p50/p90),
by runtime and by batch size. e2e wall clock is gated by the single slowest rollout
(endpoint-queue stragglers), so the distribution is the honest comparator.

    uv run --with matplotlib python bench/plot.py [benchmark.json]   # writes bench/benchmark.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "bench/benchmark.json"))
runs = {(r["runtime"], r["n"]): sorted(r["gen_durations"]) for r in data["runs"]}
runtimes = sorted(
    {rt for rt, _ in runs}, key=["subprocess", "docker", "prime", "modal"].index
)
sizes = sorted({n for _, n in runs})


def pct(rt: str, n: int, q: float) -> float:
    v = runs.get((rt, n)) or []
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

# left — by runtime: p50 line + p10–p90 band, x = batch size
for rt in runtimes:
    ax1.plot(sizes, [pct(rt, n, 0.5) for n in sizes], marker="o", label=rt)
    ax1.fill_between(
        sizes,
        [pct(rt, n, 0.1) for n in sizes],
        [pct(rt, n, 0.9) for n in sizes],
        alpha=0.12,
    )
ax1.set_xlabel("batch size (concurrent rollouts)")
ax1.set_ylabel("gen duration (s)")
ax1.set_xticks(sizes)
ax1.set_title("by runtime — p50 line, p10–p90 band")
ax1.legend()
ax1.grid(True, alpha=0.3)

# right — by batch size: grouped p50 bars with p10/p90 whiskers, x = runtime
x = np.arange(len(runtimes))
w = 0.8 / len(sizes)
for i, n in enumerate(sizes):
    p50 = [pct(rt, n, 0.5) for rt in runtimes]
    lo = [p50[j] - pct(rt, n, 0.1) for j, rt in enumerate(runtimes)]
    hi = [pct(rt, n, 0.9) - p50[j] for j, rt in enumerate(runtimes)]
    ax2.bar(x + i * w - 0.4 + w / 2, p50, w, yerr=[lo, hi], capsize=3, label=f"n={n}")
ax2.set_xticks(x)
ax2.set_xticklabels(runtimes)
ax2.set_ylabel("gen duration (s)")
ax2.set_title("by batch size — p50 bar, p10/p90 whiskers")
ax2.legend()
ax2.grid(True, axis="y", alpha=0.3)

fig.suptitle(
    f"{data['taskset']} single-turn gen duration "
    f"(max_tokens={data['max_tokens']}, multiplex={data['multiplex']})"
)
fig.tight_layout()
fig.savefig("bench/benchmark.png", dpi=130)
print("wrote bench/benchmark.png")
for rt in runtimes:
    for n in sizes:
        print(
            f"{rt:11s} n={n:<4d} p10={pct(rt, n, 0.1):.0f} p50={pct(rt, n, 0.5):.0f} p90={pct(rt, n, 0.9):.0f}"
        )

"""Plot per-rollout generation-duration distribution (min / p50 / p90) across bench runs.

    uv run --with matplotlib python bench/plot_e2e.py [run_dir ...]

e2e wall clock is gated by the single slowest rollout (endpoint-queue stragglers), so it's a
noisy comparator and the max hides the typical case. This reads each run's results.jsonl,
pools per-rollout `timing.generation.duration`, and plots min / p50 / p90 — by runtime and
by batch size. Default runs: the capped (max_tokens=2048) subprocess/docker/prime sweep;
pass run dirs to compare others. Writes bench/gen_by_runtime.png + bench/gen_by_batchsize.png.
"""

import collections
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

dirs = sys.argv[1:] or sorted(
    d
    for rt in ("subprocess", "docker", "prime")
    for n in (32, 64, 128)
    for d in glob.glob(f"/tmp/vbench/{rt}-{n}-t2048-r*")
)

gens: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
for d in dirs:
    base = d.rstrip("/").split("/")[-1]
    parts = base.split("-")
    rt, n = parts[0], int(parts[1])
    for line in open(f"{d}/run/results.jsonl"):
        g = json.loads(line).get("timing", {}).get("generation", {}).get("duration")
        if g is not None:
            gens[(rt, n)].append(g)

runtimes = sorted({rt for rt, _ in gens}, key=["subprocess", "docker", "prime", "modal"].index)
sizes = sorted({n for _, n in gens})


def pct(rt: str, n: int, q: float) -> float:
    v = sorted(gens[(rt, n)])
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


# view 1 — by runtime: p50 line + min..p90 band per runtime, x = batch size
fig, ax = plt.subplots(figsize=(7, 4.5))
for rt in runtimes:
    lo = [pct(rt, n, 0.0) for n in sizes]
    md = [pct(rt, n, 0.5) for n in sizes]
    hi = [pct(rt, n, 0.9) for n in sizes]
    line, = ax.plot(sizes, md, marker="o", label=rt)
    ax.fill_between(sizes, lo, hi, alpha=0.15, color=line.get_color())
ax.set_xlabel("batch size (num_tasks)")
ax.set_ylabel("gen duration (s)")
ax.set_xticks(sizes)
ax.set_title("gsm8k-v1 gen duration by runtime — p50 line, min–p90 band (max_tokens=2048)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("bench/gen_by_runtime.png", dpi=130)

# view 2 — by batch size: grouped p50 bars with min/p90 whiskers, x = runtime
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(runtimes))
w = 0.8 / len(sizes)
for i, n in enumerate(sizes):
    md = [pct(rt, n, 0.5) for rt in runtimes]
    err_lo = [md[j] - pct(rt, n, 0.0) for j, rt in enumerate(runtimes)]
    err_hi = [pct(rt, n, 0.9) - md[j] for j, rt in enumerate(runtimes)]
    ax.bar(x + i * w - 0.4 + w / 2, md, w, yerr=[err_lo, err_hi], capsize=3, label=f"n={n}")
ax.set_xticks(x)
ax.set_xticklabels(runtimes)
ax.set_ylabel("gen duration (s)")
ax.set_title("gsm8k-v1 gen duration by batch — p50 bar, min/p90 whiskers (max_tokens=2048)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("bench/gen_by_batchsize.png", dpi=130)

print("wrote bench/gen_by_runtime.png + bench/gen_by_batchsize.png")
for rt, n in sorted(gens):
    print(
        f"{rt:11s} n={n:<4d} min={pct(rt, n, 0.0):.1f} p50={pct(rt, n, 0.5):.1f} "
        f"p90={pct(rt, n, 0.9):.1f}  (n_roll={len(gens[(rt, n)])})"
    )

"""Plot e2e wall-clock from bench BENCH lines: two views (by runtime, by batch size).

    uv run --with matplotlib python bench/plot_e2e.py [bench_log]

Parses `BENCH runtime=<rt> n=<N> ... elapsed_s=<s>` lines (default /tmp/vbench/capped2048.log),
averages repeated runs per (runtime, N), and writes bench/e2e_by_runtime.png + bench/e2e_by_batchsize.png.
"""

import collections
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vbench/capped2048.log"
pat = re.compile(r"BENCH runtime=(\S+) n=(\d+).*?elapsed_s=(\d+)")

runs = collections.defaultdict(list)  # (runtime, n) -> [elapsed_s, ...]
for line in open(log):
    m = pat.search(line)
    if m:
        runs[(m.group(1), int(m.group(2)))].append(int(m.group(3)))

runtimes = sorted(
    {rt for rt, _ in runs}, key=["subprocess", "docker", "prime", "modal"].index
)
sizes = sorted({n for _, n in runs})


def mean(rt, n):
    return sum(runs[(rt, n)]) / len(runs[(rt, n)]) if runs.get((rt, n)) else None


# view 1 — grouped by runtime: x = batch size, one line per runtime
fig, ax = plt.subplots(figsize=(7, 4.5))
for rt in runtimes:
    ax.plot(sizes, [mean(rt, n) for n in sizes], marker="o", label=rt)
ax.set_xlabel("batch size (num_tasks)")
ax.set_ylabel("e2e wall clock (s)")
ax.set_xticks(sizes)
ax.set_title("gsm8k-v1 e2e by runtime (max_tokens=2048, mean of 2 runs)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("bench/e2e_by_runtime.png", dpi=130)

# view 2 — grouped by batch size: x = runtime, bars grouped by batch size
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(runtimes))
w = 0.8 / len(sizes)
for i, n in enumerate(sizes):
    ax.bar(x + i * w - 0.4 + w / 2, [mean(rt, n) for rt in runtimes], w, label=f"n={n}")
ax.set_xticks(x)
ax.set_xticklabels(runtimes)
ax.set_ylabel("e2e wall clock (s)")
ax.set_title("gsm8k-v1 e2e by batch size (max_tokens=2048, mean of 2 runs)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("bench/e2e_by_batchsize.png", dpi=130)

print("wrote bench/e2e_by_runtime.png + bench/e2e_by_batchsize.png")
for (rt, n), v in sorted(runs.items()):
    print(f"{rt:11s} n={n:<4d} runs={v} mean={sum(v) / len(v):.1f}")

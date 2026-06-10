"""Plot interception multiplexing: baseline (tunnel/rollout) vs --multiplex 32.

    uv run --with matplotlib python bench/plot_mux.py

Reads the prime gsm8k-v1 runs (capped max_tokens=2048) at batch 32/64/128/256 for both
variants and plots per-rollout generation-duration p50 (line) + min–p90 (band) — e2e is
gated by endpoint-queue stragglers, so the distribution is the honest comparator.
Writes bench/mux_vs_base.png.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = [32, 64, 128, 256]


def gens(variant: str, n: int) -> list[float]:
    path = f"/tmp/vbench/prime-{n}-t2048-{variant}/run/results.jsonl"
    return sorted(
        json.loads(line)["timing"]["generation"]["duration"] for line in open(path)
    )


def pct(v: list[float], q: float) -> float:
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


fig, ax = plt.subplots(figsize=(7.5, 4.5))
for variant, label, color in [
    ("base", "baseline (1 tunnel / rollout)", "#d62728"),
    ("mux32", "multiplex = 32", "#2ca02c"),
]:
    data = {n: gens(variant, n) for n in SIZES}
    p50 = [pct(data[n], 0.5) for n in SIZES]
    lo = [pct(data[n], 0.0) for n in SIZES]
    hi = [pct(data[n], 0.9) for n in SIZES]
    ax.plot(SIZES, p50, marker="o", color=color, label=label)
    ax.fill_between(SIZES, lo, hi, alpha=0.15, color=color)

ax.set_xlabel("batch size (concurrent rollouts)")
ax.set_ylabel("gen duration (s)")
ax.set_xticks(SIZES)
ax.set_title(
    "prime gsm8k-v1 — interception multiplexing\n"
    "p50 line, min–p90 band (max_tokens=2048, 512 tunnel limiter)"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("bench/mux_vs_base.png", dpi=130)
print("wrote bench/mux_vs_base.png")
for variant in ("base", "mux32"):
    for n in SIZES:
        v = gens(variant, n)
        print(
            f"{variant:6s} n={n:<4d} min={pct(v, 0.0):.0f} p50={pct(v, 0.5):.0f} p90={pct(v, 0.9):.0f}"
        )

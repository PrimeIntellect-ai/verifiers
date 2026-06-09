"""Summarize a benchmark run's results.jsonl into one row of stats.

    python bench/summarize.py <results.jsonl> --elapsed <wall_clock_s> [--runtime r] [--n N]

Reports the end-to-end wall clock alongside per-rollout generation-duration
percentiles and error/truncation counts, so a slow runtime (provisioning
overhead) is distinguishable from a slow model (generation latency).
"""

import argparse
import json
import statistics


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--elapsed", type=float, required=True)
    ap.add_argument("--runtime", default="?")
    ap.add_argument("--n", default="?")
    args = ap.parse_args()

    traces = [json.loads(line) for line in open(args.results) if line.strip()]
    gen = [t["timing"]["generation"]["duration"] for t in traces]
    rewards = [t.get("reward", 0.0) for t in traces]
    errors = sum(1 for t in traces if t.get("errors"))
    truncated = sum(1 for t in traces if t.get("is_truncated"))

    print(
        f"runtime={args.runtime} n={args.n} rollouts={len(traces)} "
        f"e2e_s={args.elapsed:.0f} "
        f"throughput_rps={len(traces) / args.elapsed:.2f} "
        f"gen_min={min(gen):.1f} gen_p50={statistics.median(gen):.1f} "
        f"gen_p95={pct(gen, 95):.1f} gen_max={max(gen):.1f} "
        f"reward_mean={statistics.mean(rewards):.3f} "
        f"errors={errors} truncated={truncated}"
    )


if __name__ == "__main__":
    main()

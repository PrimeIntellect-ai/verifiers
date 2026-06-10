"""Aggregate a single-turn benchmark run into benchmark.json — the input bench/plot.py reads.

    uv run python bench/aggregate.py <out_dir> <taskset> <max_tokens>   > bench/benchmark.json

`out_dir` holds one `<runtime>-<n>/` eval output dir per (runtime, batch size) plus `e2e.txt`
(`<runtime> <n> <e2e_seconds>` lines). For each run we record the e2e wall clock, the full
per-rollout `generation.duration` list (so the plot can take p10/p50/p90 — e2e is gated by
endpoint-queue stragglers, so the distribution is the honest comparator), reward, and error
count. `multiplex` + `model` are read off a run's resolved config.toml.
"""

import glob
import json
import os
import sys
import tomllib

out, taskset, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])

e2e: dict[tuple[str, int], int] = {}
for line in open(os.path.join(out, "e2e.txt")):
    rt, n, secs = line.split()
    e2e[(rt, int(n))] = int(secs)

multiplex: int | None = None
model: str | None = None
runs = []
for d in sorted(p for p in glob.glob(f"{out}/*") if os.path.isdir(p)):
    rt, n = os.path.basename(d).rsplit("-", 1)
    rows = [json.loads(line) for line in open(f"{d}/results.jsonl")]
    gens = sorted(r["timing"]["generation"]["duration"] for r in rows)
    rewards = [r["reward"] for r in rows if r.get("reward") is not None]
    if multiplex is None:
        cfg = tomllib.load(open(f"{d}/config.toml", "rb"))
        multiplex, model = cfg.get("multiplex"), cfg.get("model")
    runs.append(
        {
            "runtime": rt,
            "n": int(n),
            "e2e_s": e2e.get((rt, int(n))),
            "reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
            "errors": sum(1 for r in rows if r.get("errors")),
            "gen_durations": [round(g, 3) for g in gens],
        }
    )

print(
    json.dumps(
        {
            "taskset": taskset,
            "model": model,
            "max_tokens": max_tokens,
            "multiplex": multiplex,
            "runs": runs,
        },
        indent=2,
    )
)

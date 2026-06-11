"""Aggregate a worker-pool benchmark run (single-turn or agentic) into JSON.

    python bench/bench_aggregate.py <out_dir> <label>

`out_dir` holds one `w<workers>-r<rollouts>/` eval output dir per (workers, group size)
plus `e2e.txt` (`<workers> <rollouts> <e2e_seconds>` lines). For each run we record the
e2e wall clock, the full per-rollout `generation.duration` list (so p10/p50/p90 — e2e is
straggler-gated, so the distribution is the honest comparator), reward, and error count.
`workers=0` is the single in-process server; `>0` the worker pool. `label` is free-text
run metadata (e.g. the taskset + runtime).
"""

import glob
import json
import os
import sys

out, label = sys.argv[1], sys.argv[2]

e2e: dict[tuple[int, int], int] = {}
for line in open(os.path.join(out, "e2e.txt")):
    w, r, secs = line.split()
    e2e[(int(w), int(r))] = int(secs)

runs = []
for d in sorted(p for p in glob.glob(f"{out}/*") if os.path.isdir(p)):
    base = os.path.basename(d)  # w<workers>-r<rollouts>
    workers = int(base.split("-")[0][1:])
    rollouts = int(base.split("-")[1][1:])
    rows = [json.loads(line) for line in open(f"{d}/results.jsonl")]
    gens = sorted(r["timing"]["generation"]["duration"] for r in rows)
    rewards = [r["reward"] for r in rows if r.get("reward") is not None]
    runs.append(
        {
            "workers": workers,
            "rollouts": rollouts,
            "e2e_s": e2e.get((workers, rollouts)),
            "reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
            "errors": sum(1 for r in rows if r.get("errors")),
            "gen_durations": [round(g, 3) for g in gens],
        }
    )

print(json.dumps({"label": label, "runs": runs}, indent=2))

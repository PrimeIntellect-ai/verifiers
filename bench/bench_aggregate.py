"""Aggregate a runtime benchmark run (single-turn or agentic) into JSON.

    python bench/bench_aggregate.py <out_dir> <label>

`out_dir` holds one `<runtime>-r<rollouts>/` eval output dir per (runtime, group size)
plus `e2e.txt` (`<runtime> <rollouts> <e2e_seconds>` lines). For each run we record the
e2e wall clock, the full per-rollout `setup.duration` and `generation.duration` lists (so
p10/p50/p90 — e2e is straggler-gated, so the distribution is the honest comparator), reward,
and error count. `setup` (runtime provisioning + serving) separates provisioning overhead
from generation in the runtime matrix; it reads 0 on traces that don't record a setup span.
`label` is free-text run metadata (e.g. the taskset / task).
"""

import glob
import json
import os
import sys

out, label = sys.argv[1], sys.argv[2]

e2e: dict[tuple[str, int], int] = {}
for line in open(os.path.join(out, "e2e.txt")):
    rt, r, secs = line.split()
    e2e[(rt, int(r))] = int(secs)

runs = []
for d in sorted(p for p in glob.glob(f"{out}/*") if os.path.isdir(p)):
    base = os.path.basename(d)  # <runtime>-r<rollouts>
    runtime, _, rollouts = base.rpartition("-r")
    rollouts = int(rollouts)
    rows = [json.loads(line) for line in open(f"{d}/results.jsonl")]
    setups = sorted(r["timing"].get("setup", {}).get("duration", 0.0) for r in rows)
    gens = sorted(r["timing"]["generation"]["duration"] for r in rows)
    rewards = [r["reward"] for r in rows if r.get("reward") is not None]
    runs.append(
        {
            "runtime": runtime,
            "rollouts": rollouts,
            "e2e_s": e2e.get((runtime, rollouts)),
            "reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
            "errors": sum(1 for r in rows if r.get("errors")),
            "setup_durations": [round(s, 3) for s in setups],
            "gen_durations": [round(g, 3) for g in gens],
        }
    )

print(json.dumps({"label": label, "runs": runs}, indent=2))

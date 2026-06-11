"""Aggregate a benchmark run into per-cell JSON — one file per matrix element.

    python bench/bench_aggregate.py <work_dir> <results_dir>

`<work_dir>` holds one `<runtime>-r<rollouts>/` eval output dir per cell run this time, plus
`e2e.txt` (`<runtime> <rollouts> <e2e_seconds>` lines). For each cell we write
`<results_dir>/<runtime>-r<rollouts>.json` with the e2e wall clock, the per-rollout
`setup.duration` + `generation.duration` lists (so p10/p50/p90 — e2e is straggler-gated, so
the distribution is the honest comparator), the resolved `max_concurrent`, reward and error
count. One file per cell means re-running a subset (e.g. just docker) refreshes only those
cells and never clobbers the rest. `bench/plot.py` reads the whole results dir.
"""

import glob
import json
import os
import sys
import tomllib

work, results = sys.argv[1], sys.argv[2]
os.makedirs(results, exist_ok=True)

e2e: dict[tuple[str, int], int] = {}
for line in open(os.path.join(work, "e2e.txt")):
    rt, r, secs = line.split()
    e2e[(rt, int(r))] = int(secs)

for d in sorted(p for p in glob.glob(f"{work}/*") if os.path.isdir(p)):
    base = os.path.basename(d)  # <runtime>-r<rollouts>
    runtime, _, rollouts = base.rpartition("-r")
    rollouts = int(rollouts)
    rows = [json.loads(line) for line in open(f"{d}/results.jsonl")]
    rewards = [r["reward"] for r in rows if r.get("reward") is not None]
    cfg = {}
    if os.path.exists(f"{d}/config.toml"):
        with open(f"{d}/config.toml", "rb") as f:
            cfg = tomllib.load(f)
    cell = {
        "runtime": runtime,
        "rollouts": rollouts,
        "max_concurrent": cfg.get("max_concurrent"),  # absent in the dump => unbounded
        "e2e_s": e2e.get((runtime, rollouts)),
        "reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
        "errors": sum(1 for r in rows if r.get("errors")),
        "setup_durations": sorted(
            round(r["timing"].get("setup", {}).get("duration", 0.0), 3) for r in rows
        ),
        "gen_durations": sorted(
            round(r["timing"]["generation"]["duration"], 3) for r in rows
        ),
    }
    out = os.path.join(results, f"{base}.json")
    with open(out, "w") as f:
        json.dump(cell, f, indent=2)
    print(f"wrote {out}")

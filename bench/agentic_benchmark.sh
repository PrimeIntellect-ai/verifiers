#!/usr/bin/env bash
# Agentic runtime benchmark: run ONE harbor task at group sizes (-r) across container
# runtimes (docker / prime) and write per-rollout durations + e2e wall clock, which
# bench/bench_aggregate.py summarizes. Each rollout is its own container/sandbox (a coding
# agent + the harbor verifier); rollouts are independent, so this stresses concurrent
# agentic execution + scoring across the two runtimes.
#
#   bench/agentic_benchmark.sh
#   ROLLOUTS="64 512" RUNTIMES="docker prime" TASK=fix-git bench/agentic_benchmark.sh
#
# Needs the `harbor` CLI (`uv tool install harbor`) and the `terminal-bench-2-v1` example
# taskset (an editable dep); docker needs the daemon, prime needs PRIME_API_KEY in ~/.env.
# Concurrency is unbounded (every rollout in flight at once).
set -uo pipefail

TASKSET="${TASKSET:-terminal-bench-2-v1}"
TASK="${TASK:-fix-git}"
RUNTIMES="${RUNTIMES:-docker prime}"
ROLLOUTS="${ROLLOUTS:-64 512}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
MAX_TURNS="${MAX_TURNS:-32}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a; . "$HOME/.env" 2>/dev/null || true; set +a

OUT="/tmp/vbench/agentic"
rm -rf "$OUT"; mkdir -p "$OUT"; : > "$OUT/e2e.txt"
for rt in $RUNTIMES; do
  for r in $ROLLOUTS; do
    label="$rt-r$r"
    echo "== $label (task=$TASK max_turns=$MAX_TURNS) =="
    start=$(date +%s)
    uv run eval "$TASKSET" --taskset.tasks "[\"$TASK\"]" \
      --harness.id default --harness.enable_bash true --harness.runtime.type "$rt" \
      --num_tasks 1 --num_rollouts "$r" \
      --max_concurrent None --retry.attempts 1 --max_turns "$MAX_TURNS" \
      --rich false --output_dir "$OUT/$label" \
      > "$OUT/$label.stdout" 2> "$OUT/$label.log"
    rc=$?
    echo "$rt $r $(( $(date +%s) - start ))" >> "$OUT/e2e.txt"
    echo "rc=$rc e2e=$(tail -1 "$OUT/e2e.txt" | awk '{print $3}')s"
  done
done

# Aggregate into agentic_benchmark.json: per-(runtime, rollouts) e2e + the per-rollout
# generation-duration list (p10/p50/p90), reward, and error count.
uv run python bench/bench_aggregate.py "$OUT" "$TASK (max_turns=$MAX_TURNS)" > "$OUT/agentic_benchmark.json"
echo "wrote $OUT/agentic_benchmark.json"

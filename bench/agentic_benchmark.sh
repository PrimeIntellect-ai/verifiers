#!/usr/bin/env bash
# Agentic benchmark: run ONE harbor task at group sizes (-r) across env-server modes and
# write bench/agentic_benchmark.json (per-rollout durations + e2e wall clock), which
# bench/agentic_aggregate.py summarizes. Each rollout is its own sandbox (a coding agent +
# the harbor verifier); with no group reward the rollouts are independent, so the worker
# pool round-robins them across workers — this stresses concurrent agentic execution +
# scoring (where the single-loop server is most likely to stall).
#
#   bench/agentic_benchmark.sh
#   ROLLOUTS="8 16" WORKERS="0 4" TASK=fix-git bench/agentic_benchmark.sh
#
# Compares WORKERS modes: 0 = single in-process server, N = an N-worker pool. Needs the
# `harbor` CLI (`uv tool install harbor`) and the `terminal-bench-2-v1` example taskset
# (an editable dep), plus a container runtime (prime default; PRIME_API_KEY in ~/.env).
set -uo pipefail

TASKSET="${TASKSET:-terminal-bench-2-v1}"
TASK="${TASK:-fix-git}"
RUNTIME="${RUNTIME:-prime}"
ROLLOUTS="${ROLLOUTS:-32 64 128}"
WORKERS="${WORKERS:-0 4}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
MAX_TURNS="${MAX_TURNS:-30}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a; . "$HOME/.env" 2>/dev/null || true; set +a

OUT="/tmp/vbench/agentic"
rm -rf "$OUT"; mkdir -p "$OUT"; : > "$OUT/e2e.txt"
for w in $WORKERS; do
  for r in $ROLLOUTS; do
    label="w$w-r$r"
    echo "== $label (task=$TASK runtime=$RUNTIME max_turns=$MAX_TURNS) =="
    start=$(date +%s)
    uv run eval "$TASKSET" --taskset.tasks "[\"$TASK\"]" \
      --harness.id default --harness.enable_bash true --harness.runtime.type "$RUNTIME" \
      --num_tasks 1 --num_rollouts "$r" --num_workers "$w" \
      --max_concurrent 512 --retry.attempts 1 --max_turns "$MAX_TURNS" \
      --rich false --output_dir "$OUT/$label" \
      > "$OUT/$label.stdout" 2> "$OUT/$label.log"
    rc=$?
    echo "$w $r $(( $(date +%s) - start ))" >> "$OUT/e2e.txt"
    echo "rc=$rc e2e=$(tail -1 "$OUT/e2e.txt" | awk '{print $3}')s"
  done
done

# Aggregate into bench/agentic_benchmark.json: per-(workers, rollouts) e2e + the per-rollout
# generation-duration list (p10/p50/p90), reward, and error count.
uv run python bench/agentic_aggregate.py "$OUT" "$TASKSET/$TASK" "$MAX_TURNS" > bench/agentic_benchmark.json
echo "wrote bench/agentic_benchmark.json"

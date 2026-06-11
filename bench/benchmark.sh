#!/usr/bin/env bash
# Single-turn pool benchmark: run gsm8k-v1 at group sizes (-r, rollouts of ONE task) across
# env-server modes (in-process vs N-worker pool) and write per-rollout durations + e2e, which
# bench/bench_aggregate.py summarizes. The single-turn counterpart to agentic_benchmark.sh:
# per-rollout CPU is light (no agent loop, no verifier), so this is where the pool's fixed
# per-worker overhead is most visible against its event-loop relief.
#
#   bench/benchmark.sh
#   ROLLOUTS="32 64 128" WORKERS="0 4" RUNTIME=subprocess MAX_TOKENS=1024 bench/benchmark.sh
#
# Compares WORKERS modes (0 = single in-process server, N = an N-worker pool), concurrency
# capped at 512, default model. RUNTIME defaults to subprocess (no sandbox provisioning).
set -uo pipefail

TASKSET="${TASKSET:-gsm8k-v1}"
RUNTIME="${RUNTIME:-subprocess}"
ROLLOUTS="${ROLLOUTS:-32 64 128}"
WORKERS="${WORKERS:-0 4}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a; . "$HOME/.env" 2>/dev/null || true; set +a

OUT="/tmp/vbench/single-turn"
rm -rf "$OUT"; mkdir -p "$OUT"; : > "$OUT/e2e.txt"
for w in $WORKERS; do
  for r in $ROLLOUTS; do
    label="w$w-r$r"
    echo "== $label (runtime=$RUNTIME max_tokens=$MAX_TOKENS) =="
    start=$(date +%s)
    uv run eval "$TASKSET" --harness.id default --harness.enable_bash false \
      --harness.runtime.type "$RUNTIME" --num_tasks 1 --num_rollouts "$r" --num_workers "$w" \
      --max_concurrent 512 --retry.attempts 1 --sampling.max_tokens "$MAX_TOKENS" \
      -m "$MODEL" --rich false --output_dir "$OUT/$label" \
      > "$OUT/$label.stdout" 2> "$OUT/$label.log"
    rc=$?
    echo "$w $r $(( $(date +%s) - start ))" >> "$OUT/e2e.txt"
    echo "rc=$rc e2e=$(tail -1 "$OUT/e2e.txt" | awk '{print $3}')s"
  done
done

# Aggregate into benchmark.json: per-(workers, rollouts) e2e + the per-rollout
# generation-duration list (p10/p50/p90), reward, and error count.
uv run python bench/bench_aggregate.py "$OUT" "$TASKSET ($RUNTIME)" > "$OUT/benchmark.json"
echo "wrote $OUT/benchmark.json"

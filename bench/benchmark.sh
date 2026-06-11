#!/usr/bin/env bash
# Single-turn runtime benchmark: run gsm8k-v1 at group sizes (-r, rollouts of ONE task)
# across runtimes (subprocess / docker / prime) and write per-rollout durations + e2e, which
# bench/bench_aggregate.py summarizes. Single-turn per-rollout CPU is light (no agent loop,
# just a fast integer verify), so this isolates each runtime's provisioning + round-trip
# overhead under concurrency.
#
#   bench/benchmark.sh
#   ROLLOUTS="64 512" RUNTIMES="subprocess docker prime" bench/benchmark.sh
#
# Concurrency and generation length are both unbounded (rollouts stop on EOS); default model.
set -uo pipefail

TASKSET="${TASKSET:-gsm8k-v1}"
RUNTIMES="${RUNTIMES:-subprocess docker prime}"
ROLLOUTS="${ROLLOUTS:-64 512}"
MAX_TOKENS="${MAX_TOKENS:-}"  # empty = unbounded generation (gsm8k answers stop on EOS)
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Creds for the model endpoint (+ prime runtime); skip the FIFO read if already in the env.
[ -n "${PRIME_API_KEY:-}" ] || { set -a; . "$HOME/.env" 2>/dev/null || true; set +a; }

SAMPLING=()
[ -n "$MAX_TOKENS" ] && SAMPLING+=(--sampling.max_tokens "$MAX_TOKENS")

OUT="/tmp/vbench/single-turn"
rm -rf "$OUT"; mkdir -p "$OUT"; : > "$OUT/e2e.txt"
for rt in $RUNTIMES; do
  for r in $ROLLOUTS; do
    label="$rt-r$r"
    echo "== $label (max_tokens=${MAX_TOKENS:-unbounded}) =="
    start=$(date +%s)
    uv run eval "$TASKSET" --harness.id default --harness.enable_bash false \
      --harness.runtime.type "$rt" --num_tasks 1 --num_rollouts "$r" \
      --max_concurrent None \
      --retries.rollout.max_retries 0 --retries.model.max_retries 0 --retries.runtime.max_retries 0 \
      "${SAMPLING[@]}" \
      -m "$MODEL" --rich false --output_dir "$OUT/$label" \
      > "$OUT/$label.stdout" 2> "$OUT/$label.log"
    rc=$?
    echo "$rt $r $(( $(date +%s) - start ))" >> "$OUT/e2e.txt"
    echo "rc=$rc e2e=$(tail -1 "$OUT/e2e.txt" | awk '{print $3}')s"
  done
done

# Aggregate into benchmark.json: per-(runtime, rollouts) e2e + the per-rollout
# generation-duration list (p10/p50/p90), reward, and error count.
uv run python bench/bench_aggregate.py "$OUT" "$TASKSET" > "$OUT/benchmark.json"
echo "wrote $OUT/benchmark.json"

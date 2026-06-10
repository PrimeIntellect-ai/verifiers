#!/usr/bin/env bash
# Single-turn runtime benchmark: run gsm8k-v1 across runtimes x batch sizes and write
# bench/benchmark.json (per-rollout generation durations + e2e wall clock), which
# bench/plot.py visualizes. Measures the overhead each runtime adds, with multiplexing at
# its default (interception servers/tunnels shared across rollouts).
#
#   bench/benchmark.sh
#   RUNTIMES="subprocess prime" SIZES="32 256" MAX_TOKENS=2048 bench/benchmark.sh
#
# One rollout per task (-r 1), concurrency capped at 512, default model + multiplex. docker
# needs the `docker` group active, so the whole script re-execs once under `sg docker`.
set -uo pipefail

RUNTIMES="${RUNTIMES:-subprocess docker prime}"
SIZES="${SIZES:-32 64 128}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TASKSET="${TASKSET:-gsm8k-v1}"

if [ -z "${_BENCH_SG:-}" ] && [[ " $RUNTIMES " == *" docker "* ]] && command -v sg >/dev/null; then
  exec sg docker -c "_BENCH_SG=1 RUNTIMES=$(printf '%q' "$RUNTIMES") SIZES=$(printf '%q' "$SIZES") MAX_TOKENS=$(printf '%q' "$MAX_TOKENS") TASKSET=$(printf '%q' "$TASKSET") $(printf '%q ' "$0")"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # the worktree this script lives in
cd "$ROOT"
set -a; . "$HOME/.env" 2>/dev/null || true; set +a

OUT="/tmp/vbench/single-turn"
rm -rf "$OUT"
mkdir -p "$OUT"
: > "$OUT/e2e.txt"
for rt in $RUNTIMES; do
  for n in $SIZES; do
    echo "== $rt n=$n (max_tokens=$MAX_TOKENS) =="
    start=$(date +%s)
    uv run eval "$TASKSET" \
      --harness.id default --harness.enable_bash false \
      --harness.runtime.type "$rt" \
      --max_concurrent 512 --num_tasks "$n" --num_rollouts 1 \
      --retry.attempts 1 --sampling.max_tokens "$MAX_TOKENS" \
      --rich false --output_dir "$OUT/$rt-$n" \
      > "$OUT/$rt-$n.stdout" 2> "$OUT/$rt-$n.log"
    rc=$?
    echo "$rt $n $(( $(date +%s) - start ))" >> "$OUT/e2e.txt"
    echo "rc=$rc elapsed=$(tail -1 "$OUT/e2e.txt" | awk '{print $3}')s"
  done
done

# Aggregate the runs into bench/benchmark.json (the plot's input): metadata + per-(runtime,n)
# e2e wall clock, the full per-rollout generation-duration list (for p10/p50/p90), reward,
# and error count. multiplex is read off a run's resolved config.
uv run python bench/aggregate.py "$OUT" "$TASKSET" "$MAX_TOKENS" > bench/benchmark.json
echo "wrote bench/benchmark.json"

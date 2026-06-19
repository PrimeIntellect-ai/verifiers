#!/usr/bin/env bash
# Single-turn runtime benchmark: run gsm8k-v1 at group sizes (-r) across runtimes
# (subprocess / docker / prime), writing ONE results file per matrix cell to
# bench/results/single_turn/<runtime>-r<rollouts>.json — so re-running a subset (e.g. just
# docker) refreshes only those cells and never clobbers the rest. Single-turn per-rollout
# CPU is light, so this isolates each runtime's provisioning (setup) + round-trip overhead.
#
#   bench/benchmark.sh
#   RUNTIMES="docker" MAX_CONCURRENT=128 bench/benchmark.sh   # add/refresh just docker cells
#
# Concurrency defaults to unbounded; cap it with MAX_CONCURRENT (e.g. for docker, whose
# per-container cold-install is disk-heavy at scale). gsm8k generation is unbounded (EOS).
set -uo pipefail

TASKSET="${TASKSET:-gsm8k-v1}"
RUNTIMES="${RUNTIMES:-subprocess docker prime}"
ROLLOUTS="${ROLLOUTS:-64 512}"
MAX_TOKENS="${MAX_TOKENS:-}"            # empty = unbounded generation (gsm8k stops on EOS)
MAX_CONCURRENT="${MAX_CONCURRENT:-None}"  # None = unbounded rollouts in flight
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Creds for the model endpoint (+ prime runtime); skip the FIFO read if already in the env.
[ -n "${PRIME_API_KEY:-}" ] || { set -a; . "$HOME/.env" 2>/dev/null || true; set +a; }

SAMPLING=()
[ -n "$MAX_TOKENS" ] && SAMPLING+=(--sampling.max_tokens "$MAX_TOKENS")

WORK="/tmp/vbench/single-turn"           # scratch (raw eval outputs), wiped each run
RESULTS="$ROOT/bench/results/single_turn"  # committed per-cell results, NOT wiped
rm -rf "$WORK"; mkdir -p "$WORK" "$RESULTS"; : > "$WORK/e2e.txt"
for rt in $RUNTIMES; do
  for r in $ROLLOUTS; do
    label="$rt-r$r"
    echo "== $label (max_concurrent=$MAX_CONCURRENT max_tokens=${MAX_TOKENS:-unbounded}) =="
    start=$(date +%s)
    uv run eval "$TASKSET" --harness.id default --harness.enable_bash false \
      --harness.runtime.type "$rt" --num_tasks 1 --num_rollouts "$r" \
      --max_concurrent "$MAX_CONCURRENT" \
      --retries.rollout.max_retries 0 --retries.model.max_retries 0 --retries.runtime.max_retries 0 \
      "${SAMPLING[@]}" \
      -m "$MODEL" --rich false --output_dir "$WORK/$label" \
      > "$WORK/$label.stdout" 2> "$WORK/$label.log"
    rc=$?
    echo "$rt $r $(( $(date +%s) - start ))" >> "$WORK/e2e.txt"
    echo "rc=$rc e2e=$(tail -1 "$WORK/e2e.txt" | awk '{print $3}')s"
  done
done

# One JSON per cell into the committed results dir (a subset run won't clobber other cells).
uv run python bench/bench_aggregate.py "$WORK" "$RESULTS"
echo "wrote per-cell results to $RESULTS"

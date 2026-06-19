#!/usr/bin/env bash
# Agentic runtime benchmark: run ONE harbor task at group sizes (-r) across container
# runtimes (docker / prime), writing ONE results file per matrix cell to
# bench/results/agentic/<runtime>-r<rollouts>.json — so re-running a subset refreshes only
# those cells. Each rollout is its own container/sandbox (a coding agent + the harbor
# verifier); rollouts are independent, so this stresses concurrent agentic execution + scoring.
#
#   bench/agentic_benchmark.sh
#   RUNTIMES="docker" MAX_CONCURRENT=64 TASK=fix-git bench/agentic_benchmark.sh
#
# Needs the `harbor` CLI (`uv tool install harbor`) and the `terminal-bench-2-v1` example
# taskset (an editable dep); docker needs the daemon, prime needs PRIME_API_KEY in ~/.env.
# Concurrency defaults to unbounded; cap it with MAX_CONCURRENT.
set -uo pipefail

TASKSET="${TASKSET:-terminal-bench-2-v1}"
TASK="${TASK:-fix-git}"
RUNTIMES="${RUNTIMES:-docker prime}"
ROLLOUTS="${ROLLOUTS:-64 512}"
MAX_CONCURRENT="${MAX_CONCURRENT:-None}"  # None = unbounded rollouts in flight
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
MAX_TURNS="${MAX_TURNS:-32}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Creds for the model endpoint (+ prime runtime); skip the FIFO read if already in the env.
[ -n "${PRIME_API_KEY:-}" ] || { set -a; . "$HOME/.env" 2>/dev/null || true; set +a; }

WORK="/tmp/vbench/agentic"            # scratch (raw eval outputs), wiped each run
RESULTS="$ROOT/bench/results/agentic"   # committed per-cell results, NOT wiped
rm -rf "$WORK"; mkdir -p "$WORK" "$RESULTS"; : > "$WORK/e2e.txt"
for rt in $RUNTIMES; do
  for r in $ROLLOUTS; do
    label="$rt-r$r"
    echo "== $label (task=$TASK max_concurrent=$MAX_CONCURRENT max_turns=$MAX_TURNS) =="
    start=$(date +%s)
    uv run eval "$TASKSET" --taskset.tasks "[\"$TASK\"]" \
      --harness.id bash --harness.runtime.type "$rt" \
      --num_tasks 1 --num_rollouts "$r" \
      --max_concurrent "$MAX_CONCURRENT" --max_turns "$MAX_TURNS" \
      --retries.rollout.max_retries 0 \
      --rich false --output_dir "$WORK/$label" \
      > "$WORK/$label.stdout" 2> "$WORK/$label.log"
    rc=$?
    echo "$rt $r $(( $(date +%s) - start ))" >> "$WORK/e2e.txt"
    echo "rc=$rc e2e=$(tail -1 "$WORK/e2e.txt" | awk '{print $3}')s"
  done
done

# One JSON per cell into the committed results dir (a subset run won't clobber other cells).
uv run python bench/bench_aggregate.py "$WORK" "$RESULTS"
echo "wrote per-cell results to $RESULTS"

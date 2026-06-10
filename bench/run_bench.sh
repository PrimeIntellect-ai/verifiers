#!/usr/bin/env bash
# Benchmark one (runtime, scale) eval run of gsm8k-v1 and report wall-clock e2e.
#
#   bench/run_bench.sh <subprocess|docker|prime|modal> <num_tasks> [label]
#   MAX_TOKENS=2048 bench/run_bench.sh <runtime> <num_tasks>   # cap generation length
#   MULTIPLEX=32 bench/run_bench.sh <runtime> <num_tasks>      # share interception servers/tunnels
#
# One rollout per task (-r 1), concurrency capped at 512, default model. MAX_TOKENS caps
# per-generation length (--sampling.max_tokens); MULTIPLEX shares one interception server +
# tunnel per that many rollouts (--multiplex). Each adds a label suffix. Raw results + logs
# land under /tmp/vbench/<label>; the BENCH line carries the end-to-end wall clock. Runs the
# eval from the worktree this script lives in. The docker CLI needs the `docker` group
# active, so the docker runtime re-execs once under `sg docker`.
set -uo pipefail

RUNTIME="${1:?runtime: subprocess|docker|prime|modal}"
N="${2:?num_tasks}"
MAX_TOKENS="${MAX_TOKENS:-}"
MULTIPLEX="${MULTIPLEX:-}"
LABEL="${3:-${RUNTIME}-${N}${MAX_TOKENS:+-t${MAX_TOKENS}}${MULTIPLEX:+-mux${MULTIPLEX}}}"

if [ "$RUNTIME" = docker ] && [ -z "${_BENCH_SG:-}" ]; then
  exec sg docker -c "_BENCH_SG=1 MAX_TOKENS=${MAX_TOKENS} MULTIPLEX=${MULTIPLEX} $(printf '%q ' "$0" "$@")"
fi

OUT="/tmp/vbench/${LABEL}"
mkdir -p "$OUT"
set -a; . "$HOME/.env" 2>/dev/null || true; set +a
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # the worktree this script lives in

CAP=${MAX_TOKENS:+--sampling.max_tokens $MAX_TOKENS}
MUX=${MULTIPLEX:+--multiplex $MULTIPLEX}
start=$(date +%s)
uv run eval gsm8k-v1 \
  --harness.id default --harness.enable_bash false \
  --harness.runtime.type "$RUNTIME" \
  --max_concurrent 512 --num_tasks "$N" --num_rollouts 1 \
  --retry.attempts 1 $CAP $MUX \
  --rich false --output_dir "$OUT/run" \
  >"$OUT/stdout.log" 2>"$OUT/run.log"
rc=$?
end=$(date +%s)
echo "BENCH runtime=$RUNTIME n=$N max_tokens=${MAX_TOKENS:-none} multiplex=${MULTIPLEX:-0} rc=$rc elapsed_s=$((end - start)) out=$OUT"

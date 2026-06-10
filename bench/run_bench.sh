#!/usr/bin/env bash
# Benchmark one (runtime, scale) eval run of gsm8k-v1 and report wall-clock e2e.
#
#   bench/run_bench.sh <subprocess|docker|prime|modal> <num_tasks> [label]
#   MAX_TOKENS=4096 bench/run_bench.sh <runtime> <num_tasks>   # cap generation length
#
# One rollout per task (-r 1), concurrency capped at 512, default model. Set
# MAX_TOKENS to cap per-generation length (--sampling.max_tokens), trimming
# long-generation stragglers; the label then gets a -t<MAX_TOKENS> suffix. Raw
# results + logs land under /tmp/vbench/<label>; the BENCH line carries the
# end-to-end wall clock. The docker CLI needs the `docker` group active, so the
# docker runtime re-execs this script once under `sg docker`.
set -uo pipefail

RUNTIME="${1:?runtime: subprocess|docker|prime|modal}"
N="${2:?num_tasks}"
MAX_TOKENS="${MAX_TOKENS:-}"
LABEL="${3:-${RUNTIME}-${N}${MAX_TOKENS:+-t${MAX_TOKENS}}}"

if [ "$RUNTIME" = docker ] && [ -z "${_BENCH_SG:-}" ]; then
  exec sg docker -c "_BENCH_SG=1 MAX_TOKENS=${MAX_TOKENS} $(printf '%q ' "$0" "$@")"
fi

OUT="/tmp/vbench/${LABEL}"
mkdir -p "$OUT"
set -a; . "$HOME/.env" 2>/dev/null || true; set +a
cd "$HOME/verifiers-bench"

CAP=${MAX_TOKENS:+--sampling.max_tokens $MAX_TOKENS}
start=$(date +%s)
uv run eval gsm8k-v1 \
  --harness.id default --harness.enable_bash false \
  --harness.runtime.type "$RUNTIME" \
  --max_concurrent 512 --num_tasks "$N" --num_rollouts 1 \
  --retry.attempts 1 $CAP \
  --rich false --output_dir "$OUT/run" \
  >"$OUT/stdout.log" 2>"$OUT/run.log"
rc=$?
end=$(date +%s)
echo "BENCH runtime=$RUNTIME n=$N max_tokens=${MAX_TOKENS:-none} rc=$rc elapsed_s=$((end - start)) out=$OUT"

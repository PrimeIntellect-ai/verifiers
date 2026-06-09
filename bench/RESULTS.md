# v1 runtime benchmark — subprocess vs docker vs prime

End-to-end wall-clock comparison of the three v1 runtimes driving the **same**
workload, to measure the overhead each runtime adds — plus a fix for the prime
tunnel-creation rate limit that surfaced at scale.

## TL;DR

- Clean baselines for all three runtimes at 32 / 256 / 512 concurrent rollouts.
- **prime fell over above 32 concurrency**: each rollout opens a tunnel, and prime
  caps tunnel creation at **100/min per token** → `429 (per-token limit)` →
  `ProgramError`. At 256, **61% of rollouts failed**; at 512, ~41%.
- **Fix**: a process-wide leaky-bucket limiter on `tunnel.start()` paced to 100/min
  (`verifiers/v1/runtimes/prime.py`). After it, **prime/256 and prime/512 complete
  with 0 failures** — and faster (no retry churn / no failure stragglers).

## Setup

- **Taskset**: `gsm8k-v1`, one rollout per distinct problem (`-r 1`)
- **Harness**: `default`, `enable_bash=false` — held constant; only `runtime.type` varies
- **Model**: `deepseek/deepseek-v4-flash` (eval default) via Prime Inference (`api.pinference.ai`)
- **Concurrency**: `--max_concurrent 512`; **retries off** (`--retry.attempts 1`)
- **Scales (rollouts)**: 32 → 256 → 512
- **Runtimes**: `subprocess` (local process), `docker` (`python:3.11-slim` container/rollout), `prime` (`python:3.11-slim` cloud sandbox + tunnel/rollout)
- Harness git: verifiers `exp/v1-runtime-bench` (off `feat/nano-as-v1`)

`generation.duration` is measured from just before runtime creation to harness end,
so it **includes per-rollout provisioning** (container/sandbox spin-up, and — for
prime — the rate-limiter wait) — which is what isolates runtime overhead.

## Start commands

```bash
# subprocess / prime — env sourced (PRIME_API_KEY), from the worktree
uv run eval gsm8k-v1 \
  --harness.id default --harness.enable_bash false \
  --harness.runtime.type <subprocess|prime> \
  --max_concurrent 512 --num_tasks <32|256|512> --num_rollouts 1 \
  --retry.attempts 1 --rich false --output_dir <out>

# docker — same, but the eval needs the `docker` group active:
sg docker -c '... uv run eval ... --harness.runtime.type docker ...'
```

Wrapped by `bench/run_bench.sh <runtime> <num_tasks>`; summarized by `bench/summarize.py`.

## Results — end-to-end wall clock (seconds, retries off)

| rollouts | subprocess | docker | prime (with limiter) |
|---:|---:|---:|---:|
| 32  | **14**  | **38**  | **72**  |
| 256 | **149** | **369** | **274** |
| 512 | **306** | **543** | **583** |

All runs above completed with **0 failed rollouts**. Notable: at 256, **prime
(274s) beats docker (369s)** — docker chokes on 256 local containers, while prime
offloads to genuinely parallel cloud sandboxes (tunnel issuing paced to 100/min).

## Per-rollout detail

`gen_*` = `generation.duration` (provisioning + limiter wait + model call), seconds.

| run | e2e | roll/s | gen_min | gen_p50 | gen_p95 | gen_max | reward | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subprocess-32  | 14  | 2.29 | 3.8  | 5.4   | 10.4  | 12.1  | 1.000 | 0 |
| subprocess-256 | 149 | 1.72 | 7.8  | 11.1  | 18.3  | 146.9 | 0.941 | 0 |
| subprocess-512 | 306 | 1.67 | 13.9 | 20.7  | 26.8  | 303.4 | 0.969 | 0 |
| docker-32  | 38  | 0.84 | 11.1 | 13.7  | 20.0  | 34.4  | 0.969 | 0 |
| docker-256 | 369 | 0.69 | 38.4 | 144.5 | 182.6 | 364.6 | 0.961 | 0 |
| docker-512 | 543 | 0.94 | 30.1 | 286.0 | 359.8 | 537.4 | 0.951 | 0 |
| prime-32  | 72  | 0.44 | 21.3 | 27.6  | 34.3  | 62.7  | 0.969 | 0 |
| prime-256 | 274 | 0.93 | 24.8 | 112.2 | 189.5 | 268.6 | 0.949 | 0 |
| prime-512 | 583 | 0.88 | 26.0 | 200.6 | 359.1 | 576.1 | 0.939 | 0 |

Reward is a completion sanity-check only (default sampling temperature → run-to-run noise); not a runtime metric.

## The prime tunnel-rate-limit fix

Each prime sandbox opens a tunnel so its in-sandbox program can reach the
interception server. prime rate-limits tunnel creation to **100/min per API token**;
beyond that, `tunnel.start()` returns `429 Too Many Requests (per-token limit)`,
which raises `ProgramError` and kills the rollout. **Before** the fix (retries off):

| scale | tunnels created | failures (429) | reward |
|---:|---:|---:|---:|
| 32  | 32/32   | 0/32 (0%)   | 0.969 |
| 256 | 100/256 | 156/256 (61%) | 0.383 |
| 512 | ~200/512 | ~41% †      | 0.55 † |

† the 512 "before" run used the default 3 retries, which masked then re-tripped the
limit (1236 attempts, 936 `ProgramError`, 212/512 terminal failures).

**Fix** (`verifiers/v1/runtimes/prime.py`): one process-wide `aiolimiter.AsyncLimiter`
shared across all rollouts, wrapping `tunnel.start()`, paced to 100/min. It's phrased
as **1 token every 0.6s (capacity 1)**, not `AsyncLimiter(100, 60)`, on purpose: a
full 100-capacity bucket would let 100 tunnels fire at once and then refill, stacking
to ~200 within the first minute and still tripping the limit. **After** the fix:

| scale | tunnels created | failures | e2e | reward |
|---:|---:|---:|---:|---:|
| 256 | **256/256** | **0** | 442s → **274s** | 0.383 → **0.949** |
| 512 | **512/512** | **0** | — → **583s** | 0.55 → **0.939** |

The limiter is a no-op at 32 concurrency (well under 100/min). Its pacing cost shows
up as higher `gen_p50` (rollouts wait for a tunnel slot — issuing 512 at 100/min is
~5 min), but that's strictly better than the failure storm, and the run finishes
sooner overall (no retry churn, no failure-induced stragglers).

## Findings

1. **Fixed provisioning overhead** (`gen_min` at the smallest scale, before the endpoint saturates): subprocess ≈ 0 (3.8 ≈ raw model call), **docker ≈ +7 s/container** (11.1), **prime ≈ +18 s/sandbox+tunnel** (21.3).
2. **The model endpoint is the real bottleneck at high concurrency.** subprocess has ~0 provisioning, yet `gen_min` climbs 3.8 → 7.8 → 13.9 across 32 → 256 → 512 — Prime Inference queueing under concurrent load, a confound shared by all runtimes.
3. **docker degrades sharply at scale** — 256/512 local containers drive `gen_p50` to 144 s / 286 s (vs 13.7 s at 32). Completes cleanly but slowly.
4. **prime is tunnel-rate-limited without pacing** — fixed by the limiter above; with it, prime scales cleanly and beats docker at 256.

## Operational notes

- Failed prime runs **leak tunnels and sandboxes** (a program crashing on `ProgramError` doesn't always tear its tunnel down). Cleanup is scoped to this token's own resources by explicit ID (`prime tunnel stop <ids>` / `prime sandbox delete <ids>`); the disconnected-tunnel backlog drains over several passes. Worth hardening teardown on the prime error path.

## Next

- **endpoint**: it dominates wall clock at 512 concurrency — re-measure against a higher-throughput inference deployment to separate runtime overhead from model latency.
- **prime teardown**: guarantee tunnel/sandbox cleanup on the `ProgramError` path so failures don't leak.

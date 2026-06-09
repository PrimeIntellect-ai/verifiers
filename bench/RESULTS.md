# v1 runtime benchmark — subprocess vs docker vs prime

End-to-end wall-clock comparison of the three v1 runtimes driving the **same**
workload, to measure the overhead each runtime adds.

## Setup

- **Taskset**: `gsm8k-v1`, one rollout per distinct problem (`-r 1`)
- **Harness**: `default`, `enable_bash=false` — held constant; only `runtime.type` varies
- **Model**: `deepseek/deepseek-v4-flash` (eval default) via Prime Inference (`api.pinference.ai`)
- **Concurrency**: `--max_concurrent 512`; **retries off** (`--retry.attempts 1`)
- **Scales (rollouts)**: 32 → 256 → 512
- **Runtimes**: `subprocess` (local process), `docker` (`python:3.11-slim` container/rollout), `prime` (`python:3.11-slim` cloud sandbox + tunnel/rollout)
- Harness git: verifiers `exp/v1-runtime-bench` (off `feat/nano-as-v1`)

`generation.duration` is measured from just before runtime creation to harness end,
so it **includes per-rollout provisioning** (container/sandbox spin-up) — which is
what isolates runtime overhead.

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

## Results — end-to-end wall clock (seconds)

| rollouts | subprocess | docker | prime |
|---:|---:|---:|---:|
| 32  | **14**  | **38**  | **72**  |
| 256 | **149** | **369** | **274** |
| 512 | **306** | **543** | **583** |

All runs completed with **0 failed rollouts**. At 256, **prime (274s) beats docker
(369s)** — docker chokes on 256 local containers, while prime offloads to genuinely
parallel cloud sandboxes.

## Per-rollout detail

`gen_*` = `generation.duration` (provisioning + model call), seconds.

| run | e2e | roll/s | gen_min | gen_p50 | gen_p90 | gen_max | reward | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subprocess-32  | 14  | 2.29 | 3.8  | 5.4   | 8.9   | 12.1  | 1.000 | 0 |
| subprocess-256 | 149 | 1.72 | 7.8  | 11.1  | 14.2  | 146.9 | 0.941 | 0 |
| subprocess-512 | 306 | 1.67 | 13.9 | 20.7  | 25.7  | 303.4 | 0.969 | 0 |
| docker-32  | 38  | 0.84 | 11.1 | 13.7  | 19.0  | 34.4  | 0.969 | 0 |
| docker-256 | 369 | 0.69 | 38.4 | 144.5 | 179.4 | 364.6 | 0.961 | 0 |
| docker-512 | 543 | 0.94 | 30.1 | 286.0 | 354.5 | 537.4 | 0.951 | 0 |
| prime-32  | 72  | 0.44 | 21.3 | 27.6  | 34.0  | 62.7  | 0.969 | 0 |
| prime-256 | 274 | 0.93 | 24.8 | 112.2 | 183.3 | 268.6 | 0.949 | 0 |
| prime-512 | 583 | 0.88 | 26.0 | 200.6 | 342.6 | 576.1 | 0.939 | 0 |

Reward is a completion sanity-check only (default sampling temperature → run-to-run noise); not a runtime metric.

## Findings

1. **Fixed provisioning overhead** (`gen_min` at the smallest scale, before the endpoint saturates): subprocess ≈ 0 (3.8 ≈ raw model call), **docker ≈ +7 s/container** (11.1), **prime ≈ +18 s/sandbox+tunnel** (21.3).
2. **The model endpoint is the real bottleneck at high concurrency.** subprocess has ~0 provisioning, yet `gen_min` climbs 3.8 → 7.8 → 13.9 across 32 → 256 → 512 — Prime Inference queueing under concurrent load, a confound shared by all runtimes.
3. **docker degrades sharply at scale** — 256/512 local containers drive `gen_p50` to 144 s / 286 s (vs 13.7 s at 32). Completes cleanly but slowly.
4. **prime scales cleanly and beats docker at 256** by offloading to parallel cloud sandboxes; its per-rollout `gen` rises with concurrency (provisioning + the shared endpoint).

## Next

- **endpoint**: it dominates wall clock at 512 concurrency — re-measure against a higher-throughput inference deployment to separate runtime overhead from model latency.

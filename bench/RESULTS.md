# v1 single-turn runtime benchmark

End-to-end comparison of the v1 runtimes (`subprocess`, `docker`, `prime`) on the same
single-turn workload, to measure the overhead each adds. Multiplexing is at its default
(interception servers/tunnels shared across rollouts), so this reflects current behaviour.

## Run it

```bash
bench/benchmark.sh    # subprocess docker prime x 32 64 128, gsm8k-v1, max_tokens 1024, default multiplex
RUNTIMES="subprocess prime" SIZES="32 256" MAX_TOKENS=2048 bench/benchmark.sh   # override
```

`benchmark.sh` writes **`bench/benchmark.json`** (committed): metadata + per-(runtime, batch)
e2e wall clock, the full per-rollout `generation.duration` list, reward, and error count.
Then visualize it:

```bash
uv run --with matplotlib python bench/plot.py   # benchmark.json -> bench/benchmark.png (gitignored)
```

## Reading it

- **Compare the generation-duration distribution (p10 / p50 / p90), not e2e.** e2e wall clock
  is gated by the single slowest rollout, and an endpoint-queue straggler can sit for minutes
  even on a capped generation — so e2e is noisy and the max hides the typical case.
- `generation.duration` runs from just before runtime creation to harness end, so it
  **includes per-rollout provisioning** (container/sandbox spin-up) — what isolates runtime
  overhead. A flat **p10** across batch sizes is the fixed provisioning floor; p50/p90 rising
  with concurrency is mostly the shared inference endpoint queueing, not the runtime.

## Findings

- Ordering holds: **subprocess < docker < prime** — provisioning ≈ 0 / per-container /
  per-sandbox+tunnel. The bottleneck at scale is **endpoint queueing**, shared by all runtimes.
- Interception **multiplexing** (PR #1605) keeps remote tunnels at O(N/multiplex): it roughly
  halves prime's gen p50/p90 from n≥128 (see `bench/plot_mux.py` / `mux_vs_base.png`); the
  win is fewer tunnels, not CPU. `multiplex=32` is the sweet spot.

## Next

- The endpoint dominates at high concurrency — re-measure against a higher-throughput inference
  deployment to separate runtime overhead from model latency.

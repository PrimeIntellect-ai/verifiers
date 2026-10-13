# Model request gate

## Problem

Providers cap concurrent requests per key and model. Prime Inference returns
`429 "Too many concurrent requests for this model (limit: 32)"` for
`internal/deepseek-v4.1-flash`. Nothing in the stack knows this limit:

- The interception server sends every harness request upstream at once. On a 429 it relays
  the status, and the harness SDK retries with its own backoff
  (`verifiers/v1/interception/server.py`, `sample()`; committed streams rerun up to
  `COMMITTED_TURN_ATTEMPTS`). Each retry is blind: no other request learns that the
  provider is full, so the load stays above the limit.
- `MAX_RETRIES = 0` in `verifiers/v1/clients/base.py`: the client itself never waits.
- prime-rl's `ConcurrencyController` (`src/prime_rl/orchestrator/concurrency.py`) adapts
  rollout concurrency to local GPU KV pressure. It does not see an external API.
- flow's `Limits` (`src/prime_flow/pipeline/limits.py`) wrap whole seat runs, not requests.
  One seat runs several agents, compaction calls, and judges.

## Evidence

- Hillclimb TB2 campaign today, all seats on deepseek, cap 32 per key:
  `data/tb2-campaign/eval/campaign-20261013T130525Z/logs/attempt_1/envs/eval/hillclimb.log`
  has 77 `upstream 429 ... (limit: 32)` failures in its first 45 minutes. Retries absorb
  them, at the cost of time; earlier all-deepseek runs lost jobs to them.
- sft-gen pins verifiers `0f46357f` on branch `sft-gen/checkpoint-rewind`. That branch adds
  `_note_rate_limit` (commit `e7e65d82`, also in our base via `feat/trace-resume`), which
  appends a timestamp to `$SFT_GEN_RATE_LIMIT_FILE` per relayed 429. It is not on
  `origin/main`, and nothing in sft-gen reads the file today. sft-gen also raises the bash
  program's SDK retries to 200 (`sft_gen/bash.py`, `request_max_retries`) so that a 429
  burst does not end an agent.
- data-flywheel (`pep-pipelines/environments/data-flywheel/data_flywheel/workers.py`)
  subclasses `InterceptionServer` and wraps every server-owned client in `LimitedClient`, a
  resizable FIFO semaphore. One gate per run, so nested harness calls cannot bypass it.
  It needs a `wrap_client` hook that only its verifiers fork has. The cap is static.
- repoforge (flow PR #25, `pipelines/repoforge/scripts/model_gate.py`) runs a separate
  reverse proxy on `127.0.0.1:8899`. Campaigns point `PRIME_INFERENCE_URL` at it. It caps
  concurrent requests per payload model across all runs on the host, from a JSON file it
  re-reads. It pauses a model on a sustained 404/503 outage. It does not react to 429s.
  It is a separate process that must be started and supervised, and it injects the key.
- Bypasses of the interception server: `verifiers/v1/judge.py` and
  `verifiers/v1/judges/decision.py` build their own `AsyncOpenAI`. Hillclimb's
  `hillclimb/tasks/benchmark_guard.py` posts with a raw `httpx.AsyncClient`.

## Existing support

- Retries exist (harness SDK, committed-stream rerun, whole-rollout `retries`). Admission
  control does not.
- A 429 is already on the trace as `ModelCall.error.status_code == 429`. Only a 429 that
  ends a call is recorded. Time spent waiting is not recorded anywhere.
- `Retry-After` is dropped: `EvalClient._request` keeps only the status and body.
- Upstream `main` has no rate-limit or concurrency work (searched the log for
  429 / rate limit / concurren / throttl). The only related commit is the sandbox
  `CreationLimiter` (#2657).

## Why verifiers, and why the HTTP client

Every model request from a rollout passes through `EvalClient`, which sends with
`build_async_httpx`. Judges send through `build_async_openai`, which also uses
`build_async_httpx`. The gate is an `httpx.AsyncClient` subclass that `build_async_httpx`
returns when the client config enables it, so:

- Harness turns, nested agents, compaction calls and judges are all covered. Nested
  agents inherit the run's client config, so one knob covers a whole run.
- prime-rl eval, `vf-eval` and flow pipelines get it with no code change. Their env
  servers run the same interception server.
- No extra process, no proxy, and no key handling outside the client.

A gate in prime-rl would not see nested calls. A gate in flow would sit around seats again.

## Design

- Key: endpoint URL, API-key digest, model. Two runs on the same key share one gate. A
  different key gets its own gate.
- Shared state is in a directory, by default host-wide under the system temp dir.
  Permits are `flock`s on slot files, so the kernel releases the permit of a process
  that dies. A state file, guarded by its own `flock`, holds the adaptive limit and the
  counters. A different directory (for example a shared mount) can widen the scope,
  but only where `flock` works.
- AIMD: the limit starts at `max_concurrent`. A 429 or 529 cuts it by `decrease_factor`,
  at most once per `decrease_cooldown`, so that one burst counts as one signal. Each
  success adds `1/limit`, which is about one permit per window. The limit stays between
  `min_concurrent` and `max_concurrent`.
- A rejected request gives back its permit and waits for `Retry-After` (or
  `retry-after-ms`), otherwise for exponential backoff. Then it queues again. After
  `max_rejections` absorbed rejections, the last 429 goes to the caller as before.
- A streamed response keeps its permit until the response is closed.
- No per-call trace record of the gate. A `ModelCall.throttle` record (seconds queued,
  429s absorbed, limit at admission), filled through a context variable around each
  upstream call, was built and removed: nothing read it. The gate's state file holds the
  live limit and counters.

## What we will not build

- No proxy process (repoforge's design). The client-side gate covers the same scope
  without a process to supervise.
- No session-priority queue (repoforge's oldest-session-first). Admission in a process
  is FIFO, and across processes it is first come, first served.
- No outage pause on 404/503. That is a separate failure mode. The existing retries and
  rollout errors handle it.
- No token-rate (TPM) limiting. The observed limit is on concurrency.
- No prime-rl metric plumbing in this change. The counters are on `trace.calls`, so a
  consumer can add them up.
- No changes to the `SFT_GEN_RATE_LIMIT_FILE` hook. sft-gen can drop it, and its
  200-retry harness, once it enables the gate.

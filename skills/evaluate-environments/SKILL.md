---
name: evaluate-environments
description: Run and analyze evaluations for verifiers environments using prime eval. Use when asked to smoke-test environments, run benchmark sweeps, resume interrupted evaluations, compare models, inspect sample-level outputs, or produce evaluation summaries suitable for deciding next steps.
---

# Evaluate Environments

## Goal
Run reliable environment evaluations and produce actionable summaries, not raw logs.

## Canonical Eval Path
1. Use `prime eval run` as the default way to run evaluations.
2. Do not add `--skip-upload` or other opt-out flags unless the user explicitly requests that deviation.
3. Standard `prime eval run` runs save results automatically, keeping them available in the user's private Evaluations tab and locally in `prime eval tui`.
4. For Prime Inference models with available pricing, eval output and saved metadata include estimated total-run USD cost automatically; no extra flags or API-key handling are needed.

## Core Loop
1. Run a smoke evaluation first (do not require pre-install):
```bash
prime eval run my-env -m openai/gpt-4.1-mini -n 5
```
2. Use owner/env slug directly when evaluating Hub environments:
```bash
prime eval run owner/my-env -m openai/gpt-4.1-mini -n 5
```
3. Scale only after smoke pass:
```bash
prime eval run owner/my-env -m openai/gpt-4.1-mini -n 200 -r 3 -s
```
4. Treat ownerless env ids as local-first. If not found locally, rely on Prime resolution for your remote env where applicable.
5. When the user asks for a "real" or "base" eval, do not substitute a tiny smoke run. Use the requested model/env and make the run size explicit before interpreting results.
6. If the user says the defaults are fine or asks for no flags, use the shortest canonical command and rely on global config:
```bash
prime eval run my-env
prime eval run my-env -m openai/gpt-4.1-mini
```

## Endpoint Shortcuts And Model Family Choice
1. Encourage users to define endpoint aliases in `configs/endpoints.toml` so model, base URL, and key wiring stay reusable.
2. Use aliases via `-m <endpoint_id>` instead of repeating `-b` and `-k`.
3. Ask users explicitly whether they want an instruct or reasoning model before non-trivial evaluations.
4. Instruct go-tos for quick behavior checks: `gpt-4.1` series and `qwen3` instruct series.
5. Reasoning go-tos for deeper test coverage: `gpt-5` series, `qwen3` thinking series, and `glm` series.
6. Example endpoint registry:
```toml
[[endpoint]]
endpoint_id = "gpt-4.1-mini"
model = "gpt-4.1-mini"
url = "https://api.openai.com/v1"
key = "OPENAI_API_KEY"

[[endpoint]]
endpoint_id = "qwen3-32b-i"
model = "qwen/qwen3-32b-instruct"
url = "https://api.pinference.ai/api/v1"
key = "PRIME_API_KEY"
```
7. Endpoint entries support optional `headers` (or `extra_headers`) for custom HTTP headers sent with inference requests:
```toml
[[endpoint]]
endpoint_id = "my-proxy"
model = "gpt-4.1-mini"
url = "https://api.example/v1"
key = "OPENAI_API_KEY"
headers = { "X-Custom-Header" = "value" }
```
8. Endpoint entries support `api_client_type` when the provider is not OpenAI Chat Completions compatible. Use `openai_responses` for Responses-compatible endpoints and `anthropic_messages` for Anthropic Messages endpoints:
```toml
[[endpoint]]
endpoint_id = "gpt-responses"
model = "gpt-5.4-mini"
url = "https://api.openai.com/v1"
key = "OPENAI_API_KEY"
api_client_type = "openai_responses"
```

## Publish Gate Before Large Runs
1. After smoke tests pass and results look stable, proactively suggest pushing the environment to Hub before large eval sweeps or RL work.
2. Ask the user explicitly: should visibility be `PUBLIC` or `PRIVATE`?
3. Push with chosen visibility:
```bash
prime env push my-env --visibility PUBLIC
```
or
```bash
prime env push my-env --visibility PRIVATE
```
4. For hosted environment workflows, prefer running large jobs against the Hub slug:
```bash
prime eval run owner/my-env -m openai/gpt-4.1-mini -n 200 -r 3 -s
```

## Prefer Config-Driven Evals Beyond Smoke Tests
1. For anything beyond quick checks, nudge the user to create an eval TOML config.
2. Use config files to run multiple evals in one command and keep runs reproducible:
```bash
prime eval run configs/eval/my-benchmark.toml
```
3. Make config files the default for benchmark sweeps, multi-model comparisons, and recurring reports.
4. Use `name` on individual `[[eval]]` entries when the same environment appears multiple times. `id` selects the environment to load; `name` labels the run in displays, summaries, metadata, and saved result paths.

## Common Evaluation Patterns
1. Override v1 taskset and harness config through explicit child sections:
```bash
prime eval run my-env -a '{"config":{"taskset":{"difficulty":"hard"},"harness":{"max_turns":20}}}'
```
2. Override constructor kwargs:
```bash
prime eval run my-env -x '{"max_turns":20}'
```
3. Bound per-rollout wall-clock time (use the dedicated `--timeout` flag; wins over `-x` and TOML `[eval.extra_env_kwargs]`):
```bash
prime eval run my-env --timeout 600
```
4. Save extra state columns:
```bash
prime eval run my-env -s -C "judge_response,parsed_answer"
```
5. Resume interrupted runs:
```bash
prime eval run my-env -n 1000 -s --resume
```
6. Save results to a custom output directory:
```bash
prime eval run my-env -s -o /path/to/output
```
7. Run multi-environment TOML suites:
```bash
prime eval run configs/eval/my-benchmark.toml
```
8. Run the same environment more than once with different args by giving each entry a `name`:
```toml
[[eval]]
id = "reverse-text"
name = "reverse-text-short"

[eval.args]
max_length = 32

[[eval]]
id = "reverse-text"
name = "reverse-text-long"

[eval.args]
max_length = 256
```
9. Pass extra HTTP headers via CLI (repeatable):
```bash
prime eval run my-env -m my-proxy --header "X-Custom-Header: value"
```
10. Set headers in `[[eval]]` TOML configs as a table or list (merge order: registry row < `headers` table < `header` list / `--header`):
```toml
[[eval]]
env_id = "my-env"
headers = { "X-Custom-Header" = "value" }
header = ["X-Another: val"]
```
11. Run ablation sweeps using `[[ablation]]` blocks in TOML configs:
```toml
[[ablation]]
env_id = "my-env"

[ablation.sweep]
temperature = [0.0, 0.5, 1.0]

[ablation.sweep.taskset]
difficulty = ["easy", "hard"]
```
This generates the cartesian product (6 configs in this example). Sweep v1 environment-owned settings under `taskset` or `harness`, not as root args. Use `--abbreviated-summary` (`-A`) for compact ablation results.

## Inspect Saved Results
1. Browse locally saved runs:
```bash
prime eval tui
```
2. Check `metadata.json` for aggregate token usage and, when available, total-run `cost.input_usd`, `cost.output_usd`, and `cost.total_usd`.
3. Inspect platform-visible runs when needed:
```bash
prime eval list
prime eval get <eval-id>
prime eval samples <eval-id>
```

## Metrics Interpretation
1. Treat binary and continuous rewards differently.
2. Use pass@k-style interpretation only when rewards are effectively binary.
3. For continuous rewards, focus on distribution shifts and per-task means.
4. Always inspect samples before concluding regressions.

## Reliability Rules
1. Keep environment/model/config fixed while comparing variants.
2. Record exact command lines and key flags in the report.
3. Call out missing credentials, endpoint mismatches, and dependency errors directly.
4. Do not overinterpret tiny sample runs.
5. Distinguish a completed rollout with poor reward from an environment/runtime failure.
6. For timeout debugging, check the environment's own timeout behavior and the outer sandbox/eval timeout before changing reward logic.
7. For repo example changes, use `tests/test_envs.py -k <env>` when package installability is part of the risk, not just `prime eval run` from the current checkout.

## Output Format
Return:
1. Run configuration table.
2. Aggregate metrics and key deltas.
3. Sample-level failure themes.
4. Clear recommendation: proceed, iterate environment, or retune model/sampling.

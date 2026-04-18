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

## Common Evaluation Patterns
1. Pass args to `load_environment()`:
```bash
prime eval run my-env -a '{"difficulty":"hard"}'
```
2. Override constructor kwargs:
```bash
prime eval run my-env -x '{"max_turns":20}'
```
3. Save extra state columns:
```bash
prime eval run my-env -s -C "judge_response,parsed_answer"
```
4. Resume interrupted runs:
```bash
prime eval run my-env -n 1000 -s --resume
```
5. Save results to a custom output directory:
```bash
prime eval run my-env -s -o /path/to/output
```
6. Run multi-environment TOML suites:
```bash
prime eval run configs/eval/my-benchmark.toml
```
7. Pass extra HTTP headers via CLI (repeatable):
```bash
prime eval run my-env -m my-proxy --header "X-Custom-Header: value"
```
8. Set headers in `[[eval]]` TOML configs as a table or list (merge order: registry row < `headers` table < `header` list / `--header`):
```toml
[[eval]]
env_id = "my-env"
headers = { "X-Custom-Header" = "value" }
header = ["X-Another: val"]
```
9. Run ablation sweeps using `[[ablation]]` blocks in TOML configs:
```toml
[[ablation]]
env_id = "my-env"

[ablation.sweep]
temperature = [0.0, 0.5, 1.0]

[ablation.sweep.env_args]
difficulty = ["easy", "hard"]
```
This generates the cartesian product (6 configs in this example). Use `--abbreviated-summary` (`-A`) for compact ablation results.

## Inspect Saved Results
1. Browse locally saved runs:
```bash
prime eval tui
```
2. Inspect platform-visible runs when needed:
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

## Output Format
Return:
1. Run configuration table.
2. Aggregate metrics and key deltas.
3. Sample-level failure themes.
4. Clear recommendation: proceed, iterate environment, or retune model/sampling.

# solve-swe

Gold-patch validation of any SWE taskset — no agent, no LLM. Mirrors `rlm-swe`'s interface but runs `SolveEnv` (apply gold patch → run tests → score) instead of an agent.

### Quickstart

```bash
uv pip install -e environments/solve_swe

vf-eval solve_swe -a '{"task_type": "multiswe"}' \
    --max-concurrent 6 --max-retries 2 \
    --state-columns reason,attempts,elapsed_s,test_output_tail \
    -s --resume
```

### Args

| arg | default | description |
|---|---|---|
| `task_type` | `"r2e"` | SWE backend: `r2e`, `multiswe`, `swebench`, `openswe` |
| `**solve_kwargs` | — | forwarded to `SolveEnv(...)` (e.g. `test_timeout`, `cpu_cores`, `timeout_seconds`, `test_output_tail_chars`, `labels`) |

### Recommended state columns

`reason,attempts,elapsed_s,test_output_tail` — matches `SandboxTaskSet.validate(out_path=...)`'s row schema.

`reason` ∈ `{pass, test_failed, gold_apply_failed, setup_failed, sandbox_error, billing_error, timeout}`.

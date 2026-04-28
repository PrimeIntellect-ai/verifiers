# solve-env

LLM-free golden-patch validation for any `SandboxTaskSet`.

Streams JSONL via `-s`, resumes with `--resume`, caps concurrency with `--max-concurrent`, surfaces validate's row schema via `--state-columns reason,attempts,elapsed_s,test_output_tail`.

### Quickstart

```bash
uv pip install -e environments/solve_env

# Validate all of multi-swe-rl
vf-eval solve_env \
    -a '{"taskset": "verifiers.envs.experimental.composable.tasksets.swe:make_swe_taskset",
         "taskset_args": {"backend": "multiswe"}}' \
    --max-concurrent 6 --max-retries 2 \
    --state-columns reason,attempts,elapsed_s,test_output_tail \
    -s --resume

# r2e-gym
vf-eval solve_env -a '{"taskset": "...:make_swe_taskset", "taskset_args": {"backend": "r2e"}}' ...

# Any non-SWE SandboxTaskSet by class
vf-eval solve_env -a '{"taskset": "my.pkg:MyTaskSet", "taskset_args": {...}}' ...
```

### Args

| arg | type | description |
|---|---|---|
| `taskset` | `str` | `"module.path:attr"` — `attr` is a callable factory or a `SandboxTaskSet` subclass |
| `taskset_args` | `dict` | forwarded to the resolved callable |
| `**solve_kwargs` | — | forwarded to `SolveEnv(...)` (e.g. `test_timeout`, `cpu_cores`, `timeout_seconds`, `test_output_tail_chars`, `labels`) |

`reason` ∈ `{pass, test_failed, gold_apply_failed, setup_failed, sandbox_error, billing_error, timeout}` — same enum as `SandboxTaskSet.validate(out_path=...)`.

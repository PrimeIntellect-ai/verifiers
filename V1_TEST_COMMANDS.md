# V1 Test Commands

Run these from the repository root.

## Local Checks

These are safe to run in parallel in separate terminals.

```bash
python3 -c 'import tomllib,collections,pathlib; data=tomllib.loads(pathlib.Path("configs/endpoints.toml").read_text()); ids=[e["endpoint_id"] for e in data["endpoint"]]; assert not [k for k,v in collections.Counter(ids).items() if v>1]; assert {e["key"] for e in data["endpoint"]} == {"PRIME_API_KEY"}; assert {e["url"] for e in data["endpoint"]} == {"https://api.pinference.ai/api/v1"}; print(f"endpoints ok: {len(ids)}")'
```

```bash
env PYTHONDONTWRITEBYTECODE=1 uv run ruff check verifiers/types.py tests/test_v1_nano_core.py packages/tasksets/tasksets/replay.py environments/tau2_bench_v1/tau2_bench_v1/taskset.py environments/tau2_bench_v1/tau2_bench_v1/servers/user/user.py README.md
```

```bash
env PYTHONDONTWRITEBYTECODE=1 uv run --python 3.13 ty check verifiers/v1 packages/tasksets/tasksets packages/harnesses/harnesses
```

```bash
env PYTHONWARNINGS=ignore::SyntaxWarning uv run --no-dev --group policy semgrep --metrics=off --disable-version-check --config .semgrep/verifiers.yml --error --quiet
```

```bash
env PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_message_utils.py tests/test_v1_taskset_utils.py tests/test_v1_nano_core.py::test_openenv_and_openreward_rewards_sum_turn_rewards tests/test_v1_nano_core.py::test_openenv_and_openreward_task_schemas_are_explicit tests/test_v1_nano_core.py::test_openenv_mcp_setup_lists_tools_without_reset tests/test_v1_nano_core.py::test_openenv_user_tool_returns_bound_turn_reward_payload tests/test_v1_nano_core.py::test_openreward_user_tool_returns_bound_turn_reward_payload tests/test_eval_cli.py::test_cli_toml_per_env_shuffle tests/test_eval_cli.py::test_cli_shuffle_defaults_seed_when_enabled tests/test_multiturn_env.py tests/test_singleturn_env.py -q
```

## Default V1 Evals

Each command uses the environment default settings. For the current example
packages, that means `n=5` and `r=3`. `--disable-tui` and
`--abbreviated-summary` only change display.

Run these in separate terminals when you want broad coverage:

```bash
uv run prime eval run reverse-text-v1 --disable-tui --abbreviated-summary
```

```bash
uv run prime eval run alphabet-sort-v1 --disable-tui --abbreviated-summary
```

```bash
uv run prime eval run mcp-search-env-v1 --disable-tui --abbreviated-summary
```

```bash
uv run prime eval run math-python-v1 --disable-tui --abbreviated-summary
```

```bash
uv run prime eval run hello-group-reward-v1 --disable-tui --abbreviated-summary
```

```bash
uv run prime eval run sft-replay-v1 --disable-tui --abbreviated-summary
```

Stateful user/tool environments are slower and noisier. Run these separately
from the quick eval batch:

```bash
uv run prime eval run openenv-echo-v1 --disable-tui --abbreviated-summary
```

```bash
uv run prime eval run tau2-bench-v1 --disable-tui --abbreviated-summary
```

## PR Checks

```bash
gh pr checks 1559
```

# harbor-v1

### Overview
- **Environment ID**: `harbor-v1`
- **Short description**: Generic v1 Harbor taskset environment
- **Tags**: harbor, cli_agent, v1

### Datasets
- **Primary dataset(s)**: Harbor task directories
- **Source links**: <https://github.com/laude-institute/harbor>
- **Split sizes**: 1 bundled smoke task by default

### Task
- **Type**: multiturn, cli_agent
- **Rubric overview**: Reward returned by running Harbor verifier tests

### Quickstart
Run the environment:

```bash
prime eval run harbor-v1
```

Configure model and sampling:

```bash
prime eval run harbor-v1 -m openai/gpt-4.1-mini -n 1 -r 1 -t 1024 -T 0.7
```

Notes:
- v1 task settings belong under `config.taskset` when passed through `-a` / `--env-args`.
- Use `taskset` and `harness` config sections for v1 object configuration in TOML.
- Harbor tasks with `image` fields require a container runtime such as Docker or Prime.

### Taskset Config

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `source` | `"hub" \| "local" \| "package"` | `"package"` in this environment | Dataset source resolver. |
| `dataset` | str | `"harbor_v1"` in this environment | Hub id, local tasks path, or Python package name. |
| `task_names` | list[str] | `null` | Explicit Harbor task names to run. |
| `cache_dir` | str | `null` | Optional Hub cache root override. |
| `refresh` | bool | `false` | Refresh Hub cache before loading. |
| `require_image` | bool | `false` | Require every task to declare `[environment].docker_image`. |
| `verifier_timeout_seconds` | float | `900.0` | Default timeout for Harbor verifier scripts. |

### Harness Config

This package defaults to the reusable `OpenCode` harness. OpenCode settings
belong under `config.harness`:

```toml
[env.harness]
max_turns = 4
cwd = "/app"
```

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Harbor verifier reward, usually `0.0` or `1.0` |
| `num_turns` | Number of intercepted assistant turns |


## How It Works

1. `HarborTaskset` resolves Hub, local, or package task directories into typed
   v1 tasks.
2. The taskset maps `[environment].docker_image` and resource hints onto
   generic v1 `Task` fields.
3. `OpenCode` runs as the default agent harness.
4. Reward is computed by staging only `tests/` into the live runtime after the
   rollout and running `tests/test.sh`.

## Requirements

- Harbor task directory with `task.toml`, `instruction.md`, and `tests/`
- A container runtime when tasks declare `[environment].docker_image`

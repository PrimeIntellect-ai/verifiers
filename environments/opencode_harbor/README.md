# opencode-harbor

### Overview
- **Environment ID**: `opencode-harbor`
- **Short description**: Environment for running an agent with OpenCode on Harbor tasks
- **Tags**: opencode, cli_agent, harbor

### Datasets
- **Primary dataset(s)**: Harbor tasks (hello-world included)
- **Source links**: <https://github.com/laude-institute/harbor>
- **Split sizes**: 1 example task included

### Task
- **Type**: multiturn, cli_agent
- **Rubric overview**: Binary, returned by running task tests

### Quickstart
Run the v1 Taskset/Harness path:

```bash
prime eval run opencode-harbor -a '{"v1": true}'
```

Configure model and sampling:

```bash
prime eval run opencode-harbor   -m openai/gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `v1` | bool | `false` | Use `vf.HarborTaskset` + `vf.OpenCode`. |
| `dataset_path` | str | bundled `tasks/` | Local Harbor task directory or dataset directory. |
| `dataset` | str | `null` | `terminal-bench-sample` or `terminal-bench` task selection. |
| `tasks` | list[str] | `null` | Explicit Harbor task names to run. |
| `max_turns` | int | `4` | Harness turn cap for the v1 path. |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |


## How It Works

1. `vf.HarborTaskset` loads Harbor task rows and contributes sandbox settings,
   task uploads, env vars, and the Harbor reward.
2. `vf.OpenCode` contributes the reusable OpenCode CLI program, install/setup,
   intercepted endpoint config, MCP tool proxy, and log artifact collection.
3. v1 resolves both sides into one sandboxed command program at rollout time.
4. Reward is computed by running the Harbor test scripts after the rollout.

`HarborTaskset` and `OpenCode` are packaged under `verifiers.v1.packages` and
re-exported from `verifiers.v1`.

## Requirements

- Harbor tasks directory with `task.toml` and `instruction.md` files
- Docker images specified in task configs


## Reward

Uses Harbor's standard reward mechanism:

- Runs `tests/test.sh` after agent completion
- Reads reward from `/logs/verifier/reward.txt` or `/logs/verifier/reward.json`
- Returns float reward value (typically 0 or 1)

## Notes

- OpenCode is installed at runtime.
- Agent logs are saved to `/logs/agent/opencode.txt` in the sandbox
- Uses `@ai-sdk/openai-compatible` provider for API interception

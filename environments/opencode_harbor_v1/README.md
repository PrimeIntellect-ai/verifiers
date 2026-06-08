# opencode-harbor-v1

### Overview
- **Environment ID**: `opencode-harbor-v1`
- **Short description**: Environment for running an agent with OpenCode on Harbor tasks
- **Tags**: opencode, cli_agent, harbor

### Datasets
- **Primary dataset(s)**: Harbor tasks
- **Source links**: <https://github.com/laude-institute/harbor>
- **Split sizes**: 11 bundled tasks

### Task
- **Type**: multiturn, cli_agent
- **Rubric overview**: Binary, returned by running task tests

### Quickstart
Run the environment:

```bash
prime eval run opencode-harbor-v1
```

Configure model and sampling:

```bash
prime eval run opencode-harbor-v1 -m openai/gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- v1 task settings belong under `config.taskset` when passed through `-a` / `--env-args`.
- Use `taskset` and `harness` config sections for v1 object configuration in TOML.

### Taskset Config

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `source` | `"harbor" \| "package"` | `"package"` in this environment | Dataset source resolver. |
| `dataset` | str | `"opencode_harbor_v1"` in this environment | Harbor dataset id or Python package name. |
| `tasks` | list[str] | `null` | Explicit Harbor task names to run. |
| `cache_dir` | str | `null` | Optional Harbor cache root override. |
| `refresh` | bool | `false` | Refresh Harbor cache before loading. |
| `require_image` | bool | `false` | Require every task to declare `[environment].docker_image`. |

### Harness Config

OpenCode settings belong under `config.harness`:

```toml
[env.harness]
max_turns = 4
version = "PrimeIntellect-ai/opencode@1.1.63-rl2"
cwd = "/app"
```

The harness also accepts the packaged `OpenCodeConfig` fields for `system_prompt`,
`log_path`, `disabled_tools`, `allow_git`, `disable_compaction`,
`provider_timeout_ms`, and runtime settings.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Harbor verifier reward, usually `0.0` or `1.0` |
| `num_turns` | Number of intercepted assistant turns |


## How It Works

1. `HarborTaskset` resolves Harbor or package task directories, maps
   `[environment].docker_image` and resource hints onto generic v1 `Task`
   fields, and owns the Harbor reward.
2. `OpenCode` contributes the reusable OpenCode CLI program, install/setup,
   intercepted endpoint config, MCP tool proxy, and log artifact collection.
3. The v1 runtime resolves both sides into one sandboxed command program at rollout time.
4. Reward is computed by staging only `tests/` into the live runtime after the
   rollout and running `tests/test.sh`.

`HarborTaskset` and `OpenCode` are packaged under `tasksets` and `harnesses` and
imported by the environment package.

## Requirements

- Harbor tasks directory with `task.toml` and `instruction.md` files
- Docker images specified in task configs


## Reward

Uses Harbor's standard reward mechanism:

- Runs `tests/test.sh` after agent completion
- Reads reward from `/logs/verifier/reward.txt`
- Returns float reward value (typically 0 or 1)

## Notes

- OpenCode is installed at runtime.
- Agent logs are saved to `/logs/agent/opencode.txt` in the sandbox
- Uses `@ai-sdk/openai-compatible` provider for API interception

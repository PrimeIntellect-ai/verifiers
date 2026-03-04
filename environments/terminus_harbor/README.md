# terminus-harbor

### Overview
- **Environment ID**: `terminus-harbor`
- **Short description**: Environment for running the Terminus2 agent through Harbor on Harbor tasks
- **Tags**: cli_agent, harbor, terminus

### Datasets
- **Primary dataset(s)**: Harbor tasks (hello-world included)
- **Source links**: <https://github.com/laude-institute/harbor>
- **Split sizes**: 1 example task included

### Task
- **Type**: multiturn, cli_agent
- **Rubric overview**: Binary, returned by running task tests

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run terminus-harbor
```

Configure model and sampling:

```bash
prime eval run terminus-harbor   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |


## How It Works

1. Creates a Prime sandbox for each Harbor task
2. Installs dependencies including Harbor itself 
3. Configures Terminus agent pointing at our openai_base_url
4. Runs Terminus via Harbor agent loop on the task instruction
5. Intercepts all API requests through Prime Tunnel
6. Computes reward by running Harbor test scripts

## Requirements

- Harbor tasks directory with `task.toml` and `instruction.md` files
- Docker images specified in task configs


## Reward

Uses Harbor's standard reward mechanism:

- Runs `tests/test.sh` after agent completion
- Reads reward from `/logs/verifier/reward.txt` or `/logs/verifier/reward.json`
- Returns float reward value (typically 0 or 1)

## Notes

- The agent writes `/tmp/vf_complete` when finished
- Agent logs are saved to `/logs/agent/terminus.txt` in the sandbox

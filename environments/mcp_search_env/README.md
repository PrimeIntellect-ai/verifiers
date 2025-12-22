# mcp-search-env

### Overview

- **Environment ID**: `mcp-search-env`
- **Short description**: Example environment using `vf.MCPEnv` for MCP server integration
- **Tags**: MCP, Tools

This environment demonstrates how to use the first-class `MCPEnv` from `verifiers.envs.experimental`.

### Datasets

- **Primary dataset(s)**: N/A
- **Source links**: N/A
- **Split sizes**: N/A

### Task

- **Type**: <multi-turn | tool use>
- **Parser**: N/A
- **Rubric overview**: N/A

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval mcp-search-env
```

Configure model and sampling:

```bash
uv run vf-eval mcp-search-env   -m gpt-4.1-mini   -n 1 -r 1
```

Notes:

- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

Document any supported environment arguments and their meaning. Example:

| Arg            | Type | Default | Description                            |
| -------------- | ---- | ------- | -------------------------------------- |
| `max_examples` | int  | `-1`    | Limit on dataset size (use -1 for all) |

### Metrics

| Metric         | Meaning                                       |
| -------------- | --------------------------------------------- |
| `reward`       | Main scalar reward (weighted sum of criteria) |
| `judge_reward` | LLM judge score (1.0 if correct, 0.0 if not)  |

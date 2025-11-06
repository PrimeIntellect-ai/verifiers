# mcp-env

### Overview

- **Environment ID**: `mcp-env`
- **Short description**: MCP Environment
- **Tags**: MCP, Tools

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
uv run vf-eval mcp-env
```

Configure model and sampling:

```bash
uv run vf-eval mcp-env   -m gpt-4.1-mini   -n 1 -r 1
```

Notes:

- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

Document any supported environment arguments and their meaning. Example:

| Arg            | Type | Default | Description                            |
| -------------- | ---- | ------- | -------------------------------------- |
| `max_examples` | int  | `-1`    | Limit on dataset size (use -1 for all) |

### Metrics

Summarize key metrics your rubric emits and how they’re interpreted.

| Metric     | Meaning                                       |
| ---------- | --------------------------------------------- |
| `reward`   | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer                  |

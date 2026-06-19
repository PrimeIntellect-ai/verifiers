# openenv-echo-v1

### Overview

- **Environment ID**: `openenv-echo-v1`
- **Short description**: OpenEnv's public Echo image through the reusable v1 MCP adapter.
- **Tags**: openenv, mcp, tools, v1

### Datasets

- **Primary dataset(s)**: Stateless Echo tool-use prompts.
- **Split sizes**: `num_tasks` prompts, default 100.

### Task

- **Type**: Tool use.
- **Rubric overview**: This protocol example is unscored.

### Quickstart

```bash
uv pip install -e environments/openenv_echo_v1
uv run eval openenv-echo-v1 -n 3 --harness.runtime.type docker
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `num_tasks` | int | `100` | Number of prompts. |
| `prompt` | str | Echo instruction | Opening model instruction. |

### Metrics

| Metric | Meaning |
| --- | --- |
| `openenv_reward` | Always zero for this stateless MCP example. |

## How It Works

The taskset pins OpenEnv's public Echo image and selects the MCP contract. The
reusable adapter starts it over a Unix socket and maps its JSON-RPC tools through
`vf.JSONRPCToolset`. This package contains no server or Docker build.

# sandbox-mcp-env

1. Define the MCP server or servers you want to use
2. Setup state will 
    1. Start by creating a sandbox for the rollout and exposing a port
    2. Then create transport(s) for the mcp servers which provide the interface for using the server
    3. It will run any necessary commands required for the mcp server
    4. Run the server in StreamableHTTP mode
    5. Finally register the MCP server's available tools
3. Rollout proceeds and agent can make mcp tool calls that are safe to interact within the sandbox

### Overview
- **Environment ID**: `sandbox-mcp-env`
- **Short description**: MCPEnv via sandboxed streaming http MCP servers
- **Tags**: mcp, sandbox

### Datasets
- **Primary dataset(s)**: NA
- **Source links**: NA
- **Split sizes**: NA

### Task
- **Type**: tool use
- **Parser**: NA
- **Rubric overview**: NA

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval sandbox-mcp-env -n 1 -r 1
```

Configure model and sampling:

```bash
uv run vf-eval sandbox-mcp-env   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Demo

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Demo

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |


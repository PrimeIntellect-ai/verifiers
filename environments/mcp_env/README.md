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

The default configuration launches the Exa and Fetch MCP servers over stdio.
These are real MCP servers maintained by the community and require no manual
setup beyond providing an `EXA_API_KEY` when you want Exa search access.

To run servers with StreamableHTTP, build configurations with the
`build_streamable_http_server_config` helper in `mcp_env.py`. The helper
allocates a port, formats placeholders in your launch arguments, and points the
client at the resulting `http://host:port/path` endpoint—matching the
deployment flow described in the official MCP SDK documentation for
StreamableHTTP transports.【b1dc47†L1-L18】

```python
from environments.mcp_env.mcp_env import build_streamable_http_server_config

streamable_local = build_streamable_http_server_config(
    name="local-http",
    command="uv",
    args=[
        "run",
        "examples/snippets/servers/streamable_config.py",
        "--",
        "--port",
        "{port}",
    ],
    url_path="/mcp",
)
```

For remote StreamableHTTP providers, skip the command entirely and supply the
endpoint and headers directly—for example, Telnyx exposes an MCP endpoint at
`https://api.telnyx.com/v2/mcp` that can be accessed with a bearer token.【0a9bd8†L1-L6】

Configure model and sampling:

```bash
uv run vf-eval mcp-env \
  -m gpt-4.1-mini \
  -n 1 -r 1
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

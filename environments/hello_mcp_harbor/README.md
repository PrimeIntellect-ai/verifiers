# hello-mcp-harbor

`HarborEnv` that exercises the framework-managed MCP

Adapted from Harbor's own [`examples/tasks/hello-mcp`](https://github.com/laude-institute/harbor/tree/main/examples/tasks/hello-mcp).

## What it does

1. A FastMCP server exposing a single `get_secret` tool is uploaded to
   `/opt/mcp-server/server.py` in the sandbox and started by the framework
   on `http://0.0.0.0:8000/mcp`. The server itself is declared in
   `tasks/hello-mcp/task.toml`
2. An OpenCode agent is pointed at that URL via its config.
3. The agent calls `get_secret`, writes the result to `/app/secret.txt`.
4. `tests/test.sh` runs a pytest check that compares the file contents
   against the expected secret.

## Running it

```bash
# Install the env locally
prime env install hello-mcp-harbor

# Single rollout against your configured agent
prime eval run hello-mcp-harbor
```

## Files

```
hello_mcp_harbor/
├── hello_mcp_harbor.py      # load_environment + HelloMCPHarborEnv + launch commands
├── mcp_server/
│   └── server.py            # FastMCP server (1 tool)
├── tasks/hello-mcp/
│   ├── task.toml            # pure Harbor MCPServerConfig declaration
│   ├── instruction.md       # what the agent is told to do
│   ├── solution/solve.sh    # oracle solution
│   └── tests/
│       ├── test.sh          # verifier entrypoint (pytest + reward.txt)
│       └── test_outputs.py  # assertion
├── pyproject.toml
└── README.md
```

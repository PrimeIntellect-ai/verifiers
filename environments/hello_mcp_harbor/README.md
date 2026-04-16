# hello-mcp-harbor

Smallest runnable `HarborEnv` that exercises the framework-managed MCP
server lifecycle end-to-end. Adapted from Harbor's own
[`examples/tasks/hello-mcp`](https://github.com/laude-institute/harbor/tree/main/examples/tasks/hello-mcp),
collapsed to a single container (no docker-compose sidecar).

## What it does

1. A FastMCP server exposing a single `get_secret` tool is uploaded to
   `/opt/mcp-server/server.py` in the sandbox and started by the framework
   on `http://127.0.0.1:8000/mcp`. The server itself is declared in
   `tasks/hello-mcp/task.toml` using pure Harbor `MCPServerConfig` shape;
   the matching `HarborMCPLauncher` (command, phases, health probe) lives
   on `HelloMCPHarborEnv` in Python.
2. An OpenCode agent is pointed at that URL via the `$HARBOR_MCP_MCP_SERVER_URL`
   env var that `HarborEnv.build_env_vars` publishes automatically.
3. The agent calls `get_secret`, writes the result to `/app/secret.txt`.
4. `tests/test.sh` runs a pytest check that compares the file contents
   against the expected secret.

## task.toml stays Harbor-pure

```toml
[[environment.mcp_servers]]
name = "mcp-server"
transport = "streamable-http"
url = "http://localhost:8000/mcp"
```

That's exactly the shape `harbor.models.task.config.MCPServerConfig` takes —
no HarborEnv-specific keys live in the task file. Everything HarborEnv
needs to *launch* the server (the command to run, which user to run as,
which phases it should be up for, and how to probe readiness) is declared
on the Python class as a `HarborMCPLauncher`:

```python
_MCP_LAUNCHERS = {
    "mcp-server": HarborMCPLauncher(
        command="python /opt/mcp-server/server.py",
        phases=["agent"],
        healthcheck=HarborMCPHealthcheck(retries=10, start_period_sec=3.0),
    ),
}
```

The dict key matches the `name` field in task.toml — that's the only
coupling. If you later want to run this task under native `harbor run`
with a docker-compose sidecar, the task.toml doesn't need to change.

## Running it

```bash
# Install the env locally
prime env install hello-mcp-harbor

# Single rollout against your configured agent
prime eval run hello-mcp-harbor
```

Expected outcome: reward `1.0` on a successful rollout. On first run the
rollout takes ~60–90s (most of that is the initial OpenCode + fastmcp
install inside the sandbox).

## Why this task

It's deliberately the smallest possible thing that can fail in an
interesting way:

- If MCP lifecycle is broken → health check fails, rollout fails fast
  with the server's stderr in the error message.
- If env-var publishing is broken → OpenCode config has a literal
  `$HARBOR_MCP_MCP_SERVER_URL` instead of a URL, `get_secret` is unknown
  to the agent, secret file never written, reward 0.
- If phase handling is broken → the launcher declares `phases=["agent"]`,
  so `restart_mcp_for_phase("verifier")` inside `compute_reward` should
  stop this server before tests run.

## Portability to native Harbor

Because task.toml is pure Harbor format, the task file itself is portable
to `harbor run`. The missing piece for native Harbor would be a
`environment/docker-compose.yaml` declaring a sidecar at
`http://mcp-server:8000/mcp` — which this env deliberately does *not* ship
(Prime sandboxes are single-container). If you want to run the same task
under native Harbor, add a `docker-compose.yaml` alongside and it'll work
there too without any task.toml change.

## Files

```
hello_mcp_harbor/
├── hello_mcp_harbor.py      # load_environment + HelloMCPHarborEnv + launchers
├── mcp_server/
│   └── server.py            # FastMCP server (1 tool)
├── tasks/hello-mcp/
│   ├── task.toml            # pure Harbor MCPServerConfig declaration
│   ├── instruction.md       # what the agent is told to do
│   ├── solution/solve.sh    # oracle solution (needs docker-compose sidecar
│   │                        #   to run under native `harbor run -a oracle`)
│   └── tests/
│       ├── test.sh          # verifier entrypoint (pytest + reward.txt)
│       └── test_outputs.py  # assertion
├── pyproject.toml
└── README.md
```

# MCPEnv

Source: `docs/environments.md`.

`MCPEnv` connects a Verifiers environment to external MCP servers. It is useful
when the model should call tools served by another process, such as search,
fetch, browser, database, or custom domain tools.

The environment configuration declares each MCP server command, arguments,
environment variables, and description so the server can be launched as part of
the environment lifecycle.

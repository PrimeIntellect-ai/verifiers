# mcp-search-env

v1 `vf.Env` example for MCP-backed tool use. The taskset asks short synthetic
search questions, and the bundled stdio MCP server exposes `search_records` and
`read_record` over a small stable record corpus.

```bash
prime eval run mcp-search-env -m openai/gpt-4.1-mini -n 5 -r 1
```

Configuration belongs under v1 sections:

```toml
[env.taskset]
max_turns = 6
```

The environment does not mirror project documentation. The bundled MCP server is
only a compact search fixture so the package stays self-contained and easy to
maintain.

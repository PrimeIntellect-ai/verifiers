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

- **Type**: multi-turn | tool use
- **Parser**: N/A
- **Rubric overview**: Judge-based evaluation using gpt-4.1-mini

### MCP Tools

This environment integrates with MCP (Model Context Protocol) servers to provide tool-calling capabilities. By default, it includes:

#### Exa & Fetch Tools

- **Exa MCP Server**: Search and discovery tool for finding relevant web content (via Smithery)
  - Command: `npx -y @smithery/cli@latest run exa --key <KEY> --profile <PROFILE>`
  - Note: Authentication is handled via Smithery CLI key/profile

- **Fetch MCP Server**: Fetches and retrieves web content from URLs
  - Command: `uvx mcp-server-fetch`
  - No API key required

#### Browserbase Tools

- **Browserbase MCP Server**: Browser automation for interacting with web pages using AI-powered navigation
  - Command: `npx @browserbasehq/mcp-server-browserbase`
  - Required environment variables:
    - `BROWSERBASE_API_KEY`
    - `BROWSERBASE_PROJECT_ID`
    - `GEMINI_API_KEY`

**Customizing Tools:**

You can pass custom MCP server configurations via the `mcp_servers` argument to `load_environment()`:

```python
custom_servers = [
    {
        "name": "my-server",
        "transport": "stdio",
        "command": "npx",
        "args": ["my-mcp-server"],
        "env": {"API_KEY": "your_key"},
        "description": "Custom MCP server"
    }
]
env = load_environment(mcp_servers=custom_servers)
```

### Quickstart

**Prerequisites:**

Export the required API keys for the judge LLM and MCP tools:

```bash
# Required for judge-based evaluation
export OPENAI_API_KEY=your_openai_key

# Required for Exa MCP server (via Smithery)
export SMITHERY_KEY=your_smithery_key
export SMITHERY_PROFILE=your_smithery_profile

# Required for Browserbase MCP server (browser automation)
export BROWSERBASE_API_KEY=your_browserbase_key
export BROWSERBASE_PROJECT_ID=your_project_id
export GEMINI_API_KEY=your_gemini_key
```

**Note:** Not all API keys are required for every task. The Fetch MCP server works without any API key. Only export the keys for the tools you intend to use.

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

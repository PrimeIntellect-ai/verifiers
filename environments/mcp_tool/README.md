# mcp-tool

### Overview
- **Environment ID**: `mcp-tool`
- **Short description**: Universal multi-turn tool environment that spawns any MCP servers in Prime Intellect sandboxes and exposes them as tools to models
- **Tags**: mcp, tools, multi-turn, sandbox, prime, universal

### Datasets
- **Primary dataset(s)**: Configurable via `dataset_name` parameter (defaults to GSM8K)
- **Source links**: Uses `load_example_dataset` from verifiers
- **Split sizes**: Configurable via `num_examples` parameter

### Task
- **Type**: multi-turn tool use
- **Parser**: Default `Parser` with tool execution support
- **Rubric overview**: `ToolRubric` for MCP tool execution success and format adherence

### Quickstart

**Python FastMCP Server:**
```bash
uv run vf-eval mcp-tool -a '{
  "mcp_launch_commands": [
    {
      "name": "calculator", 
      "script": "/workspace/calc_server.py"
    }
  ],
  "num_examples": 10
}'
```

**Remote MCP Server (HTTP):**
```bash
uv run vf-eval mcp-tool -a '{
  "mcp_launch_commands": [
    {
      "name": "weather", 
      "url": "https://weather-api.example.com/mcp",
      "headers": {"Authorization": "Bearer YOUR_TOKEN"}
    }
  ],
  "num_examples": 10
}'
```

**Multiple MCP Servers:**
```bash
uv run vf-eval mcp-tool -a '{
  "mcp_launch_commands": [
    {
      "name": "calculator",
      "script": "/workspace/calc_server.py"
    },
    {
      "name": "weather",
      "url": "https://weather-api.example.com/mcp"
    },
    {
      "name": "legacy_python",
      "command": ["python", "/workspace/legacy_server.py"]
    }
  ],
  "num_examples": 10
}'
```

**Custom Configuration:**
```bash
uv run vf-eval mcp-tool \
  -m gpt-4o-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{
    "mcp_launch_commands": [
      {
        "name": "calculator",
        "script": "/workspace/calc_server.py",
        "env": {"DEBUG": "1"}
      }
    ],
    "sandbox_config": {"cpu_cores": 2, "memory_gb": 4}
  }'
```

Notes:
- **Universal MCP Support**: Works with any MCP server that follows the Model Context Protocol
- **FastMCP Integration**: Uses FastMCP Python library for efficient MCP client/server communication
- Requires Prime CLI configuration: `prime config set-api-key`
- Uses Python 3.12 slim sandbox image with FastMCP installed
- MCP servers run in isolated Prime Intellect sandboxes
- Supports Python scripts, HTTP/SSE endpoints, and stdio commands
- Automatic tool discovery via MCP protocol
- Dynamic tool schema generation from MCP server metadata
- See [Prime Sandboxes documentation](https://docs.primeintellect.ai/sandboxes) for details
- Automatic cleanup prevents resource leaks

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `mcp_launch_commands` | List[Dict] | — | **Required**. List of MCP server configurations |
| `dataset_name` | str | `"gsm8k"` | Dataset to load for evaluation |
| `split` | str | `"train"` | Dataset split to use |
| `num_examples` | int | `100` | Number of examples to use (-1 for all) |
| `sandbox_config` | Dict | `{}` | Prime sandbox configuration (cpu_cores, memory_gb, disk_size_gb) |
| `max_turns` | int | `10` | Maximum number of conversation turns |
| `prime_api_key` | str | `None` | Prime API key (or use CLI config) |

### MCP Server Configuration
Each MCP server configuration supports multiple formats:

**Python Script (auto-uploaded):**
```json
{
  "name": "calculator",
  "script": "/workspace/calc_server.py"
}
```

**Remote HTTP/SSE Server:**
```json
{
  "name": "weather",
  "url": "https://weather-api.example.com/mcp",
  "headers": {"Authorization": "Bearer TOKEN"}
}
```

**Stdio Command:**
```json
{
  "name": "legacy",
  "command": ["python", "/workspace/legacy_server.py"],
  "env": {"DEBUG": "1"},
  "workdir": "/tmp"
}
```

### Script Auto-Upload
The environment automatically uploads local script files to the sandbox:

- **`/workspace/script.py`** → Looks for `script.py` in the environment directory
- **`relative/path.py`** → Looks for `relative/path.py` relative to environment directory  
- **`/absolute/local/path.py`** → Uploads from absolute local path (if exists)

Scripts are automatically made executable and parent directories are created.

### Supported MCP Servers
The environment supports **any MCP server** that follows the Model Context Protocol. Popular servers include:

| Server | Package | Description |
|--------|---------|-------------|
| **Filesystem** | `@modelcontextprotocol/server-filesystem` | File system operations (read, write, list) |
| **Memory** | `@modelcontextprotocol/server-memory` | Persistent memory storage and retrieval |
| **Git** | `@modelcontextprotocol/server-git` | Git repository operations |
| **GitHub** | `@modelcontextprotocol/server-github` | GitHub API integration |
| **Slack** | `@modelcontextprotocol/server-slack` | Slack messaging and channel management |
| **Postgres** | `@modelcontextprotocol/server-postgres` | PostgreSQL database operations |
| **SQLite** | `@modelcontextprotocol/server-sqlite` | SQLite database operations |
| **Puppeteer** | `@modelcontextprotocol/server-puppeteer` | Web scraping and automation |
| **Google Drive** | `@modelcontextprotocol/server-gdrive` | Google Drive file operations |
| **Brave Search** | `@modelcontextprotocol/server-brave-search` | Web search capabilities |

**Custom Servers**: You can also use any custom MCP server by providing the appropriate command and setup instructions.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Weighted combination of tool execution success and format adherence |
| `tool_execution_success` | Percentage of MCP tool calls executed without errors |

### Prerequisites
1. **Prime CLI**: `pip install prime>=0.3.26` and `prime config set-api-key`
2. **API Access**: Valid Prime Intellect API credentials
3. **Sandbox Access**: See [Prime Sandboxes documentation](https://docs.primeintellect.ai/sandboxes) for setup and pricing

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval mcp-tool -a '{"mcp_launch_commands": [{"name": "filesystem", "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}]}'</code> to generate one.</p>
<!-- vf:end:reports -->
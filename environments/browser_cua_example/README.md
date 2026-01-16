# Browser CUA Mode Example

A simple example environment demonstrating **CUA (Computer Use Agent) mode** browser automation using [Browserbase](https://browserbase.com).

CUA mode uses vision-based primitives to control the browser through screenshots, similar to how a human would interact with a screen.

## How CUA Mode Works

CUA mode provides low-level vision-based operations:
- **click(x, y)**: Click at screen coordinates
- **type_text(text)**: Type text into focused element
- **scroll(direction)**: Scroll the page
- **screenshot()**: Capture current screen state
- **navigate(url)**: Go to a URL

The agent sees screenshots and decides which actions to take based on visual understanding.

## Prerequisites: CUA Server

CUA mode requires a running CUA server that handles browser automation. The server is located at:
```
verifiers/envs/integrations/browser_env/cua-server/
```

### Starting the CUA Server

```bash
cd verifiers/envs/integrations/browser_env/cua-server

# Start the server (installs dependencies automatically if needed)
./start.sh
```

The server runs on `http://localhost:3000` by default. Use `./start.sh --port 8080` for a custom port.

## Installation

```bash
# Install browser extras
uv pip install -e ".[browser]"

# Install this example environment
uv pip install -e ./environments/browser_cua_example
```

## Configuration

### Required Environment Variables

```bash
# Browserbase credentials
export BROWSERBASE_API_KEY="your-api-key"
export BROWSERBASE_PROJECT_ID="your-project-id"

# API key for agent model
export OPENAI_API_KEY="your-openai-key"
```

Note: CUA mode does NOT require `MODEL_API_KEY` since it doesn't use Stagehand.

## Usage

1. **Start the CUA server** (in a separate terminal):
   ```bash
   cd verifiers/envs/integrations/browser_env/cua-server && ./start.sh
   ```

2. **Run the evaluation**:
   ```bash
   prime eval run browser-cua-example -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
   ```

### Custom Server URL

If running the CUA server on a different port:
```bash
prime eval run browser-cua-example -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"server_url": "http://localhost:8080"}'
```

## Environment Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `max_turns` | `10` | Maximum conversation turns |
| `judge_model` | `"gpt-4o-mini"` | Model for task completion judging |
| `server_url` | `"http://localhost:3000"` | CUA server URL |
| `viewport_width` | `1024` | Browser viewport width |
| `viewport_height` | `768` | Browser viewport height |
| `save_screenshots` | `False` | Save screenshots during execution |

## DOM vs CUA Mode Comparison

| Aspect | DOM Mode | CUA Mode |
|--------|----------|----------|
| **Control** | Natural language via Stagehand | Vision-based coordinates |
| **Server** | None required | CUA server required |
| **MODEL_API_KEY** | Required (for Stagehand) | Not required |
| **Best for** | Structured web interactions | Visual/complex UIs |
| **Speed** | Faster (direct DOM) | Slower (screenshots) |

## Requirements

- Python >= 3.10
- Node.js (for CUA server)
- Browserbase account with API credentials
- OpenAI API key

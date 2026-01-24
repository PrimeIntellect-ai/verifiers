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

<!-- TODO: Update this section when MODEL_API_KEY support is added to CUA server -->
Note: When running in manual server mode, ensure `OPENAI_API_KEY` is set in the terminal where the CUA server runs (Stagehand requires it internally).

## Usage

### Sandbox Mode (Default, Recommended)

By default, CUA mode automatically deploys the server to a sandbox container. No manual server setup is required:

```bash
prime eval run browser-cua-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

### Manual Server Mode (Local Development)

For local development, you can run the CUA server manually:

1. **Start the CUA server** (in a separate terminal):
   ```bash
   cd verifiers/envs/integrations/browser_env/cua-server
   export OPENAI_API_KEY="your-openai-key"
   pnpm dev
   ```

   The server runs on `http://localhost:3000` by default.

2. **Run the evaluation with sandbox disabled**:
   ```bash
   prime eval run browser-cua-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"use_sandbox": false}'
   ```

### Custom Server URL

If running the CUA server on a different port:
```bash
prime eval run browser-cua-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"use_sandbox": false, "server_url": "http://localhost:8080"}'
```

## Environment Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `max_turns` | `15` | Maximum conversation turns |
| `judge_model` | `"gpt-4o-mini"` | Model for task completion judging |
| `use_sandbox` | `True` | Auto-deploy CUA server to sandbox (recommended) |
| `server_url` | `"http://localhost:3000"` | CUA server URL (only used when `use_sandbox=False`) |
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

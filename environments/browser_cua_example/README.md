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
# Optional: export BROWSERBASE_PROJECT_ID="your-project-id"

# API key used by the CUA server's internal Stagehand session creation
export OPENAI_API_KEY="your-openai-key"
```

`OPENAI_API_KEY` must be available to the CUA server process itself. In the normal `prime eval run` flow, `BrowserEnv` forwards it into the sandbox automatically. If you run the server manually with `pnpm dev`, `docker run`, or `prime sandbox create`, you must inject `OPENAI_API_KEY` yourself into that server process.

## Usage

### Quick Test Commands

```bash
# Default - pre-built image (fastest)
prime eval run browser-cua-example -m openai/gpt-4o-mini

# Binary upload (custom server)
prime eval run browser-cua-example -m openai/gpt-4o-mini -a '{"use_prebuilt_image": false}'

# Local development
prime eval run browser-cua-example -m openai/gpt-4o-mini -a '{"use_sandbox": false}'
```

### Pre-built Prime Image (Default, Fastest)

By default, if you do not override `prebuilt_image`, CUA mode uses the repo default `browserbase/cua-server:latest` for fastest startup. The image includes the CUA server binary and all dependencies pre-installed:

```bash
prime eval run browser-cua-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

This is the recommended approach for production use. Startup is ~5-10 seconds compared to ~30-60 seconds with binary upload.

The prebuilt image path still requires `OPENAI_API_KEY` in the rollout environment, because `BrowserEnv` forwards it to the sandboxed CUA server for Stagehand session creation.

### Binary Upload Mode (Custom Server)

If you need to use a custom version of the CUA server, disable the prebuilt image to build and upload the binary at runtime:

```bash
prime eval run browser-cua-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"use_prebuilt_image": false}'
```

This mode:
1. Builds the CUA server binary via Docker (first run only)
2. Uploads the binary to a sandbox container
3. Installs dependencies (curl) in the sandbox
4. Starts the server

### Manual Server Mode (Local Development)

For local development, you can run the CUA server manually:

1. **Start the CUA server** (in a separate terminal):
   ```bash
   cd assets/templates/browserbase/cua
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
| `max_turns` | `15` | Maximum conversation turns (recommended: 50 for complex tasks) |
| `judge_model` | `"gpt-4o-mini"` | Model for task completion judging |
| `use_sandbox` | `True` | Auto-deploy CUA server to sandbox |
| `use_prebuilt_image` | `True` | Use pre-built Prime image (fastest startup) |
| `prebuilt_image` | `"browserbase/cua-server:latest"` | Prime image to use when `use_prebuilt_image=True` |
| `server_url` | `"http://localhost:3000"` | CUA server URL (only used when `use_sandbox=False`) |
| `viewport_width` | `1024` | Browser viewport width |
| `viewport_height` | `768` | Browser viewport height |
| `save_screenshots` | `False` | Save screenshots during execution |

## Execution Modes Summary

| Mode | Flag | Startup Time | Use Case |
|------|------|--------------|----------|
| **Pre-built image** (default) | None | ~5-10s | Production, fastest startup |
| **Binary upload** | `use_prebuilt_image=false` | ~30-60s | Custom server version |
| **Manual server** | `use_sandbox=false` | Instant | Local development |

## Building a Custom Prime Image

To build and push a custom CUA server image:

```bash
cd assets/templates/browserbase/cua
./build-and-push.sh bb-project-id-optional-20260326
```

The script prints the fully qualified Prime image ref when the build finishes, for example `your-user/cua-server:bb-project-id-optional-20260326` or `team-<team-id>/cua-server:bb-project-id-optional-20260326`, depending on your active Prime context.

If you run the image directly outside `BrowserEnv`, include the OpenAI secret on sandbox creation:

```bash
prime sandbox create team-<team-id>/cua-server:bb-project-id-optional-20260326 \
  --start-command "./cua-server-linux-x64" \
  --env CUA_SERVER_PORT=3000 \
  --secret OPENAI_API_KEY="$OPENAI_API_KEY"
```

Then use your custom image:
```bash
prime eval run browser-cua-example -m openai/gpt-4.1-mini -a '{"prebuilt_image": "team-<team-id>/cua-server:bb-project-id-optional-20260326"}'
```

Use the versioned tag first for validation. Once you want to move `latest`, rerun the script from the same source revision:

```bash
./build-and-push.sh latest
```

## DOM vs CUA Mode Comparison

| Aspect | DOM Mode | CUA Mode |
|--------|----------|----------|
| **Control** | Natural language via Stagehand | Vision-based coordinates |
| **Server** | None required | CUA server (auto-deployed) |
| **MODEL_API_KEY** | Required (for Stagehand) | Not required |
| **Best for** | Structured web interactions | Visual/complex UIs |
| **Speed** | Faster (direct DOM) | Slower (screenshots) |

## Requirements

- Python >= 3.10
- Browserbase account with API credentials
- OpenAI API key

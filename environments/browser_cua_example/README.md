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

### Quick Test Commands

```bash
# Default - pre-built image (fastest)
prime eval run browser-cua-example -m openai/gpt-4o-mini

# Binary upload (custom server)
prime eval run browser-cua-example -m openai/gpt-4o-mini -a '{"use_prebuilt_image": false}'

# Local development
prime eval run browser-cua-example -m openai/gpt-4o-mini -a '{"use_sandbox": false}'
```

### Pre-built Docker Image (Default, Fastest)

By default, CUA mode uses a pre-built Docker image (`deepdream19/cua-server:latest`) for fastest startup. The image includes the CUA server binary and all dependencies pre-installed:

```bash
prime eval run browser-cua-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

This is the recommended approach for production use. Startup is ~5-10 seconds compared to ~30-60 seconds with binary upload.

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
| `use_prebuilt_image` | `True` | Use pre-built Docker image (fastest startup) |
| `prebuilt_image` | `"deepdream19/cua-server:latest"` | Docker image to use when `use_prebuilt_image=True` |
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

## Building a Custom Docker Image

To build and push a custom CUA server image:

```bash
cd assets/templates/browserbase/cua
./build-and-push.sh                    # Push as :latest
./build-and-push.sh v1.0.0             # Push with version tag
DOCKERHUB_USER=myuser ./build-and-push.sh  # Use different Docker Hub user
```

Then use your custom image:
```bash
prime eval run browser-cua-example -m openai/gpt-4.1-mini -a '{"prebuilt_image": "myuser/cua-server:v1.0.0"}'
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

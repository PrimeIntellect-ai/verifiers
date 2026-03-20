# CUA Local Mode Server

A variant of the CUA (Computer Use Agent) Primitives Server designed for localhost application testing. This server provides browser automation capabilities without internet access, running alongside your local web application in a sandbox environment.

## Key Differences from Standard CUA Server

- **No `goto` action**: The browser starts at a configured `startUrl` (typically `http://localhost:3000`) and cannot navigate to external URLs
- **LOCAL mode only**: No Browserbase integration - uses local Chromium
- **Combined deployment**: Designed to run alongside your target application in the same container

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Sandbox Container                    │
│                                                         │
│  ┌─────────────────┐          ┌─────────────────────┐   │
│  │  CUA Local      │          │  Your App           │   │
│  │  Server         │          │  (Next.js, etc.)    │   │
│  │  Port 3001      │          │  Port 3000          │   │
│  └────────┬────────┘          └──────────┬──────────┘   │
│           │                              │              │
│           │    ┌─────────────────────┐   │              │
│           └───►│     Chromium        │◄──┘              │
│                │  (headless browser) │                  │
│                └─────────────────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Available Actions

| Action | Description |
|--------|-------------|
| `click(x, y, button)` | Click at coordinates |
| `double_click(x, y)` | Double-click at coordinates |
| `type(text)` | Type text into focused element |
| `keypress(keys)` | Press keyboard keys |
| `scroll(x, y, scroll_x, scroll_y)` | Scroll at position |
| `back()` | Go back in history |
| `forward()` | Go forward in history |
| `wait(time_ms)` | Wait for specified milliseconds |
| `screenshot()` | Capture current page state |

**Note**: `goto` is intentionally omitted as this mode is for testing localhost applications without internet access.

## Local Development

```bash
# Install dependencies
pnpm install

# Start the server (development mode with hot reload)
pnpm dev

# Start the server (production mode)
pnpm start
```

## Building the Binary

For sandbox deployment, you need to build a standalone binary:

```bash
# Build using Docker (recommended for linux-x64 target)
pnpm build:binary:docker

# The binary will be at: dist/sea/cua-local-server-linux-x64
```

## Docker Build

```bash
# Build the combined image (CUA server + supervisor)
docker build --platform linux/amd64 -t cua-local:latest .
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/sessions` | List active sessions |
| POST | `/sessions` | Create browser session |
| DELETE | `/sessions/:id` | Close browser session |
| GET | `/sessions/:id/state` | Get current browser state |
| POST | `/sessions/:id/action` | Execute browser action |

## Session Creation

When creating a session, you can specify:

```json
{
  "viewport": { "width": 1024, "height": 768 },
  "startUrl": "http://localhost:3000",
  "headless": true,
  "executablePath": "/usr/bin/chromium"
}
```

The browser will automatically navigate to `startUrl` after session creation.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUA_SERVER_PORT` | 3001 | Server port |
| `CUA_SERVER_HOST` | 0.0.0.0 | Server bind address |
| `LOG_LEVEL` | info | Logging level |

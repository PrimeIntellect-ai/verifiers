# CUA Primitives API Server

A Fastify server that exposes Stagehand's Computer Use Agent (CUA) browser primitives as REST endpoints, enabling external agents to control browser sessions remotely.

> **Note**: When `BrowserEnv` runs with `mode="cua"` and `use_sandbox=True` (the default), it starts whichever image is configured in `prebuilt_image`. The repo default is `browserbase/cua-server:latest`. If you publish your own image with `prime images push`, Prime stores it under your active personal or team context, for example `team-<team-id>/cua-server:<tag>`, and you pass that fully qualified ref via `prebuilt_image`.

## Automatic Sandbox Deployment

When using `BrowserEnv(mode="cua")`, the server is automatically:
1. Started from `prebuilt_image` (default: `browserbase/cua-server:latest`)
2. Accessed via curl commands inside the sandbox
3. Cleaned up when the rollout completes

If you disable `use_prebuilt_image`, BrowserEnv falls back to uploading the SEA binary into a sandbox at rollout time.

```python
# This automatically deploys the CUA server to a sandbox
env = BrowserEnv(
    mode="cua",
    dataset=dataset,
    rubric=rubric,
)
```

## Manual Usage (Local Development)

For local development or debugging, you can run the server manually:

```bash
# Start the server (with hot reload)
pnpm dev

# Or start without hot reload
pnpm start

# Custom port via environment variable
CUA_SERVER_PORT=8080 pnpm dev
```

Then configure BrowserEnv to use the manual server:

```python
env = BrowserEnv(
    mode="cua",
    use_sandbox=False,
    server_url="http://localhost:3000",
    dataset=dataset,
    rubric=rubric,
)
```

## Building and Publishing Images

Build a local runtime image from this template:

```bash
docker build --platform linux/amd64 -f Dockerfile.runtime -t cua-server:local .
```

Publish a versioned tag to Prime Images:

```bash
./build-and-push.sh bb-project-id-optional-20260326
```

The script waits for the remote build to finish and prints the fully qualified Prime image ref to use in `prebuilt_image`, for example `your-user/cua-server:bb-project-id-optional-20260326` or `team-<team-id>/cua-server:bb-project-id-optional-20260326`, depending on your active Prime context.

If you want to move `latest`, rerun the same source revision with the `latest` tag:

```bash
./build-and-push.sh latest
```

To run the image directly in a Prime sandbox, include the server-side OpenAI key:

```bash
prime sandbox create team-<team-id>/cua-server:bb-project-id-optional-20260326 \
  --start-command "./cua-server-linux-x64" \
  --env CUA_SERVER_PORT=3000 \
  --secret OPENAI_API_KEY="$OPENAI_API_KEY"
```

`OPENAI_API_KEY` is required by the server process when creating Stagehand-backed sessions. `BROWSERBASE_API_KEY` and the optional Browserbase project id are provided per request to `POST /sessions`, not as sandbox environment variables.

## Architecture

```
External Agent -> Fastify API -> BrowserSessionManager -> Stagehand Page -> Browser
```

## Prerequisites

```bash
npm install @browserbasehq/stagehand fastify
```

## Environment Variables

Server process variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUA_SERVER_PORT` | `3000` | Server port |
| `CUA_SERVER_HOST` | `0.0.0.0` | Server host |
| `OPENAI_API_KEY` | None | Required when creating sessions; Stagehand uses it internally inside the CUA server |

## API Endpoints

### Health Check

```bash
GET /health
```

Returns server status and active session count.

### List Sessions

```bash
GET /sessions
```

Returns array of active session IDs.

### Create Session

```bash
POST /sessions
Content-Type: application/json

{
  "env": "BROWSERBASE",           // or "LOCAL"
  "browserbaseApiKey": "...", // required for BROWSERBASE sessions
  "browserbaseProjectId": "...", // optional; Browserbase default project is used when omitted
  "viewport": {
    "width": 1024,
    "height": 768
  }
}
```

Returns:
```json
{
  "sessionId": "session_1234567890_abc123",
  "state": {
    "screenshot": "base64...",
    "url": "about:blank",
    "viewport": { "width": 1024, "height": 768 }
  }
}
```

### Get Session State

```bash
GET /sessions/:id/state
```

Returns current browser state (screenshot, URL, viewport).

### Close Session

```bash
DELETE /sessions/:id
```

Closes the browser and removes the session.

### Execute Action

```bash
POST /sessions/:id/action
Content-Type: application/json

{
  "type": "click",
  "x": 100,
  "y": 200
}
```

Returns:
```json
{
  "success": true,
  "state": {
    "screenshot": "base64...",
    "url": "https://example.com",
    "viewport": { "width": 1024, "height": 768 }
  }
}
```

## Available Actions

### Mouse Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `click` | `x`, `y`, `button?`, `clickCount?` | Click at coordinates |
| `double_click` | `x`, `y` | Double-click at coordinates |
| `tripleClick` | `x`, `y` | Triple-click at coordinates |
| `drag` | `path: [{x, y}, ...]` | Drag along path |
| `move` | - | No-op (cursor visualization) |

### Keyboard Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `type` | `text` | Type text into focused element |
| `keypress` | `keys` (string or array) | Press keyboard keys |

### Navigation Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `goto` | `url` | Navigate to URL |
| `back` | - | Go back in history |
| `forward` | - | Go forward in history |
| `scroll` | `x?`, `y?`, `scroll_x?`, `scroll_y?` | Scroll the page |

### Utility Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `wait` | `timeMs?` (default: 1000) | Wait for duration |
| `screenshot` | - | No-op (always returned in response) |

## Example Usage

```bash
# Create a Browserbase-backed session
SESSION=$(curl -s -X POST http://localhost:3000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "env": "BROWSERBASE",
    "browserbaseApiKey": "'"$BROWSERBASE_API_KEY"'",
    "browserbaseProjectId": "'"$BROWSERBASE_PROJECT_ID"'"
  }' | jq -r '.sessionId')

# Navigate to a website
curl -X POST http://localhost:3000/sessions/$SESSION/action \
  -H "Content-Type: application/json" \
  -d '{"type": "goto", "url": "https://example.com"}'

# Click a button
curl -X POST http://localhost:3000/sessions/$SESSION/action \
  -H "Content-Type: application/json" \
  -d '{"type": "click", "x": 150, "y": 300}'

# Type into an input
curl -X POST http://localhost:3000/sessions/$SESSION/action \
  -H "Content-Type: application/json" \
  -d '{"type": "type", "text": "Hello, World!"}'

# Press Enter
curl -X POST http://localhost:3000/sessions/$SESSION/action \
  -H "Content-Type: application/json" \
  -d '{"type": "keypress", "keys": "Enter"}'

# Scroll down
curl -X POST http://localhost:3000/sessions/$SESSION/action \
  -H "Content-Type: application/json" \
  -d '{"type": "scroll", "x": 640, "y": 360, "scroll_y": 500}'

# Close the session
curl -X DELETE http://localhost:3000/sessions/$SESSION
```

## Response Format

All action responses include the full browser state:

```typescript
interface ActionResponse {
  success: boolean;
  error?: string;
  state: {
    screenshot: string;  // base64 PNG
    url: string;
    viewport: {
      width: number;
      height: number;
    };
  };
}
```

## Error Handling

Errors return appropriate HTTP status codes:

- `404` - Session not found
- `500` - Action execution failed

```json
{
  "error": "Session session_123 not found",
  "code": "SESSION_NOT_FOUND"
}
```

## File Structure

```
cua-server/
├── index.ts           # Entry point
├── server.ts          # Fastify routes
├── sessionManager.ts  # Browser session lifecycle
├── actionExecutor.ts  # CUA primitive execution
├── stateCapture.ts    # Screenshot & state helpers
├── types.ts           # TypeScript types
├── setup.sh           # Sandbox initialization script (used by CUASandboxMode)
├── package.json       # Dependencies
├── tsconfig.json      # TypeScript configuration
└── README.md          # This file
```

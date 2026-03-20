# Local Browser Example

Example environment for testing localhost web applications with CUA-style browser automation.

## Overview

This environment demonstrates how to:
- Run a web application (Next.js) alongside a CUA server in a sandbox
- Control the browser using vision-based primitives (click, type, scroll, etc.)
- Test web application UIs without internet access

## Key Features

- **No Internet Access**: The browser can only interact with the localhost application
- **No `goto` Tool**: Browser starts at the app's URL and cannot navigate externally
- **Vision-Based Control**: Uses screenshots and coordinates for interaction
- **Sandbox Isolation**: Both the app and browser run in an isolated container

## Quick Start

```bash
# Using the example Next.js app (default)
prime eval run local-browser-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

## Using Your Own Application

You can test your own Next.js (or other Node.js) application:

```bash
# Specify your app path
prime eval run local-browser-example -m openai/gpt-4.1-mini \
  -a '{"app_path": "/path/to/your/nextjs/app"}'

# With custom build/start commands
prime eval run local-browser-example -m openai/gpt-4.1-mini \
  -a '{
    "app_path": "/path/to/your/app",
    "app_build_command": "yarn && yarn build",
    "app_start_command": "yarn start"
  }'
```

## Available Tools

The agent has access to these browser primitives:

| Tool | Description |
|------|-------------|
| `click(x, y, button)` | Click at coordinates |
| `double_click(x, y)` | Double-click at coordinates |
| `type_text(text)` | Type text into focused element |
| `keypress(keys)` | Press keyboard keys |
| `scroll(x, y, scroll_x, scroll_y)` | Scroll at position |
| `back()` | Go back in history |
| `forward()` | Go forward in history |
| `wait(time_ms)` | Wait for specified milliseconds |
| `screenshot()` | Capture current page state |

**Note**: There is no `goto` tool - the browser starts at the application's URL and cannot navigate to external sites.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `app_path` | Example app | Path to your Next.js application |
| `app_port` | 3000 | Port for the application |
| `app_start_command` | `npm run start` | Command to start the app |
| `app_build_command` | `npm install && npm run build` | Command to build the app |
| `cua_server_port` | 3001 | Port for CUA server |
| `viewport_width` | 1024 | Browser viewport width |
| `viewport_height` | 768 | Browser viewport height |
| `cpu_cores` | 2 | CPU cores for sandbox |
| `memory_gb` | 4 | Memory in GB for sandbox |

## Example Tasks

The default dataset includes these example tasks:

1. **Counter Test**: Click the increment button and report the count
2. **Input Test**: Type text and submit it
3. **Selection Test**: Select an option and verify the selection

## Creating Custom Datasets

You can create your own dataset for testing your application:

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "question": [
        "Navigate to the settings page and enable dark mode",
        "Fill out the contact form with test data",
    ],
    "answer": [
        "Dark mode enabled",
        "Form submitted successfully",
    ],
    "task_id": [
        "settings-test",
        "form-test",
    ],
})
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Sandbox Container                      │
│                                                         │
│  ┌─────────────────┐          ┌─────────────────────┐   │
│  │  CUA Server     │          │  Your Application   │   │
│  │  Port 3001      │          │  Port 3000          │   │
│  └────────┬────────┘          └──────────┬──────────┘   │
│           │                              │              │
│           │    ┌─────────────────────┐   │              │
│           └───►│     Chromium        │◄──┘              │
│                │  (headless browser) │                  │
│                └─────────────────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Agent (LLM)         │
              │   - Receives screenshots
              │   - Sends actions     │
              └───────────────────────┘
```

## Requirements

- `verifiers[browser]` - Browser automation support
- `prime-sandboxes` - Sandbox execution
- Docker (for building the CUA server binary)

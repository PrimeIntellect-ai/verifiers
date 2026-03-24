# Full Browse

Browser automation environment with an expanded toolset aligned to the Wide Browse subagent traces.

## Overview

This environment demonstrates how to:
- Run a web application (Next.js) alongside a CUA server in a sandbox
- Control the browser using a unified `computer` tool with action batching
- Inspect pages using `get_page_text`, `read_page`, `find`, and `form_input`
- Test web application UIs without internet access

## Key Features

- **No Internet Access**: The browser can only interact with the localhost application
- **No `goto`/`navigate` Tool**: Browser starts at the app's URL and cannot navigate externally
- **Unified `computer` Tool**: Batched action execution matching the Wide Browse trace format
- **Page Inspection**: `get_page_text`, `read_page` (accessibility tree), and `find` (element search)
- **Form Filling**: `form_input` sets values by element ref, avoiding click-then-type timing issues
- **Sandbox Isolation**: Both the app and browser run in an isolated container

## Quick Start

```bash
# Using the example Next.js app (default)
prime eval run full-browse -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

## Using Your Own Application

You can test your own Next.js (or other Node.js) application:

```bash
# Specify your app path
prime eval run full-browse -m openai/gpt-4.1-mini \
  -a '{"app_path": "/path/to/your/nextjs/app"}'

# With custom build/start commands
prime eval run full-browse -m openai/gpt-4.1-mini \
  -a '{
    "app_path": "/path/to/your/app",
    "app_build_command": "yarn && yarn build",
    "app_start_command": "yarn start"
  }'
```

## Available Tools

### `computer(actions, user_description)` — Unified browser interaction

Executes one or more actions sequentially. Returns a screenshot after the last action.

| Action | Parameters | Description |
|--------|-----------|-------------|
| `left_click` | `coordinate: [x, y]` | Click at pixel coordinates |
| `right_click` | `coordinate: [x, y]` | Right-click at pixel coordinates |
| `double_click` | `coordinate: [x, y]` | Double-click at coordinates |
| `triple_click` | `coordinate: [x, y]` | Triple-click to select text |
| `type` | `text: string` | Type text into focused element |
| `key` | `key: string` | Press a key (e.g. `"Enter"`, `"Tab"`) |
| `scroll` | `coordinate: [x, y], direction: "up"\|"down"` | Scroll at position |
| `wait` | `duration: int` (seconds) | Wait for page load or animations |
| `screenshot` | (none) | Capture current viewport |
| `back` | (none) | Browser back |
| `forward` | (none) | Browser forward |
| `drag` | `path: [{x, y}, ...]` | Drag between points |

Example:
```json
{
  "actions": [
    {"action": "left_click", "coordinate": [500, 300]},
    {"action": "wait", "duration": 1}
  ],
  "user_description": "Clicking the search button and waiting for results"
}
```

### `get_page_text(user_description)` — Extract page text

Returns the full text content of the current page as a string. No screenshot returned.

### `read_page(user_description, filter)` — Accessibility tree

Returns the page's element tree with roles, labels, refs, coordinates, and ARIA attributes (`url=`, `expanded=`, `type=`, etc.). Each element gets a ref (e.g. `ref_42`) usable with `form_input`.

- `filter="all"` (default): Full element tree
- `filter="interactive"`: Only buttons, links, inputs, etc.
- `depth=N`: Limit tree depth (use 2-3 on large pages)
- `ref_id="ref_42"`: Focus on the subtree of a specific element

### `find(query, user_description)` — Element search

Searches for elements on the page matching a natural-language query. Returns up to 20 matches with refs and coordinates.

### `form_input(ref, value, user_description)` — Form field interaction

Sets a form field's value using its ref from `read_page` or `find`. Handles input, textarea, select, checkbox, radio, and contenteditable elements.

### `tabs_context()` — Tab state

Returns current browser tab state (always 1 tab in local mode).

**Note**: There is no `goto`/`navigate` tool — the browser starts at the application's URL and cannot navigate to external sites.

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
3. **Selection Test**: Use `read_page` to find elements, then select an option

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
              │   - Sends batched actions
              │   - Reads page text/tree
              └───────────────────────┘
```

## Requirements

- `verifiers[browser]` - Browser automation support
- `prime-sandboxes` - Sandbox execution
- Docker (for building the CUA server binary)

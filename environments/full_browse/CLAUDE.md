# Full Browse Environment - Agent Guide

This document provides guidance for agents working with the Full Browse environment to test web applications.

## Overview

The Full Browse environment allows you to test localhost web applications using an expanded browser automation toolset. The browser runs alongside your application in an isolated sandbox without internet access. Unlike the basic Local Browser environment, Full Browse provides a unified `computer` tool with action batching plus page inspection tools (`get_page_text`, `read_page`, `find`, `form_input`).

## Using the Example App

For quick testing, use the built-in example Next.js app:

```bash
prime eval run full-browse -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

## Using Your Own Application

### Requirements

Your application must:
1. Be a Node.js application (Next.js, React, Vue, etc.)
2. Have a `package.json` with build and start scripts
3. Run on a configurable port (default: 3000)

### Basic Usage

```bash
prime eval run full-browse -m openai/gpt-4.1-mini \
  -a '{"app_path": "/absolute/path/to/your/app"}'
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `app_path` | Example app | Absolute path to your application |
| `app_port` | 3000 | Port your app runs on |
| `app_build_command` | `npm install && npm run build` | Build command |
| `app_start_command` | `npm run start` | Start command |
| `cua_server_port` | 3001 | CUA server port (don't change) |
| `viewport_width` | 1024 | Browser viewport width |
| `viewport_height` | 768 | Browser viewport height |

### Example Configurations

**Next.js with yarn:**
```bash
prime eval run full-browse -m openai/gpt-4.1-mini \
  -a '{
    "app_path": "/path/to/nextjs-app",
    "app_build_command": "yarn install && yarn build",
    "app_start_command": "yarn start"
  }'
```

**Create React App:**
```bash
prime eval run full-browse -m openai/gpt-4.1-mini \
  -a '{
    "app_path": "/path/to/cra-app",
    "app_build_command": "npm install && npm run build",
    "app_start_command": "npx serve -s build -l 3000"
  }'
```

**Vue.js:**
```bash
prime eval run full-browse -m openai/gpt-4.1-mini \
  -a '{
    "app_path": "/path/to/vue-app",
    "app_build_command": "npm install && npm run build",
    "app_start_command": "npm run preview -- --port 3000"
  }'
```

## Creating Custom Tasks

### Dataset Format

Create a `Dataset` with these fields:
- `question`: The task for the agent to complete
- `answer`: Expected answer/result
- `task_id`: Unique identifier

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "question": [
        "Click the login button and describe what appears",
        "Fill the search box with 'test query' and press Enter",
    ],
    "answer": [
        "Login modal with email and password fields",
        "Search results page showing test query results",
    ],
    "task_id": [
        "login-test",
        "search-test",
    ],
})
```

### Custom Environment Setup

```python
from environments.full_browse import FullBrowseEnv, FULL_BROWSE_SYSTEM_PROMPT
import verifiers as vf

# Create custom rubric
rubric = vf.JudgeRubric(
    judge_model="gpt-4o-mini",
    judge_prompt="Your custom judge prompt...",
)

# Create environment
env = FullBrowseEnv(
    dataset=your_dataset,
    rubric=rubric,
    app_path="/path/to/your/app",
    max_turns=25,
)
```

## Available Browser Tools

The agent has access to these tools (NO `goto`/`navigate` — browser starts at your app):

### Vision + Action

```
computer(actions, user_description)
  Actions:
    left_click(coordinate=[x, y])     - Click at coordinates
    right_click(coordinate=[x, y])    - Right-click at coordinates
    double_click(coordinate=[x, y])   - Double-click at coordinates
    triple_click(coordinate=[x, y])   - Triple-click to select text
    type(text="...")                   - Type text into focused element
    key(key="Enter")                  - Press keyboard key(s)
    scroll(coordinate=[x, y],
           direction="up"|"down")     - Scroll at position
    wait(duration=2)                  - Wait seconds
    screenshot()                      - Capture screenshot
    back()                            - Browser back
    forward()                         - Browser forward
    drag(path=[{x, y}, ...])          - Drag between points
```

### Page Inspection

```
get_page_text(user_description)                  - Full page text content
read_page(user_description, filter="all"|"interactive",
         depth=N, ref_id="ref_42")          - Element tree with refs
find(query, user_description)                    - Search elements by text
form_input(ref, value, user_description)         - Fill form field by ref
tabs_context()                                   - Current tab state
```

## Troubleshooting

### App Build Failures

Check that your build command works locally:
```bash
cd /path/to/your/app
npm install && npm run build
```

### App Not Starting

1. Verify your start command serves on the correct port
2. Check if `PORT` environment variable is respected
3. Ensure `npm run start` doesn't require interactive input

### Sandbox Timeout

For large apps, increase timeouts:
```bash
prime eval run full-browse -m openai/gpt-4.1-mini \
  -a '{
    "app_path": "/path/to/app",
    "server_ready_timeout": 300,
    "app_ready_timeout": 180
  }'
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Sandbox Container                      │
│                                                         │
│  ┌─────────────────┐          ┌─────────────────────┐   │
│  │  CUA Server     │          │  Your App           │   │
│  │  :3001          │◄────────►│  :3000              │   │
│  └─────────────────┘          └─────────────────────┘   │
│           │                              │              │
│           ▼                              ▼              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Chromium (headless)                │   │
│  │          Browser viewing :3000                  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Best Practices

1. **Use descriptive task questions**: Be specific about what the agent should do
2. **Provide verifiable answers**: Make expected answers easy to check
3. **Start simple**: Test with basic interactions before complex flows
4. **Use unique IDs**: Give each UI element a unique `id` attribute for easier identification
5. **Handle loading states**: Your app should have clear loading indicators
6. **Prefer `read_page` over screenshots for form-heavy pages**: Element refs make form filling more reliable
7. **Batch related actions**: Combine click + wait in a single `computer` call to reduce round trips

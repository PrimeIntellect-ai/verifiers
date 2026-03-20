# Local Browser Environment - Agent Guide

This document provides guidance for agents working with the Local Browser environment to test web applications.

## Overview

The Local Browser environment allows you to test localhost web applications using vision-based browser automation. The browser runs alongside your application in an isolated sandbox without internet access.

## Using the Example App

For quick testing, use the built-in example Next.js app:

```bash
prime eval run local-browser-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

## Using Your Own Application

### Requirements

Your application must:
1. Be a Node.js application (Next.js, React, Vue, etc.)
2. Have a `package.json` with build and start scripts
3. Run on a configurable port (default: 3000)

### Basic Usage

```bash
prime eval run local-browser-example -m openai/gpt-4.1-mini \
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
prime eval run local-browser-example -m openai/gpt-4.1-mini \
  -a '{
    "app_path": "/path/to/nextjs-app",
    "app_build_command": "yarn install && yarn build",
    "app_start_command": "yarn start"
  }'
```

**Create React App:**
```bash
prime eval run local-browser-example -m openai/gpt-4.1-mini \
  -a '{
    "app_path": "/path/to/cra-app",
    "app_build_command": "npm install && npm run build",
    "app_start_command": "npx serve -s build -l 3000"
  }'
```

**Vue.js:**
```bash
prime eval run local-browser-example -m openai/gpt-4.1-mini \
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
from environments.local_browser_example import LocalBrowserEnv, LOCAL_CUA_SYSTEM_PROMPT
import verifiers as vf

# Create custom rubric
rubric = vf.JudgeRubric(
    judge_model="gpt-4o-mini",
    judge_prompt="Your custom judge prompt...",
)

# Create environment
env = LocalBrowserEnv(
    dataset=your_dataset,
    rubric=rubric,
    app_path="/path/to/your/app",
    max_turns=20,
)
```

## Available Browser Tools

The agent has access to these tools (NO `goto` - browser starts at your app):

```
click(x, y, button="left")   - Click at coordinates
double_click(x, y)           - Double-click at coordinates
type_text(text)              - Type text into focused element
keypress(keys)               - Press keyboard key(s)
scroll(x, y, scroll_x, scroll_y) - Scroll at position
back()                       - Browser back
forward()                    - Browser forward
wait(time_ms)                - Wait milliseconds
screenshot()                 - Capture screenshot
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
prime eval run local-browser-example -m openai/gpt-4.1-mini \
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

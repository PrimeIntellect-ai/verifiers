# Browser Environment

Vision-based browser control environment using CUA (Computer Use Agent) primitives.

## Overview

This environment provides browser automation tools for training and evaluating agents that can interact with web pages through:
- **Vision feedback**: Each action returns a screenshot of the page state
- **Tool-based actions**: Click, type, scroll, navigate, etc.
- **Session management**: One browser session per rollout

## Prerequisites

1. **Start the CUA Server**

```bash
cd environments/browser_env/cua-server
pnpm install
pnpm start
```

The server runs on `http://localhost:3000` by default.

2. **Install Dependencies**

```bash
vf-install browser_env
```

## Usage

### Basic Evaluation

```bash
vf-eval browser_env -n 5 -m gpt-4o
```

### Programmatic Usage

```python
import verifiers as vf

env = vf.load_environment("browser_env", server_url="http://localhost:3000")
```

### Custom Configuration

```python
from environments.browser_env.browser_env import load_environment

env = load_environment(
    server_url="http://localhost:3000",
    env="LOCAL",  # or "BROWSERBASE" for cloud browsers
    viewport_width=1280,
    viewport_height=720,
    max_turns=20,
    efficiency_weight=0.1,
    task_completion_weight=1.0,
)
```

## Available Tools

| Tool | Description |
|------|-------------|
| `click(x, y, button)` | Click at coordinates |
| `double_click(x, y)` | Double-click at coordinates |
| `type_text(text)` | Type text into focused element |
| `keypress(keys)` | Press keyboard key(s) |
| `scroll(x, y, scroll_x, scroll_y)` | Scroll at position |
| `goto(url)` | Navigate to URL |
| `back()` | Go back in history |
| `forward()` | Go forward in history |
| `wait(time_ms)` | Wait for specified time |
| `screenshot()` | Capture current page |

## Reward Functions

### Built-in Rewards

1. **efficiency_reward** (weight: 0.1): Penalizes long rollouts. Fewer actions = higher reward.

2. **task_completion_reward** (weight: 1.0): Placeholder for task-specific completion reward.

### Custom Rewards

Override `task_completion_reward` or add custom reward functions:

```python
async def my_custom_reward(state: vf.State, **kwargs) -> float:
    # Check if browser is on target URL
    browser_state = state.get("browser_state", {})
    current_url = browser_state.get("url", "")
    target_url = state.get("answer", "")
    
    if target_url in current_url:
        return 1.0
    return 0.0

# Add to rubric
browser_rubric.add_reward_func(my_custom_reward, weight=1.0)
```

## Environment Variables

- `BROWSERBASE_API_KEY`: API key for Browserbase cloud browsers (optional)
- `BROWSERBASE_PROJECT_ID`: Project ID for Browserbase (optional)


# TODO

- Reward function - overall browser trajectory + "how are we getting closer after each step" (hard-ish)
- Custom LLM client - via vLLM (easy)
- DOM-based option - I started with CUA (vision-based) since there seems to be more near-term market demand with full understanding verifier's has been not focused on multimodal training much yet (afaik)

## Architecture

```
┌─────────────────┐     HTTP/REST     ┌──────────────────┐
│   BrowserEnv    │ ◄──────────────► │   CUA Server      │
│   (Python)      │                   │   (Fastify/TS)   │
└─────────────────┘                   └──────────────────┘
        │                                      │
        │                                      ▼
        │                             ┌──────────────────┐
        │                             │   Stagehand V3   │
        │                             │   (Direct CDP)   │
        ▼                             └──────────────────┘
┌─────────────────┐                           │
│   Model (LLM)   │                           ▼
│   gpt-4o, etc.  │                   ┌──────────────────┐
└─────────────────┘                   │   Browser        │
                                      │   (Chrome)       │
                                      └──────────────────┘
```


# BrowserEnv Integration

`BrowserEnv` is Verifiers' Browserbase integration for browser automation tasks.

It supports two execution modes:

- **DOM mode** (`mode="dom"`): natural-language actions through Stagehand (`act`, `observe`, `extract`)
- **CUA mode** (`mode="cua"`): vision-based coordinate actions (`click`, `type_text`, `scroll`, `screenshot`)

Use this integration when your environment needs real browser interaction during rollout.

## Install

From the `verifiers` repo (or a project using `verifiers`):

```bash
uv sync --extra browser
```

Or with pip/uv pip:

```bash
uv pip install -e ".[browser]"
```

## Required Credentials

Set Browserbase credentials:

```bash
export BROWSERBASE_API_KEY="your-api-key"
export BROWSERBASE_PROJECT_ID="your-project-id"
```

For DOM mode, also set a model key used by Stagehand:

```bash
export MODEL_API_KEY="your-model-key"
```

## Quick Usage

```python
import verifiers as vf
from datasets import Dataset
from verifiers.envs.integrations.browser_env import BrowserEnv


def load_environment() -> vf.Environment:
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Go to https://example.com and tell me the page title.",
                    }
                ]
            }
        ]
    )

    async def scored(completion) -> float:
        return 1.0 if "example domain" in completion[-1]["content"].lower() else 0.0

    rubric = vf.Rubric(funcs=[scored])

    return BrowserEnv(
        mode="dom",  # switch to "cua" for vision-based interaction
        dataset=dataset,
        rubric=rubric,
        max_turns=10,
    )
```

## Mode Configuration

### DOM mode

Use DOM mode for structured websites where semantic element access is effective.

Common args:

- `mode="dom"`
- `model_api_key_var` (default: `"MODEL_API_KEY"`)
- `stagehand_model` (default: `"openai/gpt-4o-mini"`)
- `proxy_model_to_stagehand` (default: `False`)

### CUA mode

Use CUA mode for visually complex pages where coordinate-based control works better.

Common args:

- `mode="cua"`
- `use_sandbox=True` (default; auto-deploys CUA server)
- `use_prebuilt_image=True` (default; fastest startup)
- `server_url` (used when `use_sandbox=False`)
- `viewport_width` / `viewport_height`

CUA execution options:

1. **Prebuilt image** (default): fastest startup
2. **Binary upload** (`use_prebuilt_image=False`): custom server workflows
3. **Manual local server** (`use_sandbox=False`): local development/debugging

## Example Environments

For complete reference implementations, see:

- **DOM example:** `environments/browser_dom_example/`
  - `environments/browser_dom_example/browser_dom_example.py`
  - `environments/browser_dom_example/README.md`
- **CUA example:** `environments/browser_cua_example/`
  - `environments/browser_cua_example/browser_cua_example.py`
  - `environments/browser_cua_example/README.md`

These examples show end-to-end `load_environment()` setup, evaluation commands, and recommended runtime flags.

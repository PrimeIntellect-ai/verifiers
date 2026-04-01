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

When you publish an environment that uses `BrowserEnv`, list `verifiers[browser]` in that package’s `pyproject.toml` `dependencies` so installs from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments) pull the extra. Validate required variables early in `load_environment()` with `vf.ensure_keys([...])` (see [Required API Keys](../../../../docs/environments.md#required-api-keys) in the Verifiers environments guide).

## Required credentials

`BrowserEnv` reads credentials from process environment variables. Defaults:

| Variable | Required | Notes |
| -------- | -------- | ----- |
| `BROWSERBASE_API_KEY` | Yes | Browserbase API key |
| `BROWSERBASE_PROJECT_ID` | Yes | Browserbase project id |
| `MODEL_API_KEY` | DOM only | Stagehand’s LLM, unless `proxy_model_to_stagehand=True` (then the rollout client supplies the key) |
| `OPENAI_API_KEY` (or your judge provider) | If you use `JudgeRubric` / LLM judges | Not used by `BrowserEnv` itself |

Override names with `browserbase_api_key_var` and `model_api_key_var` if needed. For an LLM judge, set the provider’s API key env (often `OPENAI_API_KEY`) in the same places as the other variables—locally or on the environment—see [Browser environments](https://docs.primeintellect.ai/guides/browser-environments) for judge-oriented examples.

### Local development

Shell exports for local runs (e.g. `prime eval run`):

```bash
export BROWSERBASE_API_KEY="your-api-key"
export BROWSERBASE_PROJECT_ID="your-project-id"
```

For DOM mode (default Stagehand routing), also:

```bash
export MODEL_API_KEY="your-model-key"
```

### Environments Hub

On the [Environments Hub](https://app.primeintellect.ai/dashboard/environments), open your environment. On the **Secrets** tab, add **direct** secrets or **link** global secrets from [Keys & Secrets](https://app.primeintellect.ai/dashboard/tokens?tab=secrets). **Variables** (same tab) is for non-sensitive configuration only (see [Environment variables](https://docs.primeintellect.ai/tutorials-environments/environment-variables)); API keys belong in [Secrets](https://docs.primeintellect.ai/tutorials-environments/secrets). Hub secret and variable names must start with an uppercase letter and use only uppercase letters, digits, and underscores ([Secrets](https://docs.primeintellect.ai/tutorials-environments/secrets)); the defaults above already satisfy this.

If the same name appears as a variable, a linked global secret, and a direct secret, precedence is: variable (lowest), then linked global secret, then direct secret (highest).

Secrets and variables are injected automatically for Environment Actions, hosted evaluations, and hosted training ([Secrets](https://docs.primeintellect.ai/tutorials-environments/secrets))—do not pass secret *values* through `load_environment` arguments (`env_args` / `-a` / TOML `env_args`); use those only for non-secret options (e.g. `mode`, `num_examples`). For hosted CLI runs, `prime eval run ... --hosted --custom-secrets '{"NAME":"value"}'` is for extra per-run secrets only; routine keys should live on the environment in the Hub ([Hosted evaluations](https://docs.primeintellect.ai/tutorials-environments/hosted-evaluations)).

CLI examples:

```bash
prime env secret create owner/my-env --name BROWSERBASE_API_KEY --value "..."
prime env secret link <global-secret-id> owner/my-env
prime env var create owner/my-env --name MAX_TURNS --value 10
```

### See also

- [Browser environments](https://docs.primeintellect.ai/guides/browser-environments) — DOM/CUA workflows, judges, and training-oriented notes
- [Environments Hub getting started](https://docs.primeintellect.ai/tutorials-environments/getting-started)

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

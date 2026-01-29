# Integration Environments

Integrations with third-party environment libraries, which may require additional dependencies.

| Environment | Extra | Install Command |
|-------------|-------|-----------------|
| `TextArenaEnv` | `ta` | `uv add 'verifiers[ta]'` |
| `ReasoningGymEnv` | `rg` | `uv add 'verifiers[rg]'` |
| `BrowserEnv` | `browser` | `uv add 'verifiers[browser]'` |

## TextArenaEnv

Wrapper for text-based [TextArena](https://github.com/LeonGuertler/TextArena) game environments. Handles game state management, observation parsing, and turn-based interaction. Currently optimized for Wordle but extensible to other single-player TextArena games.

## ReasoningGymEnv

Wrapper for [reasoning-gym](https://github.com/open-thought/reasoning-gym) procedural datasets. Supports single datasets via name string or composite mixtures via `DatasetSpec` configuration. Uses reasoning-gym's built-in scoring for reward computation.

## BrowserEnv

Unified browser automation environment supporting two modes:

- **DOM mode**: Natural language operations via [Stagehand SDK](https://github.com/browserbase/stagehand)
- **CUA mode**: Vision-based primitives via HTTP server

### Quick Start

```python
from verifiers.envs.integrations.browser_env import BrowserEnv
from datasets import Dataset
import verifiers as vf

# Create your dataset
dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": "Navigate to example.com and find the main heading"}]},
])

# Create a rubric
rubric = vf.Rubric(funcs=[my_reward_func])

# DOM mode (natural language)
env = BrowserEnv(
    mode="dom",
    dataset=dataset,
    rubric=rubric,
)

# CUA mode (vision-based) - auto-deploys server to sandbox (no setup required)
env = BrowserEnv(
    mode="cua",
    dataset=dataset,
    rubric=rubric,
)
```

### DOM Mode Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stagehand_model` | `"openai/gpt-4o-mini"` | Model Stagehand uses for page understanding |
| `model_api_key` | `MODEL_API_KEY` env | API key for Stagehand's model |
| `proxy_model_to_stagehand` | `False` | Route LLM calls through verifiers client |

#### `proxy_model_to_stagehand` Flag

Controls how Stagehand's internal LLM calls (for `observe`, `act`, `extract`) are routed:

- **`False` (default)**: Stagehand uses its own configured model (`stagehand_model`) with the `model_api_key`. Best for production where you want Stagehand to use a fast/cheap model (e.g., `gpt-4o-mini`) independently of the agent model.

- **`True`**: Stagehand's LLM calls are routed through the same model/endpoint as the verifiers client. The agent's `api_key` and `base_url` are injected into Stagehand tool calls. Useful for:
  - Using a single model for both agent reasoning and browser understanding
  - Routing through custom API endpoints (e.g., vLLM, custom inference servers)
  - Training scenarios where you want consistent model usage

### CUA Mode Options

CUA mode automatically deploys the CUA server to Browserbase sandboxes using a pre-built Docker image. **No manual setup is required** - just set your environment variables and run.

#### Execution Modes

| Mode | Parameter | Startup | Use Case |
|------|-----------|---------|----------|
| **Pre-built Image** (default) | `use_prebuilt_image=True` | ~5-10s | Production, fastest startup |
| **Binary Upload** | `use_prebuilt_image=False` | ~30-60s | Custom server modifications |
| **Manual Server** | `use_sandbox=False` | Manual | Local development/debugging |

#### Default Behavior

By default, CUA mode uses the pre-built Docker image (`deepdream19/cua-server:latest`) which is automatically deployed to Browserbase sandboxes:

```python
env = BrowserEnv(
    mode="cua",
    dataset=dataset,
    rubric=rubric,
    # These are the defaults (no need to specify):
    # use_sandbox=True,
    # use_prebuilt_image=True,
)
```

#### Manual Server Mode (Optional)

For local development or debugging, you can disable sandbox mode and run the CUA server manually:

```python
env = BrowserEnv(
    mode="cua",
    dataset=dataset,
    rubric=rubric,
    use_sandbox=False,  # Disable automatic sandbox deployment
    cua_server_url="http://localhost:3001",  # Point to local server
)
```

To start the server locally:

```bash
cd assets/templates/browserbase/cua
pnpm install
./start.sh
```

### Environment Variables

```bash
BROWSERBASE_API_KEY         # Browserbase cloud API key
BROWSERBASE_PROJECT_ID      # Browserbase cloud project
MODEL_API_KEY               # For DOM mode LLM calls (Stagehand's model)
OPENAI_API_KEY              # For LLM judge evaluation
```

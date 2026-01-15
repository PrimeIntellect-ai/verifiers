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
from verifiers.envs.integrations.browser_env import load_environment

# DOM mode (natural language)
env = load_environment(mode="dom", benchmark="gaia")

# CUA mode (vision-based) - requires starting CUA server first
env = load_environment(mode="cua", benchmark="webvoyager")
```

### Benchmarks

- `smoke_test`: Basic navigation test (1 task)
- `gaia`: GAIA web tasks (difficulty: "easy", "hard")
- `webvoyager`: WebVoyager navigation tasks (643 tasks)
- `onlineMind2Web`: Mind2Web tasks (difficulty: "easy", "medium", "hard")

### CUA Server Setup

For CUA mode, start the TypeScript server first:

```bash
cd verifiers/envs/integrations/browser_env/cua-server
pnpm install
./start.sh
```

### Environment Variables

```bash
BROWSERBASE_API_KEY         # Browserbase cloud API key
BROWSERBASE_PROJECT_ID      # Browserbase cloud project
MODEL_API_KEY               # For DOM mode LLM calls
OPENAI_API_KEY              # For LLM judge evaluation
```

# Integration Environments

Integrations with third-party environment libraries, which may require additional dependencies.

| Environment | Extra | Install Command |
|-------------|-------|-----------------|
| `TextArenaEnv` | `ta` | `uv add 'verifiers[ta]'` |
| `ReasoningGymEnv` | `rg` | `uv add 'verifiers[rg]'` |

## TextArenaEnv

Wrapper for text-based [TextArena](https://github.com/LeonGuertler/TextArena) game environments. Handles game state management, observation parsing, and turn-based interaction. Currently optimized for Wordle but extensible to other single-player TextArena games.

## ReasoningGymEnv

Wrapper for [reasoning-gym](https://github.com/open-thought/reasoning-gym) procedural datasets. Supports single datasets via name string or composite mixtures via `DatasetSpec` configuration. Uses reasoning-gym's built-in scoring for reward computation.


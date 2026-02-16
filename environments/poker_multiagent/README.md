# poker-multiagent

Single-hand no-limit Texas Hold'em environment built on `MultiAgentEnv`.

## Overview
- Configurable number of players (`2-9`, default `4`).
- Real 52-card deck with deterministic seeded shuffling.
- Blinds are enabled by default (`small_blind=1`, `big_blind=2`).
- Shared tool for all actors: `take_action(action, amount?)`.
- Canonical actions: `fold`, `check`, `call`, `raise` (`raise_to` semantics).
- Invalid in-tool actions are treated as folds.
- Hand ends on folds to one player or at showdown.
- Optional debug file logging via `hand_log_path` env arg (writes full hand state, hole cards, and showdown scoring).

## Quickstart
```bash
prime eval run poker-multiagent
```

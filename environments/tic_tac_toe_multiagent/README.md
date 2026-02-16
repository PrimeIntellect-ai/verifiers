# tic-tac-toe-multiagent

Two-agent tic-tac-toe environment built on `MultiAgentEnv`.

## Overview
- Two actors (`player_x`, `player_o`) alternate turns.
- One tool is exposed: `make_move(position)`.
- Board state is stored in rollout state and injected as a hidden tool arg.
- The rollout ends early when a win/draw sets `state["final_env_response"]`.

## Quickstart
```bash
prime eval run tic-tac-toe-multiagent
```

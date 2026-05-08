# MultiTurnEnv

Source: `docs/environments.md`.

`MultiTurnEnv` supports custom conversation loops where the environment can add
messages after model turns, inspect tool calls, stop on task-specific
conditions, and score the full trajectory.

Use it when a task needs stateful interaction that goes beyond one assistant
response.

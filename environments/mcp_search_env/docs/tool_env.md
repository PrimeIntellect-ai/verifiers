# ToolEnv

Source: `docs/environments.md`.

`ToolEnv` exposes Python functions as tools during a rollout. Tool definitions
come from callable signatures and docstrings, and the model can call those
tools while solving the task.

Use `StatefulToolEnv` when per-rollout state must persist across tool calls,
such as sandbox handles, sessions, or database connections.

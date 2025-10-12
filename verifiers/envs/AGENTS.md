# Environments -- Usage Patterns

**WIP** -- subject to change.

The following environments need `envs` extras installed:
- TextArenaEnv
- SandboxEnv
- PythonEnv

```bash
uv add 'verifiers[envs]'
```
or 
```bash
uv sync --extra envs
```
if developing in the `verifiers` repo.

## Usage Patterns

Avoid storing global state in the environment. Instead, use the state argument to the environment's methods. 

### ToolEnv

ToolEnv is for tools that are stateless, idempotent, and can be passed as Python functions, with all arguments exposed to the model.

For managing additional state that should be passed to tools, use StatefulToolEnv and the `update_tool_args` method.

### StatefulToolEnv

StatefulToolEnv is for tools that require state, such as a sandbox ID.

Use StatefulToolEnv and the `update_tool_args` method to manage additional state that should be passed to tools.

See `verifiers/envs/sandbox_env.py` for an example.

### SandboxEnv

SandboxEnv is for environments that require a sandboxed container, and is configured for `prime` sandboxes. You can use similar patterns for other sandbox providers.

All sandbox setup logic should be included in the start command, and queued via `setup_state`, but not awaited. Instead, you should await all required resources only when they are first needed (e.g. a tool call to the sandbox). This allows provisioning logic to run in the background, and be overlapped with the model's rollout, only blocking when resources are needed but not yet ready.

See `verifiers/envs/python_env.py` for an example.

### TextArenaEnv

TextArenaEnv is for text-based environments, such as games or simulations.

It requires `textarena` and `nltk` to be installed.

For adding new `TextArena` environments, investigate the `textarena` source code and determine the observation format which will be rendered. Often you will want to re-render the observation via `feedback_fn` for showing to the model, as `verifiers` currently does not allow overwriting past messages in a rollout (only concatenation), and many `textarena` games give responses which express the full game state rather than just the turn-level "diff". 
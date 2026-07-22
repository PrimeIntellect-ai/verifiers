# NeMo Gym

verifiers supports NeMo Gym resources-server tasks through `NeMoGymTaskset`. A
verifiers harness owns the model and agent loop, while Gym remains responsible for
per-rollout state, tools, and verification.

This is not native NeMo Gym agent execution. A custom Gym agent is replaced by the
selected verifiers harness, so use native Gym rollout collection when that agent is
part of the benchmark semantics.

## Weather example

Run the five bundled tasks with any MCP-capable verifiers harness:

```bash
uv run eval nemo-gym-weather --env.agent.harness.id bash --no-push -n 5
```

The taskset installs the published `nemo-gym==0.4.0` package in an isolated script
environment and starts its `ExampleMCPWeatherResourcesServer` for the duration of the
evaluation. It does not start Gym's agent, model, Ray, or head-server stack.

A taskset can manage another server shipped in that package by naming its resource
class:

```python
class Taskset(NeMoGymTaskset):
    resource_server = "resources_servers.my_server.app:MyResourcesServer"
```

Export `NeMoGymEnv` beside that taskset so Verifiers owns its startup and cleanup.

## Custom resources server

The generic taskset connects to an existing resources server. Point it at the Gym
JSONL dataset and server URL:

```toml
model = "openai/gpt-5-mini"

[env.taskset]
id = "nemo-gym"
dataset_path = "/path/to/tasks.jsonl"

[env.taskset.task]
resources_url = "http://127.0.0.1:8000"

[env.agent.harness]
id = "bash"
```

Validate and run the configuration with:

```bash
uv run eval @ config.toml --dry-run
uv run eval @ config.toml
```

Every non-empty JSONL row must contain a `responses_create_params` object. The exact
row is sent to `POST /seed_session` before the rollout and to `POST /verify` with the
completed response when scoring. Tools can be returned as MCP metadata by
`/seed_session`, or declared as Responses function schemas and exposed by matching
resources-server endpoints.

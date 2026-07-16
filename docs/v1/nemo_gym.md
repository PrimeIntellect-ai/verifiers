# NeMo Gym

verifiers supports NeMo Gym resources-server tasks through `NeMoGymTaskset`. A
verifiers harness owns the model and agent loop, while Gym remains responsible for
per-rollout state, tools, and verification.

This is not native NeMo Gym agent execution. A custom Gym agent is replaced by the
selected verifiers harness, so use native Gym rollout collection when that agent is
part of the benchmark semantics.

## Weather example

Start the bundled example's resources server:

```bash
uv run verifiers/v1/tasksets/nemo_gym_weather/server.py
```

Then run its five example tasks with any MCP-capable verifiers harness:

```bash
uv run eval nemo-gym-weather --harness.id default --no-push -n 5
```

The launcher installs the published `nemo-gym==0.4.0` package and starts only the
`example_mcp_weather` resources server. It does not start Gym's agent, model, or head
server stack.

## Custom resources server

Start the resources server separately and point the generic taskset at its JSONL
dataset and URL:

```toml
model = "openai/gpt-5-mini"

[taskset]
id = "nemo-gym"
dataset_path = "/path/to/tasks.jsonl"

[taskset.task]
resources_url = "http://127.0.0.1:8000"

[harness]
id = "default"
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

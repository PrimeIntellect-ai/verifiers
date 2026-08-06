# nemo-gym-weather-v1

NeMo Gym's official MCP weather resource server as a minimal v1 example. The
resource-server lifecycle, per-rollout session bridge, and scoring live in
`verifiers`; this package only pins the server class and dataset.

## Develop

Run it from the Verifiers checkout with any MCP-capable harness:

```bash
uv run --with-editable . --with-editable environments/nemo_gym_weather_v1 \
  eval nemo-gym-weather-v1 -n 5 --env.agent.harness.id bash --no-push
```

## Layout

- `nemo_gym_weather_v1/taskset.py` — a thin config over the reusable
  `NeMoGymTaskset`.
- `nemo_gym_weather_v1/example.jsonl` — five weather tasks from NeMo Gym.

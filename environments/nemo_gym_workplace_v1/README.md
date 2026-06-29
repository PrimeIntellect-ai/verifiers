# nemo-gym-workplace-v1

NeMo Gym's `workplace_assistant` example through the built-in `NeMoGymTaskset`.
This environment pins the taskset config only; the standard Verifiers harness owns the
rollout loop, and NeMo Gym's packaged resource server owns tool execution.

## Develop

Install + run:

```bash
uv run --with-editable . --with-editable environments/nemo_gym_workplace_v1 \
  --with nemo-gym==0.3.0 eval nemo-gym-workplace-v1 -n 1 -r 1 -c 1
```

## Layout

- `nemo_gym_workplace_v1/taskset.py` — a thin config wrapper over `NeMoGymTaskset`.

Tune the packaged dataset from the CLI with `--taskset.data-name`, and use ordinary
Verifiers harness flags for rollout behavior.

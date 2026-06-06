# nemo-gym-env

### Overview
- **Environment ID**: `nemo-gym-env`
- **Short description**: Minimal v1 Verifiers environment that runs a NeMo Gym task through `NeMoGymTaskset` and `NeMoGymHarness`.
- **Tags**: nemo-gym, tool-use, v1, train, eval

### Datasets
- **Primary dataset(s)**: NeMo Gym `example_single_tool_call`.
- **Source links**: Packaged `resources_servers/example_single_tool_call/data/example.jsonl` from `nemo-gym`.
- **Split sizes**: 5 packaged examples; default eval uses 1 example via `pyproject.toml`.

### Task
- **Type**: `vf.Env` with `NeMoGymTaskset` from `tasksets` and `NeMoGymHarness` from `harnesses`.
- **Rubric overview**: Reward and metrics are returned by the NeMo Gym resources server.

### How it works
The taskset loads NeMo Gym JSONL rows from the installed `nemo-gym` package and stores each original row under `nemo_gym_row`. The harness starts the NeMo Gym stack once per env worker, exposes a stable local OpenAI-compatible model URL for NeMo Gym, and routes those model calls back through the matching Verifiers rollout endpoint.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run nemo-gym-env
```

When running directly from this repository before the NeMo Gym integration is released on PyPI, point Prime at the in-repo environments directory:

```bash
prime eval run nemo-gym-env --env-dir-path environments
```

Configure model and sampling:

```bash
prime eval run nemo-gym-env \
  -m gpt-4.1-mini \
  -n 1 -r 1 -t 128
```

### Configuration
NeMo Gym settings belong under the v1 taskset/harness config sections:

```toml
[env.taskset]
nemo_env = "example_single_tool_call"
limit = 1

[env.harness]
nemo_env = "example_single_tool_call"
timeout_seconds = 30
```

### Adapting
This example is intentionally tied to one NeMo Gym task. To create another Verifiers environment, copy this directory and change `NEMO_ENV` in `nemo_gym_env_v1/taskset.py` to another packaged NeMo Gym environment name, such as `example_multi_step`, `mcqa`, or `structured_outputs`.

### Metrics
| Metric | Meaning |
| --- | --- |
| `reward` | Reward returned by the NeMo Gym resources server. |

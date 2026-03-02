# nemo-workplace-assistant

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/nemo_workplace_assistant">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `nemo-workplace-assistant`
- **Short description**: Tool-rich NeMo Gym `workplace_assistant` resource server adapter environment, executed in sandbox per rollout.
- **Tags**: nemo-gym, tools, session-state, sandbox

### Why this example
This environment is a representative NeMo Gym integration pattern because it exercises:
- session seeding (`/seed_session`) before tool use
- many dynamic tools exposed per row from `responses_create_params.tools`
- final scoring through `/verify`

### Datasets
- **Primary dataset(s)**: `resources_servers/workplace_assistant/data/<split>.jsonl` from `nemo-gym`.
- **Source links**: [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)
- **Split sizes**: use `dataset_split` (`example`, `train`, `validation`) and optional `dataset_limit`.

### Task
- **Type**: Multi-turn tool use
- **Rubric overview**: reward is `verify_response.reward` returned by the NeMo server `/verify` endpoint.

### Quickstart
Install local environment package:

```bash
uv run vf-install nemo_workplace_assistant -p ./environments
```

Run an evaluation:

```bash
uv run vf-eval nemo-workplace-assistant -m anthropic/claude-sonnet-4.5 -n 1 -r 1
```

Override sandbox NeMo package (if needed):

```bash
uv run vf-eval nemo-workplace-assistant -m anthropic/claude-sonnet-4.5 -n 1 -r 1 \
  --env-args '{"nemo_package":"https://test-files.pythonhosted.org/packages/c0/58/451a826009a0b206c932e1ebde3dcff2a8b31152c77133fdde7e5f7ccd90/nemo_gym-0.2.9892rc0-py3-none-any.whl"}'
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_split` | str | `"example"` | Dataset split (`example`, `train`, `validation`) |
| `dataset_path` | str \| None | `None` | Optional explicit JSONL path override |
| `dataset_limit` | int \| None | `None` | Optional row cap |
| `max_turns` | int | `16` | Max turns per rollout |
| `nemo_package` | str | TestPyPI wheel URL | Package/wheel installed inside sandbox |
| `nemo_package_version` | str \| None | `None` | Optional version pin when using package name |

Any additional kwargs are forwarded to `verifiers.envs.integrations.nemo_gym_env.load_environment`.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | scalar reward from NeMo `/verify` response |
| `_verify_error_metric` | 1.0 if verify failed and adapter used fallback, else 0.0 |
| `total_tool_calls` | number of executed tool calls |
| `num_turns` | number of turns in rollout |

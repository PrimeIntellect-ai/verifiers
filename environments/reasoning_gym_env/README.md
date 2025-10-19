# reasoning-gym-env

### Overview
- **Environment ID**: `reasoning-gym-env`
- **Short description**: Single-turn evaluation over `reasoning_gym` procedural tasks with XML formatting.
- **Tags**: reasoning, procedural, single-turn, xml, synthetic

### Datasets
- **Primary dataset(s)**: Generated via `reasoning_gym` (e.g., `arc_1d`, or composite configs)
- **Source links**: `reasoning_gym` library
- **Split sizes**: Configurable counts for train/eval via loader args

### Task
- **Type**: single-turn
- **Parser**: `XMLParser(["answer"])`
- **Rubric overview**: Score computed via `reasoning_gym` task-specific scorer; optional format component

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval reasoning-gym-env
```

Configure model and sampling:

```bash
uv run vf-eval reasoning-gym-env \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"gym": "arc_1d", "num_train_examples": 2000, "num_eval_examples": 2000}'
```

Notes:
- Use `gym` to select a single dataset name, a list of names, or a composite specification.
- Reports are written under `./environments/reasoning_gym_env/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `gym` | str | `"arc_1d"` | Single task name, list of names, or composite config |
| `num_train_examples` | int | `2000` | Number of training examples |
| `num_eval_examples` | int | `2000` | Number of evaluation examples |
| `seed` | int | `0` | Random seed for dataset generation |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Task-specific score from `reasoning_gym` for parsed answer |
| `format_reward` | Adherence to `<think>`/`<answer>` XML format |

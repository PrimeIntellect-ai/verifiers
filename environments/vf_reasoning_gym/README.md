# vf-reasoning-gym

### Overview
- **Environment ID**: `vf-reasoning-gym`
- **Short description**: Single-turn evaluation over `reasoning_gym` procedural tasks with XML formatting.
- **Tags**: reasoning, procedural, single-turn, xml, synthetic

### Datasets
- **Primary dataset(s)**: Generated via `reasoning_gym` (e.g., `arc_1d`, or composite configs)
- **Source links**: `reasoning_gym` library
- **Split sizes**: Configurable counts for train/eval via loader args

### Task
- **Type**: single-turn
- **Parser**: `XMLParser(["think","answer"])`
- **Rubric overview**: Score computed via `reasoning_gym` task-specific scorer; optional format component

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-reasoning-gym
```

Configure model and sampling:

```bash
uv run vf-eval vf-reasoning-gym \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"gym": "arc_1d", "num_train_examples": 2000, "num_eval_examples": 2000}'
```

Notes:
- Use `gym` to select a single dataset name, a list of names, or a composite specification.
- Reports are written under `./environments/vf_reasoning_gym/reports/` and auto-embedded below.

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

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-reasoning-gym -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->

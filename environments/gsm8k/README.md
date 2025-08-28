# gsm8k

### Overview
- **Environment ID**: `gsm8k`
- **Short description**: Single-turn GSM8K math word problems with boxed numeric answers and CoT.
- **Tags**: math, gsm8k, single-turn, think, boxed-answer

### Datasets
- **Primary dataset(s)**: `gsm8k` train (train) and test (eval) via `load_example_dataset`
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via args; defaults to full train/test

### Task
- **Type**: single-turn
- **Parser**: `ThinkParser` with boxed answer extraction
- **Rubric overview**: Exact match on parsed boxed answer; optional format check

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval gsm8k
```

Configure model and sampling:

```bash
uv run vf-eval gsm8k \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": -1, "num_eval_examples": -1}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports are written under `./environments/gsm8k/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `-1` | Limit training set size (`-1` for all) |
| `num_eval_examples` | int | `-1` | Limit eval set size (`-1` for all) |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if parsed boxed answer equals target, else 0.0 |
| `format_reward` | Adherence to `<think>` + boxed `\boxed{...}` format |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval gsm8k -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->

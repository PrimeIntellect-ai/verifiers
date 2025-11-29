# doublecheck

### Overview
- **Environment ID**: `doublecheck`
- **Short description**: Two-turn math QA that asks the model to answer, then prompts “Are you sure?”; scored with a math rubric.
- **Tags**: math, multi-turn, xml, think-answer, verification

### Datasets
- **Primary dataset(s)**: `math` (example dataset loaded via `load_example_dataset`)
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via args; defaults to `train` split and all examples

### Task
- **Type**: multi-turn
- **Parser**: XMLParser with fields `think`, `answer` (from `MathRubric`)
- **Rubric overview**: `MathRubric` combining exact/equivalence math grading and a small format component

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval doublecheck
```

Configure model and sampling:

```bash
uv run vf-eval doublecheck \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"dataset_name": "math", "dataset_split": "train", "num_train_examples": -1}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"math"` | Example dataset name for math problems |
| `dataset_split` | str | `"train"` | Dataset split to load |
| `num_train_examples` | int | `-1` | Limit on dataset size (`-1` for all) |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Math answer correctness (symbolic/numeric equivalence) | 
# math-python-v1

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/math_python_v1">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `math-python-v1`
- **Short description**: v1 tool-using math environment with a task-owned Python `Toolset`; graded by symbolic equivalence.
- **Tags**: math, tools, python, single-turn, boxed-answer

### Datasets
- **Primary dataset(s)**: Example `math` dataset via `load_example_dataset`
- **Source links**: Uses example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via args; defaults to `train` split and all examples

### Task
- **Type**: `vf.Env` with a math `vf.Taskset`, base `vf.Harness`, and task-owned Python `Toolset`.
- **Rubric overview**: Correctness by `math_verify.parse` + `verify` over the final boxed answer.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run math-python-v1
```

Configure model and sampling:

```bash
prime eval run math-python-v1 \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"config": {"taskset": {"dataset_name": "math", "dataset_split": "train", "num_train_examples": -1}}}'
```

Notes:
- v1 task settings belong under `config.taskset` when passed through `-a` / `--env-args`.

### Taskset Config
| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"math"` | Example dataset to load |
| `dataset_split` | str | `"train"` | Split to load |
| `num_train_examples` | int | `-1` | Limit dataset size (`-1` for all) |
| `pip_install_packages` | str | `"numpy sympy scipy"` | Packages listed in the generated system prompt |

### Harness Config
| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `100` | Maximum model turns per rollout |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if symbolic verification passes, else 0.0 |
| `num_turns` | Number of recorded model turns |

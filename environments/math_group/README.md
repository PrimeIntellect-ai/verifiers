# math-group

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/gsm8k">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `math-group`
- **Short description**: Groups GSM8K and MATH single-turn sub-environments into one evaluation with a shared math CoT parser.
- **Tags**: math, group, single-turn, gsm8k, math, think

### Datasets
- **Primary dataset(s)**: `gsm8k` (subset of 1000 train examples) and `math` (subset of 1000 train examples)
- **Source links**: Uses example loader in `verifiers.utils.data_utils`
- **Split sizes**: 1000 per sub-environment (train-only in this group)

### Task
- **Type**: single-turn (EnvGroup of two SingleTurnEnv instances)
- **Parser**: `ThinkParser` with boxed answer extraction
- **Rubric overview**: Exact numeric/equivalent match per sub-environment plus optional format check

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval math-group
```

Configure model and sampling:

```bash
uv run vf-eval math-group \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- This environment bundles two math datasets; reporting aggregates across both.
- Reports are written under `./environments/math_group/reports/` and auto-embedded below.

### Environment Arguments
This loader does not expose custom arguments.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if parsed boxed answer equals target (per sub-env), else 0.0 |
| `format_reward` | Adherence to `<think>` + boxed `\boxed{...}` format |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-math-group -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->

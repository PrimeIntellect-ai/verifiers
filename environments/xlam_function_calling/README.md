# xlam-function-calling

### Overview
- **Environment ID**: `xlam-function-calling`
- **Short description**: Function-calling reproduction from XLAM-60k: model must emit a JSON array of tool calls.

### Datasets
- **Primary dataset(s)**: `Salesforce/xlam-function-calling-60k` (train split)
- **Source links**: Hugging Face Datasets
- **Split sizes**: Uses the `train` split for training/evaluation

### Task
- **Type**: single-turn
- **Parser**: `XMLParser(["think","tool"], answer_field="tool")`
- **Rubric overview**: Exact set equality between parsed JSON array and target tool list

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval xlam-function-calling
```

Configure model and sampling:

```bash
uv run vf-eval xlam-function-calling \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

### Environment Arguments
This loader does not expose custom arguments.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if parsed tool JSON exactly matches target, else 0.0 |


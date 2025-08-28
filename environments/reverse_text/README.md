# reverse-text

### Overview
- **Environment ID**: `reverse-text`
- **Short description**: Reverse a given paragraph; evaluated by LCS similarity to the exact reversal.
- **Tags**: text, transformation, single-turn, xml

### Datasets
- **Primary dataset(s)**: `agentlans/wikipedia-paragraphs` mapped to question/answer pairs
- **Source links**: Hugging Face Datasets
- **Split sizes**: Train/eval split controlled by `num_train_examples` and `num_eval_examples`

### Task
- **Type**: single-turn
- **Parser**: `XMLParser(["think","answer"])`
- **Rubric overview**: LCS similarity between parsed answer and ground-truth reversed text; optional format check

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval reverse-text
```

Configure model and sampling:

```bash
uv run vf-eval reverse-text \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 2000, "num_eval_examples": 200}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `2000` | Number of training examples |
| `num_eval_examples` | int | `200` | Number of evaluation examples |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | LCS similarity between reversed text and parsed answer |
| `format_reward` | Adherence to `<think>`/`<answer>` XML format |

# tool-test

### Overview
- **Environment ID**: `tool-test`
- **Short description**: Sanity-check tool-calling environment that asks models to invoke a random subset of dummy tools.
- **Tags**: tools, single-turn, function-calling, sanity

### Datasets
- **Primary dataset(s)**: Synthetic HF dataset generated in-memory with prompts specifying required tools
- **Source links**: N/A (programmatically generated)
- **Split sizes**: Controlled by `num_train_examples` and `num_eval_examples`

### Task
- **Type**: tool use (single-turn ToolEnv)
- **Parser**: default `Parser`
- **Rubric overview**: ToolRubric checks tool execution and adds exact match on the required tool set

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval tool-test
```

Configure model and sampling:

```bash
uv run vf-eval tool-test \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 1000, "num_eval_examples": 100}'
```


### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `1000` | Number of training examples |
| `num_eval_examples` | int | `100` | Number of evaluation examples |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if called tool set equals required set, else 0.0 |
| ToolRubric metrics | Tool execution success and format adherence |

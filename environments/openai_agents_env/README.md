# openai-agents-env

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/openai_agents_env">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `openai-agents-env`
- **Short description**: ApiEnv example using the OpenAI Agents SDK with a calculator tool on GSM8K math problems.
- **Tags**: api-env, agents, tool-use, math, gsm8k

### Datasets
- **Primary dataset(s)**: `gsm8k` train (train) and test (eval) via `load_example_dataset`
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via args; defaults to 50 train / 20 eval

### Task
- **Type**: multi-turn (ApiEnv with interception proxy)
- **Rubric overview**: Exact match on numeric answer extracted from `ANSWER: <value>` pattern

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run openai-agents-env
```

Configure model and sampling:

```bash
prime eval run openai-agents-env \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 50, "num_eval_examples": 20}'
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `50` | Number of training examples |
| `num_eval_examples` | int | `20` | Number of evaluation examples |
| `timeout_seconds` | float | `120.0` | Per-rollout timeout in seconds |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if agent's ANSWER matches target, else 0.0 |

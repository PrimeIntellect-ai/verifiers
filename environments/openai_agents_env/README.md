# openai-agents-env

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/openai_agents_env">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `openai-agents-env`
- **Short description**: V1 Taskset/Harness example using the OpenAI Agents SDK with a calculator tool on GSM8K math problems.
- **Tags**: v1, taskset, harness, agents, tool-use, math, gsm8k

### Datasets
- **Primary dataset(s)**: `gsm8k` train (train) and test (eval) via `load_example_dataset`
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via taskset config; defaults to 50 train / 20 eval

### Task
- **Type**: `vf.Env` with a GSM8K `vf.Taskset` and OpenAI Agents SDK `vf.Harness`
- **Rubric overview**: Exact match on numeric answer extracted from `ANSWER: <value>` pattern

### How it works
The taskset owns GSM8K source/eval rows and reward logic. The harness runs an in-process OpenAI Agents SDK program, builds its client from `state.get_endpoint_config(api="chat")`, and routes every model call through the V1 interception endpoint.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run openai-agents-env
```

Configure model and sampling:

```bash
prime eval run openai-agents-env \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

### Taskset Config
| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `50` | Number of training examples |
| `num_eval_examples` | int | `20` | Number of evaluation examples |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if agent's ANSWER matches target, else 0.0 |

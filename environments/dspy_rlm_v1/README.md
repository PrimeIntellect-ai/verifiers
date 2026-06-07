# dspy-rlm-v1

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/dspy_rlm_v1">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `dspy-rlm-v1`
- **Short description**: V1 Taskset/Harness example using DSPy's RLM (Recursive Language Model) module on GSM8K math problems.
- **Tags**: v1, taskset, harness, dspy, rlm, math, gsm8k

### Datasets

- **Primary dataset(s)**: `gsm8k` train (train) and test (eval) via `load_example_dataset`
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via taskset config; defaults to 50 train / 20 eval

### Task

- **Type**: `vf.Env` with a GSM8K `vf.Taskset` and DSPy RLM `vf.Harness`
- **Rubric overview**: Exact numeric match on answer extracted from DSPy structured output

### How it works

The taskset owns GSM8K train/eval task loading and reward logic. The harness runs an in-process DSPy RLM agent, starts the v1 protocol endpoint for the rollout, and gives DSPy an OpenAI-compatible LM pointed at that endpoint.

DSPy RLM requires Deno to be available in the runtime environment.

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run dspy-rlm-v1
```

Configure model and sampling:

```bash
prime eval run dspy-rlm-v1 \
  -m gpt-4.1-mini \
  -n 10 -r 3 -t 1024 -T 0.7
```

### Taskset Config

| Field                | Type  | Default | Description                    |
| -------------------- | ----- | ------- | ------------------------------ |
| `num_train_examples` | int   | `50`    | Number of training examples    |
| `num_eval_examples`  | int   | `20`    | Number of evaluation examples  |

### Metrics

| Metric   | Meaning                                                    |
| -------- | ---------------------------------------------------------- |
| `reward` | 1.0 if agent's answer matches target numerically, else 0.0 |

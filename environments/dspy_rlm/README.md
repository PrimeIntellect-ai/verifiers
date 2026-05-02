# dspy-rlm

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/dspy_rlm">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `dspy-rlm`
- **Short description**: ApiEnv example using DSPy's RLM (Recursive Language Model) module on GSM8K math problems.
- **Tags**: api-env, dspy, rlm, math, gsm8k

### Datasets

- **Primary dataset(s)**: `gsm8k` train (train) and test (eval) via `load_example_dataset`
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via args; defaults to 50 train / 20 eval

### Task

- **Type**: multi-turn (ApiEnv with interception proxy)
- **Rubric overview**: Exact numeric match on answer extracted from DSPy structured output

### Prerequisites

RLM requires [Deno](https://deno.land/) for its WASM sandbox:

```bash
curl -fsSL https://deno.land/install.sh | sh
```

[Additional installation options can be found here.](https://docs.deno.com/runtime/getting_started/installation/)

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run dspy-rlm
```

Configure model and sampling:

```bash
prime eval run dspy-rlm \
  -m gpt-4.1-mini \
  -n 10 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 50, "num_eval_examples": 20}'
```

### Environment Arguments

| Arg                  | Type  | Default | Description                    |
| -------------------- | ----- | ------- | ------------------------------ |
| `num_train_examples` | int   | `50`    | Number of training examples    |
| `num_eval_examples`  | int   | `20`    | Number of evaluation examples  |
| `timeout_seconds`    | float | `180.0` | Per-rollout timeout in seconds |

### Metrics

| Metric   | Meaning                                                    |
| -------- | ---------------------------------------------------------- |
| `reward` | 1.0 if agent's answer matches target numerically, else 0.0 |

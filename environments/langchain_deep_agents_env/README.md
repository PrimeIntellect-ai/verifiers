# langchain-deep-agents-env

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/langchain_deep_agents_env">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `langchain-deep-agents-env`
- **Short description**: V1 Taskset/Harness example using the [LangChain Deep Agents SDK](https://docs.langchain.com/oss/python/deepagents/overview) (`deepagents.create_deep_agent`) with a calculator tool on GSM8K math problems.
- **Tags**: v1, taskset, harness, tool-use, langchain, deep-agents

### Datasets

- **Primary dataset(s)**: `gsm8k` train (train) and test (eval) via `load_example_dataset`
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via args; defaults to 50 train / 20 eval

### Task

- **Type**: `vf.Env` with a GSM8K `vf.Taskset` and LangChain Deep Agents `vf.Harness`
- **Rubric overview**: Exact numeric match on the value extracted from `ANSWER: <value>` in the agent's final message

### How it works

The taskset owns GSM8K source/eval rows and reward logic. The harness runs an in-process LangChain Deep Agents program, builds its chat model from `state.get_endpoint_config(api="chat")`, and routes every model call through the V1 interception endpoint.

`deepagents.create_deep_agent` returns a compiled LangGraph agent equipped with planning (`write_todos`), filesystem tools, summarization middleware, and a built-in `general-purpose` subagent. A small `calculate` tool is provided for arithmetic.

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run langchain-deep-agents-env
```

Configure model and sampling:

```bash
prime eval run langchain-deep-agents-env \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 50, "num_eval_examples": 20}'
```

### Environment Arguments

| Arg                  | Type  | Default | Description                    |
| -------------------- | ----- | ------- | ------------------------------ |
| `num_train_examples` | int   | `50`    | Number of training examples    |
| `num_eval_examples`  | int   | `20`    | Number of evaluation examples  |

### Metrics

| Metric   | Meaning                                                    |
| -------- | ---------------------------------------------------------- |
| `reward` | 1.0 if agent's ANSWER matches target numerically, else 0.0 |

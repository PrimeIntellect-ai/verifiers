# Evaluation

This section explains how to run evaluations with Verifiers environments. See [Environments](environments.md) for information on building your own environments.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Command Reference](#command-reference)
  - [Environment Selection](#environment-selection)
  - [Model Configuration](#model-configuration)
  - [Sampling Parameters](#sampling-parameters)
  - [Evaluation Scope](#evaluation-scope)
  - [Concurrency](#concurrency)
  - [Output and Saving](#output-and-saving)
- [Environment Defaults](#environment-defaults)

Use `vf-eval` to execute rollouts against any OpenAI-compatible model and report aggregate metrics.

## Basic Usage

Environments must be installed as Python packages before evaluation. From a local environment:

```bash
uv run vf-install my-env           # installs ./environments/my_env as a package
uv run vf-eval my-env -m gpt-4.1-mini -n 10
```

`vf-eval` imports the environment module using Python's import system, calls its `load_environment()` function, runs 10 examples with 3 rollouts each (the default), scores them using the environment's rubric, and prints aggregate metrics.

## Command Reference

### Environment Selection

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `env_id` | (positional) | — | Environment module name (e.g., `my-env` or `gsm8k`) |
| `--env-args` | `-a` | `{}` | JSON object passed to `load_environment()` |
| `--extra-env-kwargs` | `-x` | `{}` | JSON object passed to environment constructor |
| `--env-dir-path` | `-p` | `./environments` | Base path for saving output files |

The `env_id` is converted to a Python module name (`my-env` → `my_env`) and imported. The module must be installed (via `vf-install` or `uv pip install`).

The `--env-args` flag passes arguments to your `load_environment()` function:

```bash
vf-eval my-env -a '{"difficulty": "hard", "num_examples": 100}'
```

The `--extra-env-kwargs` flag passes arguments directly to the environment constructor, useful for overriding defaults like `max_turns` which may not be exposed via `load_environment()`:

```bash
vf-eval my-env -x '{"max_turns": 20}'
```

### Model Configuration

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | `gpt-4.1-mini` | Model name or endpoint alias |
| `--api-base-url` | `-b` | `https://api.openai.com/v1` | API base URL |
| `--api-key-var` | `-k` | `OPENAI_API_KEY` | Environment variable containing API key |
| `--endpoints-path` | `-e` | `./configs/endpoints.py` | Path to endpoints registry |
| `--header` | — | — | Extra HTTP header (`Name: Value`), repeatable |

For convenience, define model endpoints in `./configs/endpoints.py` to avoid repeating URL and key flags:

```python
ENDPOINTS = {
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "qwen3-235b-i": {
        "model": "qwen/qwen3-235b-a22b-instruct-2507",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
}
```

Then use the alias directly:

```bash
vf-eval my-env -m qwen3-235b-i
```

If the model name isn't found in the registry, the `--api-base-url` and `--api-key-var` flags are used instead.

### Sampling Parameters

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--max-tokens` | `-t` | model default | Maximum tokens to generate |
| `--temperature` | `-T` | model default | Sampling temperature |
| `--sampling-args` | `-S` | — | JSON object for additional sampling parameters |

The `--sampling-args` flag accepts any parameters supported by the model's API:

```bash
vf-eval my-env -S '{"temperature": 0.7, "top_p": 0.9}'
```

### Evaluation Scope

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--num-examples` | `-n` | 5 | Number of dataset examples to evaluate |
| `--rollouts-per-example` | `-r` | 3 | Rollouts per example (for pass@k, variance) |

Multiple rollouts per example enable metrics like pass@k and help measure variance. The total number of rollouts is `num_examples × rollouts_per_example`.

### Concurrency

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--max-concurrent` | `-c` | 32 | Maximum concurrent requests |
| `--max-concurrent-generation` | — | same as `-c` | Concurrent generation requests |
| `--max-concurrent-scoring` | — | same as `-c` | Concurrent scoring requests |
| `--no-interleave-scoring` | `-N` | false | Disable interleaved scoring |

By default, scoring runs interleaved with generation. Use `--no-interleave-scoring` to score all rollouts after generation completes.

### Output and Saving

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--verbose` | `-v` | false | Enable debug logging |
| `--save-results` | `-s` | false | Save results to disk |
| `--save-every` | `-f` | -1 | Save checkpoint every N rollouts |
| `--state-columns` | `-C` | — | Extra state columns to save (comma-separated) |
| `--save-to-hf-hub` | `-H` | false | Push results to Hugging Face Hub |
| `--hf-hub-dataset-name` | `-D` | — | Dataset name for HF Hub |

Results are saved to `./outputs/evals/{env_id}--{model}/` as a Hugging Face dataset.

The `--state-columns` flag allows saving environment-specific state fields that your environment stores during rollouts:

```bash
vf-eval my-env -s -C "judge_response,parsed_answer"
```

## Environment Defaults

Environments can specify default evaluation parameters in their `pyproject.toml` (See [Developing Environments](environments.md#developing-environments)):

```toml
[tool.verifiers.eval]
num_examples = 100
rollouts_per_example = 5
```

These defaults are used when flags aren't explicitly provided. Priority order: CLI flags → environment defaults → global defaults.


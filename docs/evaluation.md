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
- [Multi-Environment Evaluation](#multi-environment-evaluation)
  - [TOML Configuration](#toml-configuration)
  - [Configuration Precedence](#configuration-precedence)

Use `prime eval` to execute rollouts against any OpenAI-compatible model and report aggregate metrics.

## Basic Usage

Environments must be installed as Python packages before evaluation. From a local environment:

```bash
prime env install my-env           # installs ./environments/my_env as a package
prime eval run my-env -m openai/gpt-4.1-mini -n 10
```

`prime eval` imports the environment module using Python's import system, calls its `load_environment()` function, runs 5 examples with 3 rollouts each (the default), scores them using the environment's rubric, and prints aggregate metrics.

## Command Reference

### Environment Selection

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `env_id_or_path` | (positional) | — | Environment ID(s) or path to TOML config |
| `--env-args` | `-a` | `{}` | JSON object passed to `load_environment()` |
| `--extra-env-kwargs` | `-x` | `{}` | JSON object passed to environment constructor |
| `--env-dir-path` | `-p` | `./environments` | Base path for saving output files |

The positional argument accepts two formats:
- **Single environment**: `gsm8k` — evaluates one environment
- **TOML config path**: `configs/eval/benchmark.toml` — evaluates multiple environments defined in the config file

Environment IDs are converted to Python module names (`my-env` → `my_env`) and imported. Modules must be installed (via `prime env install` or `uv pip install`).

The `--env-args` flag passes arguments to your `load_environment()` function:

```bash
prime eval run my-env -a '{"difficulty": "hard", "num_examples": 100}'
```

The `--extra-env-kwargs` flag passes arguments directly to the environment constructor, useful for overriding defaults like `max_turns` which may not be exposed via `load_environment()`:

```bash
prime eval run my-env -x '{"max_turns": 20}'
```

### Model Configuration

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | `openai/gpt-4.1-mini` | Model name or endpoint alias |
| `--api-base-url` | `-b` | `https://api.pinference.ai/api/v1` | API base URL |
| `--api-key-var` | `-k` | `PRIME_API_KEY` | Environment variable containing API key |
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
prime eval run my-env -m qwen3-235b-i
```

If the model name is in the registry, those values are used by default, but you can override them with `--api-base-url` and/or `--api-key-var`. If the model name isn't found, the CLI flags are used (falling back to defaults when omitted).

### Sampling Parameters

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--max-tokens` | `-t` | model default | Maximum tokens to generate |
| `--temperature` | `-T` | model default | Sampling temperature |
| `--sampling-args` | `-S` | — | JSON object for additional sampling parameters |

The `--sampling-args` flag accepts any parameters supported by the model's API:

```bash
prime eval run my-env -S '{"temperature": 0.7, "top_p": 0.9}'
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
| `--max-retries` | — | 0 | Retries per rollout on transient `InfraError` |

By default, scoring runs interleaved with generation. Use `--no-interleave-scoring` to score all rollouts after generation completes.

The `--max-retries` flag enables automatic retry with exponential backoff when rollouts fail due to transient infrastructure errors (e.g., sandbox timeouts, API failures).

### Output and Saving

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--verbose` | `-v` | false | Enable debug logging |
| `--tui` | `-u` | false | Use alternate screen mode (TUI) for display |
| `--debug` | `-d` | false | Disable Rich display; use normal logging and tqdm progress |
| `--save-results` | `-s` | false | Save results to disk |
| `--save-every` | `-f` | -1 | Save checkpoint every N rollouts |
| `--state-columns` | `-C` | — | Extra state columns to save (comma-separated) |
| `--save-to-hf-hub` | `-H` | false | Push results to Hugging Face Hub |
| `--hf-hub-dataset-name` | `-D` | — | Dataset name for HF Hub |

Results are saved to `./outputs/evals/{env_id}--{model}/` as a Hugging Face dataset.

The `--state-columns` flag allows saving environment-specific state fields that your environment stores during rollouts:

```bash
prime eval run my-env -s -C "judge_response,parsed_answer"
```

## Environment Defaults

Environments can specify default evaluation parameters in their `pyproject.toml` (See [Developing Environments](environments.md#developing-environments)):

```toml
[tool.verifiers.eval]
num_examples = 100
rollouts_per_example = 5
```

These defaults are used when higher-priority sources don't specify a value. The full priority order is:

1. TOML per-environment settings (when using a config file)
2. CLI flags
3. Environment defaults (from `pyproject.toml`)
4. Global defaults

See [Configuration Precedence](#configuration-precedence) for more details on multi-environment evaluation.

## Multi-Environment Evaluation

You can evaluate multiple environments using `prime eval` with a TOML configuration file. This is useful for running comprehensive benchmark suites.

### TOML Configuration

For multi-environment evals or fine-grained control over settings, use a TOML configuration file. When using a config file, CLI arguments are ignored.

```bash
prime eval run configs/eval/my-benchmark.toml
```

The TOML file uses `[[eval]]` sections to define each evaluation. You can also specify global defaults at the top:

```toml
# configs/eval/my-benchmark.toml

# Global defaults (optional)
model = "openai/gpt-4.1-mini"
num_examples = 50

[[eval]]
env_id = "gsm8k"
num_examples = 100  # overrides global default
rollouts_per_example = 5

[[eval]]
env_id = "alphabet-sort"
# Uses global num_examples (50)
rollouts_per_example = 3

[[eval]]
env_id = "math-python"
# Uses global defaults and built-in defaults for unspecified values
```

A minimal config requires only a single `[[eval]]` section:

```toml
[[eval]]
env_id = "gsm8k"
```

Each `[[eval]]` section must contain an `env_id` field. All other fields are optional:

| Field | Type | Description |
|-------|------|-------------|
| `env_id` | string | **Required.** Environment module name |
| `env_args` | table | Arguments passed to `load_environment()` |
| `num_examples` | integer | Number of dataset examples to evaluate |
| `rollouts_per_example` | integer | Rollouts per example |
| `extra_env_kwargs` | table | Arguments passed to environment constructor |
| `model` | string | Model to evaluate |

Example with `env_args`:

```toml
[[eval]]
env_id = "math-python"
num_examples = 50

[eval.env_args]
difficulty = "hard"
split = "test"
```

### Configuration Precedence

When using a **config file**, CLI arguments are ignored. Settings are resolved as:

1. **TOML per-eval settings** — Values specified in `[[eval]]` sections
2. **TOML global settings** — Values at the top of the config file
3. **Environment defaults** — Values from the environment's `pyproject.toml`
4. **Built-in defaults** — (`num_examples=5`, `rollouts_per_example=3`)

When using **CLI only** (no config file), settings are resolved as:

1. **CLI arguments** — Flags passed on the command line
2. **Environment defaults** — Values from the environment's `pyproject.toml`
3. **Built-in defaults** — (`num_examples=5`, `rollouts_per_example=3`)

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
  - [Comma-Separated Environments](#comma-separated-environments)
  - [TOML Configuration](#toml-configuration)
  - [Configuration Precedence](#configuration-precedence)

Use `prime eval` to execute rollouts against any OpenAI-compatible model and report aggregate metrics.

## Basic Usage

Environments must be installed as Python packages before evaluation. From a local environment:

```bash
prime env install my-env           # installs ./environments/my_env as a package
prime eval run my-env -m gpt-4.1-mini -n 10
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

The positional argument accepts three formats:
- **Single environment**: `gsm8k` — evaluates one environment
- **Comma-separated list**: `gsm8k,alphabet-sort` — evaluates multiple environments in parallel
- **TOML config path**: `configs/eval/benchmark.toml` — evaluates environments defined in the config file

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

You can evaluate multiple environments in parallel using `prime eval`. This is useful for running comprehensive benchmark suites.

### Comma-Separated Environments

The simplest way to evaluate multiple environments is to provide a comma-separated list of environment IDs:

```bash
prime eval run gsm8k,alphabet-sort -n 5 -r 3
```

All environments share the same CLI arguments. This approach works well when you want to evaluate the default settings for all environments or only change a few settings for all environments.

```bash
# Evaluate three environments with shared settings
prime eval run gsm8k,math-python,wordle -m qwen3-235b-i -n 10 -r 5 -c 64
```

### TOML Configuration

For fine-grained control over per-environment and per-model settings, use a TOML configuration file. This allows you to specify different parameters for each environment or model while still running them in parallel.

```bash
prime eval run configs/eval/my-benchmark.toml -m gpt-4.1-mini
```

The TOML file uses `[env]`/`[model]` defaults with one or more `[[eval]]` entries:

```toml
# configs/eval/my-benchmark.toml

[env]
num_examples = 100
rollouts_per_example = 5

[model]
model = "openai/gpt-4.1-mini"

[[eval]]
[eval.env]
env_id = "gsm8k"

[[eval]]
[eval.env]
env_id = "alphabet-sort"
num_examples = 50

[[eval]]
[eval.env]
env_id = "math-python"
```

Eval TOML files define an evaluation run with defaults plus one or more `[[eval]]` entries. Defaults apply to any eval that does not override them.

| Table | Field | Type | Description |
|-------|-------|------|-------------|
| `[env]` or `[eval.env]` | `env_id` | string | **Required in `[eval.env]`.** Environment module name |
| `[env]` or `[eval.env]` | `env_args` | table | Arguments passed to `load_environment()` |
| `[env]` or `[eval.env]` | `num_examples` | integer | Number of dataset examples to evaluate |
| `[env]` or `[eval.env]` | `rollouts_per_example` | integer | Rollouts per example |
| `[env]` or `[eval.env]` | `interleave_scoring` | boolean | Whether to interleave scoring with generation |
| `[env]` or `[eval.env]` | `state_columns` | array | State columns to save |
| `[env]` or `[eval.env]` | `extra_env_kwargs` | table | Arguments passed to environment constructor |
| `[model]` or `[eval.model]` | `model` | string | Model name or endpoint alias |
| `[model]` or `[eval.model]` | `sampling_args` | table | Sampling arguments |
| `[model]` or `[eval.model]` | `max_concurrent` | integer | Max concurrent requests |
| `[model]` or `[eval.model]` | `max_concurrent_generation` | integer | Max concurrent generation requests |
| `[model]` or `[eval.model]` | `max_concurrent_scoring` | integer | Max concurrent scoring requests |
| `[model]` or `[eval.model]` | `api_key_var` | string | API key env var name override |
| `[model]` or `[eval.model]` | `api_base_url` | string | API base URL override |
| `[model]` or `[eval.model]` | `extra_headers` | table | Extra HTTP headers |
| `[save]` or `[eval.save]` | `save_results` | boolean | Save results to disk |
| `[save]` or `[eval.save]` | `save_every` | integer | Save every N rollouts |
| `[save]` or `[eval.save]` | `save_to_hf_hub` | boolean | Save dataset to Hugging Face Hub |
| `[save]` or `[eval.save]` | `hf_hub_dataset_name` | string | Hugging Face dataset name override |

Example with defaults and per-eval overrides:

```toml
# configs/eval/my-benchmark.toml

[env]
num_examples = 50
rollouts_per_example = 3

[model]
model = "openai/gpt-4.1-mini"
max_concurrent = 64

[save]
save_results = true

[[eval]]
[eval.env]
env_id = "math-python"
num_examples = 100

[eval.env.env_args]
difficulty = "hard"
split = "test"

[[eval]]
[eval.env]
env_id = "gsm8k"

[eval.model]
model = "qwen3-235b-i"

[eval.save]
hf_hub_dataset_name = "gsm8k-qwen3-235b-i"
```

### Configuration Precedence

Settings are resolved with the following priority order (highest to lowest) for per-environment fields:

1. **Per-eval TOML env settings** — Values in `[eval.env]`
2. **TOML env defaults** — Values in `[env]`
3. **CLI arguments** — Flags passed on the command line
4. **Environment defaults** — Values from the environment's `pyproject.toml`
5. **Global defaults** — Built-in defaults (`num_examples=5`, `rollouts_per_example=3`)

This means per-eval TOML settings take precedence over CLI arguments for the environments where they're specified, while environments without per-eval settings fall back to the shared defaults and CLI arguments:

```toml
[env]
num_examples = 10

[[eval]]
[eval.env]
env_id = "gsm8k"
num_examples = 100  # This env uses 100 examples

[[eval]]
[eval.env]
env_id = "alphabet-sort"
# No num_examples specified — uses defaults
```

```bash
# gsm8k uses 100 examples (from TOML), alphabet-sort uses 10 (from defaults)
prime eval run configs/eval/mixed.toml
```

Model settings resolve with the following priority (highest to lowest):

1. **Per-eval TOML model settings** — Values in `[eval.model]`
2. **TOML model defaults** — Values in `[model]`
3. **CLI arguments** — Flags passed on the command line

Save settings resolve with the following priority (highest to lowest):

1. **Per-eval TOML save settings** — Values in `[eval.save]`
2. **TOML save defaults** — Values in `[save]`
3. **CLI arguments** — Flags passed on the command line

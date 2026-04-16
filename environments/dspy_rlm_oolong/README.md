# dspy-rlm-oolong

### Overview

- **Environment ID**: `dspy-rlm-oolong`
- **Short description**: ApiEnv using DSPy's RLM on Oolong long-context benchmark tasks.
- **Tags**: api-env, dspy, rlm, long-context, oolong

### Datasets

- **Primary dataset(s)**: `oolongbench/oolong-synth` and `oolongbench/oolong-real` from Hugging Face
- **Subsets**: `synth` (default), `synth_with_labels`, `real` (dnd/toy_dnd)
- **Synth dataset names**: agnews, app_reviews, formality, imdb, metaphors, multinli, negation, spam, trec_coarse, yahoo
- **Context lengths** (synth only): 1K to 4M tokens

### Task

- **Type**: multi-turn (ApiEnv with interception proxy)
- **Rubric overview**: Two modes:
  - `oolong` (default): Deterministic scoring ported from the official OOLONG eval (partial credit for numeric, date parsing, list overlap)
  - `judge`: Binary LLM judge scoring (1.0/0.0)

### Prerequisites

RLM requires [Deno](https://deno.land/) for its WASM sandbox:

```bash
curl -fsSL https://deno.land/install.sh | sh
```

### Quickstart

Run a basic evaluation (synth subset, validation split):

```bash
prime eval run dspy-rlm-oolong -m gpt-4.1-mini -n 5
```

Specific dataset and context length:

```bash
prime eval run dspy-rlm-oolong \
  -m gpt-4.1-mini -n 5 \
  -a '{"subset": "synth", "dataset_name": "trec_coarse", "context_len": 131072}'
```

Real-world subset (DnD):

```bash
prime eval run dspy-rlm-oolong \
  -m gpt-4.1-mini -n 5 \
  -a '{"subset": "real", "dataset_name": "toy_dnd"}'
```

Multiple datasets/lengths:

```bash
prime eval run dspy-rlm-oolong \
  -m gpt-4.1-mini -n 5 \
  -a '{"subset": "synth", "dataset_name": ["spam", "trec_coarse"], "context_len": [131072, 262144]}'
```

With LLM judge scoring:

```bash
prime eval run dspy-rlm-oolong \
  -m gpt-4.1-mini -n 5 \
  -a '{"reward_mode": "judge", "judge_model": "gpt-4.1-nano"}'
```

### Environment Arguments

| Arg                    | Type          | Default       | Description                                           |
| ---------------------- | ------------- | ------------- | ----------------------------------------------------- |
| `subset`               | str           | `"synth"`     | `"synth"`, `"synth_with_labels"`, or `"real"`         |
| `split`                | str           | `"validation"`| `"validation"` or `"test"`                            |
| `dataset_name`         | str/list[str] | `None`        | Filter by dataset name(s)                             |
| `context_len`          | int/list[int] | `None`        | Filter by context length(s) (synth only)              |
| `shuffle`              | bool          | `False`       | Shuffle the dataset                                   |
| `seed`                 | int           | `None`        | Random seed for shuffling                             |
| `include_env_tips`     | bool          | `False`       | Include strategy tips in the prompt                   |
| `reward_mode`          | str           | `"oolong"`    | `"oolong"` (deterministic) or `"judge"` (LLM judge)  |
| `judge_model`          | str           | `"gpt-4.1-nano"` | Judge model (when reward_mode="judge")            |
| `judge_api_key_var`    | str           | `"OPENAI_API_KEY"` | Env var for judge API key                       |
| `judge_base_url`       | str           | `None`        | Custom base URL for judge API                         |
| `max_rlm_iterations`   | int           | `10`          | Max DSPy RLM iterations                               |
| `timeout_seconds`      | float         | `300.0`       | Per-rollout timeout in seconds                        |

### Metrics

| Metric          | Meaning                                                              |
| --------------- | -------------------------------------------------------------------- |
| `oolong_reward` | Deterministic score (partial credit for numeric, date, list overlap) |
| `judge_reward`  | Binary 1.0/0.0 from LLM judge (when reward_mode="judge")            |

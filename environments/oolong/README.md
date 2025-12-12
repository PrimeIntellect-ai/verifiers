# Oolong

A long-context evaluation environment for the Oolong benchmark.

## Description

This environment implements the [Oolong benchmark](https://huggingface.co/oolongbench) for evaluating long-context understanding capabilities of language models. The benchmark tests a model's ability to reason over and extract information from extended contexts.

Supports two modes:

- **RLM mode** (`use_rlm=True`, default): Uses RLMEnv with context available via Python code. The model can write code to explore the large context efficiently.
- **Standard mode** (`use_rlm=False`): Uses SingleTurnEnv with context directly in the prompt. The model must process the text directly.

## Datasets

Oolong consists of two HuggingFace datasets:

- [oolongbench/oolong-synth](https://huggingface.co/datasets/oolongbench/oolong-synth) - Synthetic long-context evaluation tasks
- [oolongbench/oolong-real](https://huggingface.co/datasets/oolongbench/oolong-real) - Real-world long-context evaluation tasks

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subset` | Literal["synth", "synth_with_labels", "real"] | "synth" | Which dataset subset to use |
| `split` | Literal["validation", "test"] | "validation" | Dataset split to use |
| `shuffle` | bool | False | Whether to shuffle the dataset |
| `use_rlm` | bool | True | If True, use RLM mode. If False, use standard SingleTurnEnv mode. |
| `include_env_tips` | bool | False | Include strategy tips in prompt for SFT data generation (RLM mode only) |
| `max_iterations` | int | 30 | Maximum REPL iterations (RLM mode only) |
| `max_output_length` | int | 8192 | Maximum code execution output length (RLM mode only) |
| `judge_model` | str | "gpt-5-mini" | Model to use for judging answer correctness |
| `judge_api_key_var` | str | "OPENAI_API_KEY" | Environment variable for judge API key |
| `metrics_output_path` | str \| None | None | Path to JSON file for logging per-rollout metrics |

### Subset Options

- **`synth`**: Uses `context_window_text` column from oolong-synth
- **`synth_with_labels`**: Uses `context_window_text_with_labels` column from oolong-synth
- **`real`**: Uses `context_window_text` column from oolong-real

### Mode Comparison

| Aspect | RLM Mode | Standard Mode |
|--------|----------|---------------|
| Environment | `RLMEnv` | `SingleTurnEnv` |
| Context location | `info["context"]` (accessible via code) | Directly in prompt |
| Model capability | Write Python code to explore | Process text directly |
| Best for | Very long contexts | Shorter contexts, direct comparison |

### SFT Data Generation

When `include_env_tips=True` (RLM mode only), environment-specific strategy tips are appended to the prompt inside `<env_tips>` tags. This is useful for generating SFT training data where a strong model can leverage the tips to produce high-quality demonstrations.

The tips suggest an effective chunking strategy for long-context retrieval:

1. Split the context into chunks
2. Write a prompt describing what to look for, then append it to each chunk
3. Call `llm_batch()` once with all prompts to scan chunks in parallel
4. Aggregate the relevant findings

Since the tips are wrapped in `<env_tips>` tags, they can be easily stripped from the training data so the student model learns to apply the right strategy without explicit hints.

```bash
# Generate SFT data with strategy tips
uv run vf-eval -s oolong -m gpt-4.1 --env-kwargs '{"include_env_tips": true}'
```

## Reward Functions

| Function | Weight | Description |
|----------|--------|-------------|
| `judge_reward` | 1.0 | Uses a judge model to determine if the response matches the ground truth |
| `exact_match_reward` | 0.0 | 1.0 if the model's answer exactly matches the expected answer (metric only) |
| `contains_answer_reward` | 0.0 | 1.0 if the model's answer contains the expected answer (metric only) |

### Why Use a Judge?

The dataset's prompts often require different formatting than the provided ground truth answers display. For example, a question might ask for a date in a specific format, but the ground truth stores it differently. A judge model can recognize semantic equivalence despite formatting differences.

### Differences from the Paper

The original Oolong paper uses `score = 0.75 ** abs(answer - response)` for numeric problems, allowing partial credit for close answers. This implementation uses only exact equality (via the judge) for simplicity and consistency across all problem types.

## Metrics Logging

When `metrics_output_path` is provided, detailed per-rollout metrics are logged to a JSON file for statistical analysis and RLM vs standard mode comparison.

### Logged Metrics

| Category | Metrics |
|----------|---------|
| **Identifiers** | `example_id`, `subset`, `prompt_preview` |
| **Performance** | `judge_correct`, `exact_match`, `contains_answer`, `final_answer` |
| **Main Branch** | `main_turns`, `main_tool_calls`, `main_prompt_tokens`, `main_completion_tokens` |
| **Sub-LLM (RLM only)** | `sub_llm_calls`, `sub_llm_total_tool_calls`, `sub_llm_prompt_tokens`, `sub_llm_completion_tokens`, `sub_llm_total_turns` |
| **Totals** | `total_prompt_tokens`, `total_completion_tokens`, `total_tokens`, `total_tool_calls` |
| **Timing** | `generation_ms`, `scoring_ms`, `total_ms` |
| **Mode/Errors** | `is_rlm_mode`, `had_error`, `error_message` |

### Example Analysis

```python
import pandas as pd

# Load metrics
metrics = pd.read_json("metrics.json")

# Compare token usage between modes
rlm_metrics = metrics[metrics["is_rlm_mode"] == True]
std_metrics = metrics[metrics["is_rlm_mode"] == False]

print(f"RLM mode - avg tokens: {rlm_metrics['total_tokens'].mean():.0f}")
print(f"Standard mode - avg tokens: {std_metrics['total_tokens'].mean():.0f}")
print(f"RLM mode - accuracy: {rlm_metrics['judge_correct'].mean():.2%}")
print(f"Standard mode - accuracy: {std_metrics['judge_correct'].mean():.2%}")
```

## Usage

```bash
# Evaluate with RLM mode (default)
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "synth"}'

# Evaluate with standard mode (context in prompt)
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"use_rlm": false}'

# Compare RLM vs standard mode on the same subset
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "synth", "use_rlm": true}'
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "synth", "use_rlm": false}'

# With metrics logging for comparison analysis
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"use_rlm": true, "metrics_output_path": "metrics.json"}'
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"use_rlm": false, "metrics_output_path": "metrics.json"}'

# Evaluate on synthetic subset with labels
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "synth_with_labels"}'

# Evaluate on real-world subset
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "real"}'

# With custom iterations (RLM mode)
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "synth", "max_iterations": 50}'
```

## Example Task

### RLM Mode

The model receives:

- **Query**: A question about information contained in a long document
- **Context**: Available via `context["input_data"]` in the REPL

The model writes Python code to efficiently explore the context and find the answer.

### Standard Mode

The model receives:

- **Query**: A question followed by the full context in `<context>` tags
- **Instructions**: To provide the answer inside `\boxed{}`

The model must process the entire context directly to find and return the correct answer.

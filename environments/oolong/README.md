# Oolong

A long-context evaluation environment using the RLM (Recursive Language Model) REPL.

## Description

This environment implements the [Oolong benchmark](https://huggingface.co/oolongbench) for evaluating long-context understanding capabilities of language models. The benchmark tests a model's ability to reason over and extract information from extended contexts.

The model operates in an RLM REPL environment where it can:

- Write Python code to explore the large context
- Search through text efficiently using string operations
- Make recursive sub-LLM calls if needed
- Return the final answer programmatically

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
| `max_iterations` | int | 30 | Maximum REPL iterations |
| `max_output_length` | int | 8192 | Maximum code execution output length |
| `judge_model` | str | "gpt-4.1-nano" | Model to use for judging answer correctness |
| `judge_api_key_var` | str | "OPENAI_API_KEY" | Environment variable for judge API key |

### Subset Options

- **`synth`**: Uses `context_window_text` column from oolong-synth
- **`synth_with_labels`**: Uses `context_window_text_with_labels` column from oolong-synth
- **`real`**: Uses `context_window_text` column from oolong-real

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

## Usage

```bash
# Evaluate on synthetic subset
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "synth"}'

# Evaluate on synthetic subset with labels
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "synth_with_labels"}'

# Evaluate on real-world subset
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "real"}'

# With custom iterations
uv run vf-eval -s oolong -m gpt-4.1-mini --env-kwargs '{"subset": "synth", "max_iterations": 50}'
```

## Example Task

The model receives:

- **Query**: A question about information contained in a long document
- **Context**: A large text document (available via `context["input_data"]` in the REPL)

The model must efficiently explore the context using Python code to find and return the correct answer.

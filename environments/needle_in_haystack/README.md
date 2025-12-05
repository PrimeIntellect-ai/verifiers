# Needle in Haystack

A needle-in-haystack environment using the RLM (Recursive Language Model) REPL.

## Description

This environment tests a model's ability to find specific information hidden in a large body of text using programmatic exploration. A random 7-digit "magic number" is hidden among thousands of lines of random filler text.

The model operates in an RLM REPL environment where it can:
- Write Python code to explore the context
- Search through lines efficiently
- Make recursive sub-LLM calls if needed
- Return the final answer programmatically

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | int | 10 | Number of samples to generate |
| `num_lines` | int | 10,000 | Number of lines in each haystack |

## Reward Functions

| Function | Weight | Description |
|----------|--------|-------------|
| `exact_match_reward` | 1.0 | 1.0 if the model's answer exactly matches the hidden number |
| `contains_answer_reward` | 0.0 | 1.0 if the model's answer contains the hidden number (metric only) |

## Usage

```bash
# Basic evaluation
uv run vf-eval -s needle-in-haystack -m gpt-4.1-mini

# With custom parameters
uv run vf-eval -s needle-in-haystack -m gpt-4.1-mini --env-kwargs '{"num_lines": 100000}'
```

## Example Task

The model receives:
- **Query**: "I'm looking for a magic number hidden in the context. Find it and return just the number."
- **Context**: ~10,000 lines of random text with one line containing "The magic number is 4829173"

The model must efficiently search through the context (e.g., using string search or iteration) to find and return the magic number.

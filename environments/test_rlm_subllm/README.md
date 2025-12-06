# Test RLM Sub-LLM Calls

A test environment specifically designed to verify the async sub-LLM call functionality in RLMEnv.

## Description

This environment presents a multi-section summarization task that **cannot be solved with pure Python code**. The model must use `llm()` calls for semantic understanding, making it ideal for testing:

- The async `llm()` function works correctly
- Parallel execution via `asyncio.gather()` functions as expected
- The semaphore-based parallelism control operates properly

The context contains multiple distinct text sections (articles about technology, health, environment, economics, and space). The model must summarize each section using `llm()` calls and combine them into a formatted answer.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_sections` | int | 5 | Number of sections to include (1-5) |
| `num_samples` | int | 3 | Number of examples in the dataset |

## Reward Functions

| Function | Weight | Description |
|----------|--------|-------------|
| `format_reward` | 0.5 | Checks if answer has correct numbered list format with N items |
| `length_reward` | 0.3 | Checks that summaries are reasonable length (10-200 words each) |
| `completion_reward` | 0.2 | Binary reward for producing any valid output (>50 chars) |

## Usage

```bash
# Install the environment
uv run vf-install test-rlm-subllm

# Basic evaluation
uv run vf-eval -s test-rlm-subllm -m gpt-4.1-mini -n 3 -r 1

# With custom parameters
uv run vf-eval -s test-rlm-subllm -m gpt-4.1-mini --env-kwargs '{"num_sections": 3}'
```

## Expected Model Behavior

The model should:

1. Inspect the context metadata to understand the structure
2. Parse the sections from the context
3. Use `asyncio.gather()` with multiple `llm()` calls to summarize sections in parallel
4. Format the summaries as a numbered list
5. Set `answer["ready"] = True` to complete

Example code the model might write:

```python
import asyncio

# Parse sections from context
sections = context["input_data"].split("===")
# ... extract section content ...

# Summarize all sections in parallel using async llm()
summaries = asyncio.run(asyncio.gather(
    llm(f"Summarize in one sentence: {section1}"),
    llm(f"Summarize in one sentence: {section2}"),
    llm(f"Summarize in one sentence: {section3}"),
    llm(f"Summarize in one sentence: {section4}"),
    llm(f"Summarize in one sentence: {section5}"),
))

# Format and submit answer
answer["content"] = "\n".join([f"{i+1}. {s}" for i, s in enumerate(summaries)])
answer["ready"] = True
```

## Why This Tests Sub-LLM Calls

Unlike needle-in-haystack or counting tasks, summarization requires semantic understanding that cannot be achieved with string manipulation or Python logic alone. The model **must** delegate to sub-LLM calls to generate meaningful summaries.

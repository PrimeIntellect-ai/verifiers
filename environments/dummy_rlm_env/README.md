# Dummy RLM Environment

A simple example environment demonstrating the Recursive Language Model (RLM) environment for tasks involving large context analysis.

## Description

This environment provides test cases where models need to:

1. Analyze structured data stored in context
2. Use Python code to parse and process the datav
3. Optionally use sub-LLM calls for semantic understanding
4. Return final answers via the answer variable

## Setup

```bash
vf-install dummy-rlm-env
```

## Usage

```bash
vf-eval -s dummy-rlm-env -m gpt-4.1-mini -n 3
```

## RLM Environment Interface

### Dataset Structure

RLMEnv works with any dataset that has a normal `prompt` field:

```python
{
    "prompt": [{"role": "user", "content": "Your question here"}],
    "answer": "expected answer",
    # Optional: large context that shouldn't be in the prompt
    "info": {
        "context": "...large data..."
    }
}
```

### Context (available in code as `context`)

```python
context = {
    "input_data": "...",           # The context data (or None if not provided)
    "input_data_metadata": {       # Metadata about the input
        "type": "string",
        "size": 1234,
        ...
    },
}
```

### Answer Variable (available in code as `answer`)

The model must set its answer using:

```python
answer["ready"] = True
answer["content"] = "your answer here"
```

### Available Functions

- `llm(prompt, **kwargs)`: Make a sub-LLM call for semantic tasks
- `rlm(prompt, sub_context=None, **kwargs)`: Make a recursive RLM call with optional context

### State Keys Set by RLMEnv

After rollout, the environment sets `state["final_answer"]` to the model's answer string (empty if none provided).

```python
def my_reward(state: vf.State) -> float:
    predicted = state.get("final_answer", "")
    expected = state.get("answer", "")
    
    if not predicted:
        return 0.0  # No answer provided
    
    return 1.0 if predicted == expected else 0.0
```

## Call Type Logging

API calls are tagged for logging/analysis:

- Root model calls: `extra_body={"rlm_call_type": "root"}`
- Sub-LLM calls: `extra_body={"rlm_call_type": "sub"}`

# Type Reference

This guide explains the key types and data structures in Verifiers.

## Core Types

### Pydantic Models

Verifiers uses Pydantic models for structured data:

```python
from pydantic import BaseModel

class GenerateInputs(BaseModel):
    """Pydantic model for generation inputs."""

    prompt: list[Messages]
    answer: list[str] | None = None
    info: list[dict] | None = None
    task: list[str] | None = None
    completion: list[Messages] | None = None
    id: list[int] | None = None

class GenerateOutputs(BaseModel):
    """Pydantic model for generation outputs."""

    prompt: list[Messages]
    completion: list[Messages]
    answer: list[str]
    state: list[State]
    info: list[Info]
    task: list[str]
    id: list[int]
    reward: list[float]
    metrics: dict[str, list[float]] = Field(default_factory=dict)
    metadata: GenerateMetadata

class RolloutScore(BaseModel):
    """Pydantic model for rollout scores."""

    reward: float
    metrics: dict[str, float] = Field(default_factory=dict)


class RolloutScores(BaseModel):
    """Pydantic model for rubric outputs."""

    reward: list[float]
    metrics: dict[str, list[float]] = Field(default_factory=dict)


class ProcessedOutputs(BaseModel):
    """Pydantic model for processed outputs."""

    prompt_ids: list[list[int]]
    prompt_mask: list[list[int]]
    completion_ids: list[list[int]]
    completion_mask: list[list[int]]
    completion_logprobs: list[list[float]]
    rewards: list[float]
```

`GenerateOutputs.metadata` captures run-level context (environment + arguments, model + sampling configuration, summary statistics, and the resolved save path) so downstream tooling can reproduce or resume evaluations without guessing defaults.

### State Dictionary

The `State` object tracks rollout information throughout an interaction:

```python
State = dict[str, Any]

# Common state fields during rollout:
{
    "prompt": list[ChatMessage],      # Original prompt messages
    "completion": list[ChatMessage],  # Model's response messages
    "answer": str,                    # Ground truth answer
    "task": str,                      # Task identifier (for EnvGroup)
    "info": dict[str, Any],          # Additional metadata from dataset
    "responses": list[Any],          # Raw LLM response objects
    "example_id": int,                # Row identifier from the dataset
    "timing": dict[str, float],       # Timing information for generation and scoring
    
    # Custom fields added by specific environments:
    "turn": int,                     # Current turn number (MultiTurnEnv)
    "tools_called": list[str],       # Tool invocations (ToolEnv)
    "game_state": Any,               # Game-specific state
}
```

The `responses` field contains raw API response objects with:
- `choices[0].logprobs.content`: Token-level log probabilities
- `choices[0].logprobs.token_ids`: Token IDs
- `choices[0].finish_reason`: Why generation stopped
- `usage`: Token usage statistics

### Message Formats

```python
# Import from verifiers.types
from verifiers.types import ChatMessage, Messages

# Chat format (recommended)
# ChatMessage is a dict with these fields:
ChatMessage = {
    "role": str,                    # "system", "user", or "assistant"
    "content": str,                 # Message text
    "tool_calls": list[...],        # Optional tool calls
    "tool_call_id": str,            # Optional tool call ID
}

Messages = str | list[ChatMessage]  # Can be string (completion) or chat

# Example chat format:
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
]

# Completion format (legacy):
completion = "Q: What is 2+2?\nA: 4"
```

### Reward Function Signature

All reward functions must follow this signature:

```python
from collections.abc import Awaitable, Callable

RewardFunc = Callable[..., float | Awaitable[float]]

def my_reward_func(
    completion: Messages,            # Model's response (chat or string)
    answer: str = "",                # Ground truth answer
    prompt: Messages | None = None,  # Original prompt
    state: State | None = None,      # Environment state
    parser: Parser | None = None,    # Parser instance (if rubric has one)
    **kwargs                         # Additional arguments
) -> float:
    """Return a float reward between 0.0 and 1.0."""
    return 1.0
```

### Environment Response

For `MultiTurnEnv.env_response`:

```python
def env_response(
    self,
    messages: list[ChatMessage],
    state: State,
    **kwargs
) -> tuple[Messages, State]:
    """
    Returns:
        - Response messages (list[ChatMessage] or str for completion mode)
        - Updated state
    """
    # Return a list of ChatMessage dicts (typical case)
    response = [{"role": "user", "content": "Environment feedback"}]
    
    # Update state as needed
    state["turn"] = state.get("turn", 0) + 1
    state["last_action"] = "provided feedback"
    
    return response, state
```

### Sampling Arguments

vLLM-specific generation parameters:

```python
SamplingArgs = dict[str, Any]

sampling_args = {
    # Basic sampling
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 2048,
    
    # Advanced vLLM options
    "extra_body": {
        "logprobs": True,              # Return token logprobs
        "top_logprobs": 5,             # Top-k logprobs per token
        "skip_special_tokens": False,  # Include special tokens
        "guided_decoding": {           # Structured generation
            "regex": r"\d{3}-\d{3}-\d{4}"  # Phone number format
        }
    }
}
```

### Dataset Info

The `info` field in datasets can contain arbitrary metadata:

```python
Info = dict[str, Any]

# Dataset row with info dict:
{
    "prompt": "Solve this problem",
    "info": {
        "answer": "42",              # Required: ground truth
        "difficulty": "medium",      # Optional metadata
        "source": "textbook",
        "chapter": 3,
        "requires_tool": True
    }
}

# Access in reward functions:
def reward_func(completion, answer, info=None, **kwargs):
    difficulty = info.get("difficulty", "unknown") if info else "unknown"
    # Adjust scoring based on difficulty...
```

## Type Utilities

### Environment Rollout Types

```python
# Rollout returns
async def rollout(...) -> tuple[Messages, State]:
    """Returns (completion, final_state)"""

# Evaluation results
async def evaluate(...) -> GenerateOutputs:
    """Async interface that returns GenerateOutputs."""

# Synchronous convenience wrappers
def evaluate_sync(...) -> GenerateOutputs:
    """Blocking helper that wraps the async evaluate coroutine."""

# Generation results
async def generate(...) -> GenerateOutputs:
    """Async interface that returns GenerateOutputs containing rollout data"""

def generate_sync(...) -> GenerateOutputs:
    """Blocking helper for integrate-with-sync-code scenarios"""
```

### Parser Types

```python
# Parser return types can be anything
def parse(text: str) -> Any:
    """Can return str, dict, dataclass, etc."""

# parse_answer must return optional string
def parse_answer(completion: Messages) -> str | None:
    """Must return string answer or None"""
```

## Common Patterns

### Accessing Completion Content

```python
def get_text_content(completion: Messages) -> str:
    """Extract text from either format."""
    if isinstance(completion, str):
        return completion
    else:
        # Chat format - get last assistant message
        return completion[-1]["content"]
```

### State Initialization

```python
def reset_for_rollout(self, prompt: Messages, answer: str, info: Info | None) -> State:
    """Initialize state for new rollout."""
    state = {
        "prompt": prompt,
        "answer": answer,
        "info": info or {},
        "task": info.get("task", "default") if info else "default",
        "responses": [],
        # Add custom fields
        "turn": 0,
        "history": []
    }
    return state
```


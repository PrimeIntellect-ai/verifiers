# Overview

Verifiers provides a flexible framework for defining custom interaction protocols between LLMs and environments, enabling sophisticated multi-turn reasoning, tool use, and interactive evaluation.

The three key pieces of environments in Verifiers are:
- Your dataset (`str` or `List[ChatMessage]`)
- Your Rubric (one or more *reward functions*)
- Your *interaction protocol*, extended from `MultiTurnEnv`

## Core Concept: Interaction Protocols

Verifiers allows defining arbitrary interaction patterns between models and environments:

```
Environment (orchestration layer)
    ├── Defines interaction protocol (what to observe respond, how to respond, when to terminate)
    ├── Manages conversation state
    ├── Integrates tools and external resources
    └── Evaluates performance via Rubrics
```

### Example Protocols

- **Q&A Tasks**: Single model response → evaluation
- **Tool Use**: Model request → tool execution → model continues
- **Games**: Model move → game state update → environment feedback → repeat
- **Tutoring**: Model attempt → hint/correction → retry until correct
- **Debate**: Model A argument → Model B rebuttal → judge evaluation

## Environment Types

### MultiTurnEnv: Maximum Flexibility

The base class for custom interaction protocols:

```python
import verifiers as vf
from verifiers.types import Messages, State
from typing import Tuple

class MyProtocol(vf.MultiTurnEnv):
    async def env_response(self, messages: Messages, state: State) -> Tuple[Messages, State]:
        """Define how environment responds to model"""
        # Custom logic for your protocol
        response = [{"role": "user", "content": "Environment feedback"}]
        # Update state
        state["turn"] = state.get("turn", 0) + 1
        return response, state
    
    async def is_completed(self, messages: Messages, state: State) -> bool:
        """Define when interaction ends"""
        return state.get("task_complete", False)
```

### ToolEnv: Native Tool Calling

Leverages models' built-in tool calling for agentic workflows:

```python
env = vf.ToolEnv(
    tools=[search, calculate, execute_code],  # Python functions
    max_turns=10,
    dataset=dataset,
    rubric=rubric
)
```

Tools may be sync or async, and are automatically converted to JSON schemas and integrated with the model's native function calling format.

### SingleTurnEnv: Simple Evaluation

For straightforward Q&A tasks without interaction:

```python
env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Answer the question."
    rubric=rubric,
)
```

## Key Components

### Rubrics: Multi-Criteria Evaluation

Rubrics define how to evaluate model responses by combining multiple criteria:

```python
# Simple reward function (can be sync or async)
async def correctness(prompt, completion, answer, state):
    return 1.0 if answer.lower() in completion[-1]['content'].lower() else 0.0

# Combine multiple criteria
rubric = vf.Rubric(
    funcs=[correctness, efficiency, clarity],
    weights=[1.0, 0.3, 0.2]  # Relative importance
)
```

Each reward function receives the full context (prompt, response, ground truth answer, and environment state) and returns a score. The rubric combines these scores based on weights to produce a final reward.

Common rubric patterns:
- **Single criterion**: One reward function (e.g., exact match)
- **Multi-criteria**: Weighted combination of multiple aspects
- **Judge-based**: Using LLMs to evaluate quality
- **Stateful**: Tracking patterns across interactions

### Environment Modules

Package your interaction protocol as a reusable module:

```
my_environment/
├── outputs/                # Evaluation logs
├── my_environment.py      # Defines load_environment() -> vf.Environment
├── pyproject.toml        # Dependencies
└── README.md            # Documentation
```

This enables:
- Easy sharing and versioning
- Dependency isolation
- Standardized interfaces


### State Management

Environments maintain state throughout interactions:

```python
state = {
    # automatically managed  
    "prompt": prompt, # inputs from dataset
    "completion": [], # trajectory so far
    "answer": answer, # golden answer (str)
    "task": task, # optional environment ID column
    "info": info, # evaluation metadata (dict) -- can use answer/info/both
    "responses": [], # Raw API responses from OpenAI client
    "turn": 0,
    # custom user-managed state
    "lives_remaining": 2,
    "inventory": {"potion": 1, "power-up": 2}
    ...
}
```

A wide variety of complex interaction protocols, reward schemes, and training algorithms can be coordinated via tracking appropriate data in `state`.

## Design Philosophy

### 1. Protocol-First Design

Start by defining your interaction pattern:
- When should the environment respond?
- What information should it provide?
- How should the conversation end?

### 2. Composable Evaluation

Build complex evaluation from simple parts:
- Individual reward functions for specific criteria
- Rubrics to combine and weight them
- Environments to orchestrate the process

### 3. OpenAI-Compatible Integration

Works with any OpenAI-compatible API:
```python
# OpenAI, vLLM, or any compatible endpoint
client = OpenAI(base_url="http://localhost:8000/v1") # or AsyncOpenAI
results = env.evaluate(client, model="llama-3.1-8b")
```

## Data Flow

1. **Dataset** provides prompts and ground truth
2. **Environment** orchestrates the interaction protocol
3. **Model** generates responses via OpenAI-compatible client
4. **Rubric** evaluates quality through reward functions
5. **Results** include full interaction traces and scores

## Evaluation lifecycle

- **Inputs expected by environments**:
  - `prompt`: str or list[ChatMessage] (chat-style). If you use `question` in your dataset, environments will turn it into a chat message, adding `system_prompt`/`few_shot` if provided.
  - `answer` or `info`: at least one is required. `answer` is a string; `info` is a dict for richer metadata.
  - `task`: optional string used by `EnvGroup`/`RubricGroup` to route behavior.

- **Running evaluation**:
  ```python
  results = env.evaluate(
      client, model,
      num_examples=100,
      rollouts_per_example=2,
      max_concurrent=32,
  )
  ```
  - `rollouts_per_example > 1` repeats dataset entries internally.
  - `max_concurrent` throttles concurrent rollouts.

- **Scoring**:
  - Each reward function returns a float. Weights applied inside `Rubric` combine them into `results.reward`.
  - All individual scores are logged under `results.metrics` keyed by function name (even if weight is 0.0).

- **Outputs** (`GenerateOutputs`):
  - `prompt`, `completion`, `answer`, `state`, `info`, `task`, `reward`, `metrics: dict[str, list[float]]`.

- **Message types**:
  - `message_type="chat"` (default) expects chat messages; `"completion"` expects raw text continuation. Choose based on your task (e.g., continuation quality uses completion).

## Optional Utilities

### Parsers

For extracting structured information when needed:
- `XMLParser`: Extract XML-tagged fields
- `ThinkParser`: Separate reasoning from answers
- Custom parsers for domain-specific formats

Parsers are optional conveniences - many environments work perfectly with raw text.

## Integration Points

### For Evaluation

The most convenient way to run quick evaluations is via the `vf-eval` CLI tool:
```bash
vf-install my-environment-module # from ./environments/my_environment_module 
vf-eval my-environment-module -m gpt-5 -n 10 -r 5 -s 
```

We also provide a TUI for browsing locally-cached (with `-s`) eval results:
```bash
vf-tui 
```

You can also evaluate models in your environments programmatically:
```python
results = env.evaluate(client, model, num_examples=100)
```

### For Training

```python
# Environments provide rollout-level interfaces for RL
completion, state = await env.rollout(
    client, model, prompt, answer
)
rewards = await env.rubric.score_rollout(
    prompt, completion, answer, state
)
# Or, process rollouts in batches for high throughput and configurable coordination
outputs = await env.a_generate(inputs, client, model, sampling_args) # `generate` for sync
```

### For Custom Workflows

All components can be used independently:
```python
# Use rubrics standalone
scores = await rubric.score_rollout(prompt, completion, answer, state)

# Create custom protocols
class MyProtocol(vf.MultiTurnEnv):
    # Your interaction logic
```

## Next Steps

- To create custom interactions, see [Environments](environments.md)
- For advanced component usage and examples, see [Components](components.md)
- To train models with your environments, see [Training](training.md)
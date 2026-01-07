# FAQs

## Getting Started

### How do I quickly test my environment?

Use `vf-eval` with a small sample:

```bash
vf-eval my-environment -m gpt-4.1-mini -n 5 -s
```

The `-s` flag prints sample outputs so you can see what's happening.

### How do I see what the model is outputting?

**If using `vf-eval`**: Results are saved automatically. Browse them interactively with:

```bash
vf-tui
```

**If using the Python API** (`env.generate()` / `env.evaluate()`):

```python
vf.print_prompt_completions_sample(outputs, n=3)
```

### How do I enable debug logging?

Set the `VF_LOG_LEVEL` environment variable:

```bash
VF_LOG_LEVEL=DEBUG vf-eval my-environment -m gpt-4.1-mini -n 5
```

## Environments

### Which environment class should I use?

- **SingleTurnEnv**: One prompt, one response (Q&A, classification)
- **MultiTurnEnv**: Custom back-and-forth interaction (games, simulations)
- **ToolEnv**: Model calls Python functions (search, calculator)
- **StatefulToolEnv**: Tools that need per-rollout state (sandbox IDs, sessions)

### What does `max_turns=-1` mean?

Unlimited turns. The rollout continues until a stop condition is triggered (e.g., model stops calling tools, or a custom condition you define).

### How do I add a custom stop condition?

Use the `@vf.stop` decorator on a method that returns `True` to end the rollout:

```python
@vf.stop
async def task_completed(self, state: State) -> bool:
    return "DONE" in state["completion"][-1]["content"]
```

### How do I handle tool call errors gracefully?

In `ToolEnv`, customize error handling:

```python
env = ToolEnv(
    tools=[my_tool],
    error_formatter=lambda e: f"Error: {type(e).__name__}: {e}",
    stop_errors=[CriticalError],  # These errors end the rollout
)
```

Non-critical errors are returned to the model as tool responses so it can retry.

## Reward Functions

### What arguments can my reward function receive?

Reward functions receive any of these via `**kwargs`:

- `completion` - the model's response
- `answer` - ground truth from dataset
- `prompt` - the input prompt
- `state` - full rollout state
- `parser` - the rubric's parser (if set)
- `task` - task identifier
- `info` - metadata dict from dataset

Just include the ones you need in your function signature.

### How do group reward functions work?

Group reward functions receive plural arguments (`completions`, `answers`, `states`) and return a list of floats. They're detected automatically by parameter names:

```python
def relative_reward(completions: list, answers: list, **kwargs) -> list[float]:
    # Score all completions for an example together
    scores = [compute_score(c, a) for c, a in zip(completions, answers)]
    # Normalize relative to group
    max_score = max(scores) if scores else 1.0
    return [s / max_score for s in scores]
```

## Training

### What's the difference between `prime-rl` and `vf-rl`?

- **prime-rl**: Production-ready, multi-node, MoE support, advanced features. Use for serious training.
- **vf-rl**: Minimal (~1000 LOC), single-node, hackable. Use for small-scale testing or as a starting point for your own training loop.

Both use the same core algorithm (async CISPO).

### How do I use a local vLLM server?

Point the client to your local server:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

outputs = await env.evaluate(client, model="your-model-name", ...)
```
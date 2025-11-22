# GEPA: Prompt Optimization

GEPA (Gradient-free Evolutionary Prompt Adaptation) is an automatic prompt optimization system that improves your environment's system prompts and tool descriptions based on rubric feedback.

## Overview

GEPA works by:
1. Testing your current prompts on examples
2. Analyzing failures using rubric feedback
3. Generating improved prompts through reflection
4. Iteratively refining until convergence

This is particularly effective when combined with `FeedbackRubric`, which provides rich textual feedback explaining why rollouts succeeded or failed.

## Installation

GEPA is available as an optional dependency:

```bash
uv add 'verifiers[gepa]'
```

This installs the `gepa` optimization engine.

## Quick Start

Optimize the system prompt for an environment:

```bash
vf-gepa wordle --auto medium
```

This will:
- Load the `wordle` environment
- Use medium budget (~12 candidate prompts)
- Optimize the `system_prompt` component
- Save results to `./gepa_results/wordle/<run_id>/`

## Budget Modes

GEPA offers three auto budget levels:

### Light (~6 candidates)
Fast iteration for testing:
```bash
vf-gepa my-env --auto light
```
- Best for: Quick experiments, sanity checks
- Time: ~5-10 minutes for simple environments
- Use when: Testing GEPA setup, iterating rapidly

### Medium (~12 candidates)  
Balanced optimization:
```bash
vf-gepa my-env --auto medium
```
- Best for: Most use cases, good improvements
- Time: ~15-30 minutes for simple environments
- Use when: Standard optimization runs

### Heavy (~18 candidates)
Thorough exploration:
```bash
vf-gepa my-env --auto heavy
```
- Best for: Final production prompts, critical environments
- Time: ~30-60 minutes for simple environments
- Use when: You need the best possible prompt

### Custom Budget

For fine control, specify exact metric calls:
```bash
vf-gepa my-env --max-metric-calls 1000
```

## Component Selection

By default, GEPA optimizes `system_prompt`. You can specify multiple components:

### System Prompt Only
```bash
vf-gepa my-env --auto medium --components system_prompt
```

### Tool Descriptions
For environments with tools, optimize their descriptions:
```bash
vf-gepa wiki-search --auto medium --components tool_descriptions
```

### Both System Prompt and Tool Descriptions
```bash
vf-gepa wiki-search --auto heavy --components system_prompt tool_descriptions
```

When optimizing `tool_descriptions`, GEPA:
1. Extracts each tool's description from `oai_tools`
2. Treats each as a separate component to optimize
3. Uses separate reflection for each tool
4. Injects optimized descriptions back into tools

## Model Configuration

### Task Model
The model being optimized (default: `gpt-4o-mini`):
```bash
vf-gepa my-env --auto medium -m gpt-4o
```

### Reflection Model
The model generating improved prompts (default: `gpt-4o`):
```bash
vf-gepa my-env --auto medium --reflection-model gpt-4o
```

### Sampling Parameters
```bash
vf-gepa my-env --auto medium \
  -T 0.7 \              # Temperature for task model
  -t 2048 \             # Max tokens
  --reflection-temperature 1.0  # Temperature for reflection
```

## Dataset Configuration

Control train/validation split sizes:

```bash
vf-gepa my-env --auto medium \
  -n 100 \              # 100 training examples
  --num-val 30          # 30 validation examples
```

**Guidelines**:
- Training: 50-100 examples (more = slower but potentially better)
- Validation: 20-30 examples (for measuring improvement)
- Use representative examples that cover your task's diversity

## Output

GEPA saves three files to `./gepa_results/<env_id>/<run_id>/`:

### 1. `<env_id>_optimized.json`
The optimized components:
```json
{
  "system_prompt": "You are a competitive Wordle player...",
  "tool_0_description": "Search Wikipedia for..."
}
```

### 2. `<env_id>_original.json`
The original components for comparison.

### 3. `<env_id>_metrics.json`
Optimization metrics:
```json
{
  "best_val_score": 0.85,
  "initial_val_score": 0.62,
  "improvement": 0.23,
  "num_candidates": 12,
  "candidates_history": [...]
}
```

## Rubric Feedback Support

For best results, have your reward functions return feedback:

```python
import verifiers as vf

def accuracy_with_feedback(parser, completion, answer, **kwargs):
    """Reward function that returns score + feedback."""
    guess = parser.parse_answer(completion)
    correct = (guess == answer)
    
    return {
        "score": 1.0 if correct else 0.0,
        "feedback": (
            f"{'✓' if correct else '✗'} "
            f"Expected: {answer}, Got: {guess}"
        )
    }

rubric = vf.Rubric(parser=parser)
rubric.add_reward_func(accuracy_with_feedback)
```

The `feedback` field is used by GEPA to understand *why* completions failed, enabling better prompt improvements. The base `Rubric` class automatically collects feedback via its `get_feedback()` method.

## Advanced Usage

### Multiple Rollouts Per Example
Increase robustness with multiple rollouts:
```bash
vf-gepa my-env --auto medium --rollouts-per-example 3
```

### Custom Log Directory
```bash
vf-gepa my-env --auto medium --log-dir ./my_optimization_runs
```

### Track Detailed Statistics
Save full outputs for analysis:
```bash
vf-gepa my-env --auto medium --track-stats
```

### Verbose Logging
Debug optimization process:
```bash
vf-gepa my-env --auto medium -v
```

## Best Practices

### 1. Provide Rich Feedback
GEPA works best when reward functions return textual feedback explaining scores. If your functions only return numbers, GEPA has less to work with.

**Good**: 
```python
return {
    "score": 0.5,
    "feedback": "Partially correct. Got step 1 right but step 2 is missing."
}
```

**OK but less effective**:
```python
return 0.5  # GEPA will only see the number
```

### 2. Use Representative Examples
Ensure your training and validation sets cover the full range of task difficulty and variety.

### 3. Start Light, Then Scale Up
Begin with `--auto light` to verify everything works, then use `medium` or `heavy` for production.

### 4. Iterate on Feedback Quality
If GEPA improvements are small, review your rubric's feedback. More specific feedback = better improvements.

### 5. Version Control Prompts
Save optimized prompts in your repo and track which version is in production.

## Troubleshooting

### "Error: GEPA is not installed"
```bash
uv add 'verifiers[gepa]'
```

### "Environment does not have component 'X'"
Check that your environment exposes the component you're trying to optimize. Use `--components system_prompt` (default) if unsure.

## Limitations

### Unsupported Environment Types
- **EnvGroup**: GEPA operates on a single environment at a time. Optimize each member separately, then compose them with `EnvGroup`.
- **Dynamic tools**: Environments that mutate their tool list during `__init__` or per rollout may not preserve those changes across candidate reconstruction.

### Requirements
- Components you optimize must be attributes on the environment object (e.g., `system_prompt`).
- `tool_descriptions` optimization requires `oai_tools` to be defined up front.
- Reward functions should emit textual feedback to unlock GEPA's reflection step.

### Operational Constraints
- Multiple rollouts per example scale linearly in cost—start small before increasing `--rollouts-per-example`.
- Heavy budgets require high-quality validation datasets; under-sized eval sets can hide regressions.
- GEPA expects deterministic environment construction. Expensive setup code will re-run for every candidate.

### Low Improvement
- Increase budget: Use `--auto heavy` or `--max-metric-calls 2000`
- Improve feedback: Make your rubric's feedback more specific
- Add more examples: Use `-n 100 --num-val 30`
- Check dataset quality: Ensure examples are representative

### Out of Memory
- Reduce batch sizes: `--reflection-minibatch-size 2`
- Reduce examples: `-n 30 --num-val 10`
- Use smaller models: `-m gpt-4o-mini`

## Examples

### Basic Optimization
```bash
vf-gepa wordle --auto medium
```

### Tool-Using Environment
```bash
vf-gepa wiki-search --auto heavy \
  --components system_prompt tool_descriptions \
  -m gpt-4o
```

### Large-Scale Optimization
```bash
vf-gepa my-env --max-metric-calls 2000 \
  -n 200 --num-val 50 \
  --rollouts-per-example 3 \
  --track-stats
```

### Custom Models
```bash
vf-gepa my-env --auto medium \
  -m claude-3-5-sonnet-20241022 \
  --reflection-model gpt-4o
```

## API Usage

For programmatic use:

```python
import verifiers as vf
from verifiers.adapters import GEPAAdapter
from gepa import optimize

# Load environment
env = vf.load_environment("wordle")

# Create adapter
adapter = GEPAAdapter(
    env=env,
    client=client,
    model="gpt-4o-mini",
    sampling_args={"temperature": 1.0, "max_tokens": 8096},
    components_to_optimize=["system_prompt"],
)

# Run optimization
result = optimize(
    seed_candidate={"system_prompt": env.system_prompt},
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    max_metric_calls=500,
    reflection_lm=reflection_function,
)

# Access results
best_prompt = result.best_candidate["system_prompt"]
improvement = max(result.val_aggregate_scores) - result.val_aggregate_scores[0]
```

## Further Reading

- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [GEPA Documentation](https://dspy.ai/api/optimizers/GEPA/overview/)
- [Creating Environments](environments.md)


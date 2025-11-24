# GEPA Integration

GEPA (Genetic-Pareto) integration for Verifiers environments.

## Overview

This integration enables automatic prompt optimization using GEPA, a reflection-based optimization system that improves prompts by analyzing rubric feedback. GEPA works by:

1. Running your environment with current prompts
2. Collecting rich feedback from rubric evaluations
3. Using an LLM to reflect on failures and propose improvements
4. Iteratively refining prompts until convergence

## Installation

```bash
uv sync --extra gepa
```

This installs the `gepa` package (>=0.0.22).

## Quick Start

Optimize a system prompt:

```bash
vf-gepa wordle --budget medium
```

Optimize system prompt + tool descriptions:

```bash
vf-gepa wiki-search --budget heavy --components system_prompt tool_descriptions
```

## Components

### `adapter.py`

The `GEPAAdapter` class bridges Verifiers environments to GEPA's optimization protocol:

- **Component management**: Extracts and injects optimizable components (system prompts, tool descriptions)
- **Evaluation**: Runs rollouts and collects scores
- **Feedback generation**: Converts rubric feedback into reflection data
- **Tool optimization**: Splits tool descriptions into separate optimizable components

### Key Methods

```python
from verifiers.adapters.gepa import GEPAAdapter

adapter = GEPAAdapter(
    env=vf_env,
    client=async_client,
    model="gpt-4o-mini",
    sampling_args={"temperature": 1.0},
    components_to_optimize=["system_prompt"],
)

# Build new environment with optimized components
new_env = adapter.build_program({"system_prompt": "Optimized prompt..."})

# Evaluate candidate prompts
results = adapter.evaluate(batch, candidate, capture_traces=True)

# Generate reflection dataset for GEPA
reflective_data = adapter.make_reflective_dataset(candidate, results, components)
```

## Rubric Feedback

GEPA works best when reward functions return structured feedback:

```python
def accuracy_with_feedback(parser, completion, answer, **kwargs):
    guess = parser.parse_answer(completion)
    correct = (guess == answer)
    
    return {
        "score": 1.0 if correct else 0.0,
        "feedback": f"Expected: {answer}, Got: {guess}. {explain_why(...)}"
    }

rubric = vf.Rubric(parser=parser)
rubric.add_reward_func(accuracy_with_feedback)
```

The `feedback` field provides context GEPA uses to understand failures and generate better prompts. Without it, GEPA only sees numeric scores.

## Tool Description Optimization

When optimizing `tool_descriptions`, the adapter:

1. Extracts each tool's description from `env.oai_tools`
2. Creates separate components: `tool_0_description`, `tool_1_description`, etc.
3. Optimizes each independently through GEPA's reflection process
4. Reconstructs `oai_tools` with improved descriptions

Example:

```bash
vf-gepa my-env --components tool_descriptions --budget medium
```

## Architecture

```
┌─────────────────┐
│  GEPA Engine    │
│  (reflection +  │
│   proposals)    │
└────────┬────────┘
         │
         ├─ evaluate()
         ├─ make_reflective_dataset()
         └─ build_program()
         │
┌────────▼────────┐
│  GEPAAdapter    │
│  (integrations/ │
│   gepa)         │
└────────┬────────┘
         │
         ├─ rollout()
         ├─ score_rollout()
         └─ get_feedback()
         │
┌────────▼────────┐
│  Verifiers Env  │
│  (dataset +     │
│   rubric)       │
└─────────────────┘
```

## Configuration

### Budget Modes

- **light** (~6 candidates): Fast iteration, ~5-10 min
- **medium** (~12 candidates): Balanced, ~15-30 min
- **heavy** (~18 candidates): Thorough, ~30-60 min

### Dataset Sizes

- Training: 50-100 examples (more = slower but potentially better)
- Validation: 20-30 examples (for measuring improvement)

### Models

- **Task model** (being optimized): `gpt-4o-mini`, `gpt-4o`, or custom
- **Reflection model** (generating proposals): `gpt-4o` recommended

## Output

GEPA saves results to `./gepa_results/<env_id>/<run_id>/`:

- `<env_id>_optimized.json` - Optimized components
- `<env_id>_original.json` - Original components (for comparison)
- `<env_id>_metrics.json` - Optimization metrics and history

## Implementation Notes

### Packaging

The GEPA adapter ships inside the `verifiers.adapters` package so it is available to `pip install verifiers` users. The legacy `integrations/gepa` module re-exports the same class for backward compatibility inside this repository.

### Feedback Collection

The base `Rubric` class automatically collects feedback when reward functions return dicts with `"feedback"` keys. The adapter checks for `rubric.get_feedback(state)` to retrieve combined feedback from all functions.

### Error Handling

The adapter validates:
- Environment has requested components (`system_prompt`, `oai_tools`)
- Tool descriptions can only be optimized if environment has tools
- Reflection datasets require `capture_traces=True`

## CLI Reference

Full documentation: [`docs/source/gepa.md`](../../docs/source/gepa.md)

```bash
# Basic
vf-gepa ENV_ID --budget light|medium|heavy

# Advanced
vf-gepa ENV_ID \
  --max-metric-calls 1000 \
  -n 100 --num-val 30 \
  --components system_prompt tool_descriptions \
  -m gpt-4o \
  --reflection-model gpt-4o \
  --rollouts-per-example 3

# Options
  -n, --num-examples       Training examples (default: 50)
  --num-val               Validation examples (default: 20)
  --budget                Budget preset: light/medium/heavy
  --max-metric-calls      Custom budget (total metric calls)
  --components            What to optimize (default: system_prompt)
  -m, --model             Task model (default: gpt-4o-mini)
  --reflection-model      Reflection model (default: gpt-4o)
  -T, --temperature       Task model temperature (default: 1.0)
  -t, --max-tokens        Max tokens (default: 8096)
  --track-stats           Save detailed statistics
  -v, --verbose           Verbose logging
```

## Links

- [GEPA Documentation](../../docs/source/gepa.md) - Complete usage guide
- [GEPA Paper](https://arxiv.org/abs/2507.19457) - Original research
- [GEPA API Docs](https://dspy.ai/api/optimizers/GEPA/overview/) - DSPy reference
- [Creating Environments](../../docs/source/environments.md) - Build custom environments

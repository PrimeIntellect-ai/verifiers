## Overview

This PR implements sparse metrics / rubrics, which enables mathematically correct averaging in multi-domain environments. The key change heere is selective averaging that excludes irrelevant zero values, solving the domain dilution problem in composite evaluation environments.

In environments like [`ProfBench`](https://arxiv.org/pdf/2510.18941), domain-specific scores get mixed with irrelevant zeros, making the averages misleading.

**Example Issue:**
Evaluating GPT-4 on 12 tasks: 10 physics + 2 chemistry tasks

```
physics_reward: [65, 72, 58, 81, 45, 67, 73, 59, 68, 74, 0, 0]  # zeros for chemistry tasks
chemistry_reward: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 76]        # zeros for physics tasks
```

- **Before**: `physics_reward: avg - 56.2` (diluted by irrelevant zeros)
- **Before**: `chemistry_reward: avg - 13.7` (misleading!)  

After,

```
physics_reward: [65, 72, 58, 81, 45, 67, 73, 59, 68, 74, -, -]  # zeros for chemistry tasks
chemistry_reward: [-, -, -, -, -, -, -, -, -, -, 88, 76]        # zeros for physics tasks
```

- **After**: `chemistry_reward: avg - 82.0 (relevant: 2/12)` (actual chemistry skill)
- **After**: `physics_reward: avg - 66.2 (relevant: 10/12)` (pure physics performance)

Which can all be done now within an `EnvGroup` with `enable_sparse_metrics=True`.

we can now

1. mark irrelevant values as sparse during scoring
2. exclude sparse values from averaging calculations
3. display sparsity clearly with `-` instead of `0.0`
4. maintain backwards compatibility with existing environments

## Core 

### 1. type extensions @ `types.py`

**New Fields Added:**

```python
class RolloutScore(BaseModel):
    sparse_metrics: set[str] | None = Field(default=None)
    # set of metric names to exclude from averaging for this rollout

class RolloutScores(BaseModel): 
    sparse_metrics: dict[str, list[bool]] | None = Field(default=None)
    # per-rolout exclusion flags for batch scoring

class GenerateOutputs(BaseModel):
    sparse_metrics: dict[str, list[bool]] | None = Field(default=None)
    # final sparse tracking for evaluation results
```

THis tracks which metric values should be excluded from averaging calculations.

### 2. Environment Sparse Tracking @ `envs/environment.py`

**Key Changes:**
- **Initialize sparse flags** for all metrics during interleaved scoring
- **Track sparse metrics** from rubric scoring results  
- **Conditionally assign** sparse_metrics only if sparsity detected (backwards compatible)

```python
# Initialize sparse tracking
sparse_flags: dict[str, list[bool]] = {name: [False] * n for name in reward_func_names}

# Process sparse flags from scoring
if rs.sparse_metrics:
    for sparse_key in rs.sparse_metrics:
        sparse_flags[sparse_key][i] = True

# Only add if sparsity detected (backwards compatible)
if any(any(flags) for flags in sparse_flags.values()):
    results.sparse_metrics = sparse_flags
```

this collects and aggregates sparse metadata during evaluation execution.

### 3. Batch Scoring with Sparse Handling @ `rubrics/rubric.py` 

**Key Changes:**
- **Collect all metric keys** across rollouts (handles mixed metrics)
- **Fill missing metrics** with 0.0 and mark as sparse
- **Track sparsity flags** from individual rollout scores
- **Return sparse metadata** only if sparsity detected

```python
# Handle missing metrics as sparse
if k in reward.metrics:
    metrics[k].append(reward.metrics[k])
    is_sparse = reward.sparse_metrics and k in reward.sparse_metrics
    sparse_flags[k].append(is_sparse)
else:
    # Missing metric -> sparse 0.0
    metrics[k].append(0.0)
    sparse_flags[k].append(True)
```

ensure consistent metric structure while preserving sparsity information.

### 4. EnvGroup Sparse Architecture @ `envs/env_group.py`)

**New Class: `EnvGroupSparseRubric`**

Extends standard `EnvGroupRubric` with domain-specific sparse marking:

```python
class EnvGroupSparseRubric(EnvGroupRubric):
    async def score_rollout(self, ...):
        # Route to domain-specific environment
        env_results = await env.rubric.score_rollout(...)
        
        # Mark uncomputed metrics as sparse
        uncomputed_metrics = set(all_rewards) - set(env_results.metrics.keys())
        sparse_metrics = uncomputed_metrics if uncomputed_metrics else None
        
        return RolloutScore(sparse_metrics=sparse_metrics, ...)
```

**Activation Logic:**
```python
# Key decision point for sparse metrics
if enable_sparse_metrics:
    rubric = EnvGroupSparseRubric(self.env_map)  # Sparse-aware
else:
    rubric = EnvGroupRubric(self.env_map)       # Standard (backwards compatible)
```

automatically mark domain-specific metrics as sparse when irrelevant.

### 5. Sparse-Aware Display @ `utils/eval_utils.py`

**Selective Averaging:**
```python
# Filter out sparse values before averaging
sparse_flags = results.sparse_metrics[k]
relevant_values = [val for val, is_sparse in zip(v, sparse_flags) if not is_sparse]

if relevant_values:
    avg = sum(relevant_values) / len(relevant_values)
    sparsity_info = f" (relevant: {len(relevant_values)}/{len(v)})"
    print(f"{k}: avg - {avg:.3f}{sparsity_info}")
else:
    print(f"{k}: no relevant data (all values sparse)")
```

**Enhanced Display:**
```python
# Show "-" for sparse values instead of misleading 0.0
if sparse_flags[idx]:
    trials.append("-")        # Sparse (excluded from averaging)
else:
    trials.append(round(v[idx], 3))  # Actual computed value
```

provide mathematically correct averages and clear visual distinction of sparsity.

## Usage 

```python
# Standard behavior (backwards compatible)
env = vf.EnvGroup(envs, names)                           # Standard averaging

# Sparse metrics enabled
env = vf.EnvGroup(envs, names, enable_sparse_metrics=True)  # Selective averaging
```

```python
def load_environment(enable_sparse_metrics: bool = True):
    return vf.EnvGroup(
        envs=domain_envs,
        env_names=domain_names, 
        enable_sparse_metrics=enable_sparse_metrics
    )
```

## To Test:

To test sparse metrics with ProfBench:

1. **Pull the ProfBench environment changes:**
   ```bash
   git clone https://github.com/vxnuaj/prime-environments.git -b vxnuaj/profbench
   cd prime-environments
   ```

2. **Pull this verifiers PR with sparse metrics implementation**

3. **Install verifiers in editable mode:**
   ```bash
   cd verifiers
   uv pip install -e .
   ```

4. **Run evaluation to see sparse metrics in action:**
   ```bash
   vf-eval -s profbench -m gpt-4.1-mini --env-args '{"judge_model": "openai/gpt-4.1-mini"}' -n 12 -r 1 
   # -n must be >= 10 for sparsity to be detected, as if we do less, then profbench only loads from the first domain ( i believe physics or chemistry )
   # feel free to do -r x \in R^n
   ```

**Expected output:**
- Domain-specific averages (e.g., `chemistry_phd_reward: avg - 72.9 (relevant: 2/12)`)
- Sparse values displayed as `-` instead of `0.0`
- Mathematically correct averages excluding irrelevant domain scores


# math-gdpo-exp

### Overview
- **Environment ID**: `math_gdpo_exp`
- **Short description**: GSM8K environment with GDPO support for testing multi-reward optimization (correctness + length).
- **Tags**: math, gsm8k, gdpo, grpo, train

### Purpose
This environment is designed to test GDPO (Group reward-Decoupled Policy Optimization) from [arXiv:2601.05242](https://arxiv.org/abs/2601.05242). It uses grade-school math problems (GSM8K) which are suitable for smaller models like Qwen 2.5 1.5B, while providing:

- **Multi-reward setup**: correctness + length rewards
- **Natural tension**: longer reasoning helps accuracy, but shorter is rewarded when correct
- **GDPO gating**: length reward only counts when answer is correct

### Datasets
- **Primary dataset**: GSM8K (grade school math)
- **Train split**: ~7.5k examples
- **Test split**: ~1.3k examples

### Task
- **Type**: single-turn
- **Parser**: `Parser` with boxed answer extraction
- **System prompt**: Standard boxed answer format (`\boxed{answer}`)

### Rewards
| Reward | Description | GDPO Gating |
| ------ | ----------- | ----------- |
| `math_answer_reward_func` | 1.0 if answer matches, 0.0 otherwise | None (primary) |
| `length_reward_func` | 1.0 if response ≤ `length_threshold` (default 500), 0.0 otherwise | Gated on correctness |

### Quickstart

**GRPO baseline:**
```toml
[env]
id = "math_gdpo_exp"

[env.args]
advantage_mode = "grpo"
```

**GDPO comparison:**
```toml
[env]
id = "math_gdpo_exp"

[env.args]
advantage_mode = "gdpo"
```

### Environment Arguments
| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `advantage_mode` | `"grpo"` \| `"gdpo"` | `"grpo"` | Advantage computation mode |
| `num_train_examples` | int | -1 | Number of training examples (-1 for all) |
| `num_eval_examples` | int | -1 | Number of eval examples (-1 for all) |

### GDPO vs GRPO

**GRPO**: Normalizes the sum of rewards across the group.

**GDPO**:
1. Normalizes each reward separately (per-reward z-score)
2. Gates length on correctness (length only counts if answer is correct)
3. Sums normalized advantages
4. Applies batch-wise normalization

This prevents reward hacking where the model optimizes length while being wrong.

# MMMU model-scale A/B

## Question

Does replacing vision inputs with ASCII create useful within-prompt reward variance for RL,
rather than only lowering average reward?

## Setup

- Taskset: native-v1 `mmmu_v1`, MMMU dev split.
- Selection: the same 20 tasks in every arm, shuffled with the framework's fixed seed.
- Sampling: 5 independent rollouts per task (`100` rollouts per arm).
- Harness/runtime: `null` harness, subprocess runtime, concurrency `16`.
- Text arm: Textify ASCII, width `160`, fixed threshold `0.5`, auto inversion.
- Vision arm: Textify disabled.
- Decoding: temperature `0.7`, top-p `0.8`, top-k `20`, min-p `0`, presence
  penalty `1.5`, repetition penalty `1.0`, Qwen non-thinking mode.
- Reward: exact match on the final boxed multiple-choice letter.
- Infrastructure failures: provider errors were retried and excluded from reward totals.

The final 9B and 122B traces contain five completed, naturally stopped, boxed responses for
every prompt. The 0.8B model often repeated until the 32,768-token output cap; those are
reported separately below and remain model-level failures in the headline rewards.

## Metrics

A prompt is **mixed** when its five binary rewards contain both success and failure. Such a
group supplies non-zero relative advantages to group-based RL methods. Mean within-prompt
sample variance is computed over the five binary rewards per prompt and then averaged over
20 prompts (`ddof=1`).

| Model | Provider | Arm | Accuracy | Mixed prompts | Mean within-prompt variance | 0/5 | 5/5 | Capped outputs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | Prime | Vision | 47% | 12/20 | 0.145 | 4 | 4 | 13 |
| Qwen3.5-0.8B | Prime | ASCII | 32% | 16/20 | 0.175 | 3 | 1 | 17 |
| Qwen3.5-9B | Prime | Vision | 80% | 3/20 | 0.040 | 3 | 14 | 0 |
| Qwen3.5-9B | Prime | ASCII | 55% | 7/20 | 0.090 | 5 | 8 | 0 |
| Qwen3.5-122B-A10B | OpenRouter | Vision | 82% | 3/20 | 0.045 | 2 | 15 | 0 |
| Qwen3.5-122B-A10B | OpenRouter | ASCII | 53% | 4/20 | 0.055 | 7 | 9 | 0 |

### Representation effect

| Model | Accuracy delta | Variance change | Mixed-group change |
|---|---:|---:|---:|
| 0.8B | -15 pp | 0.145 → 0.175 | 12 → 16 |
| 9B | -25 pp | 0.040 → 0.090 | 3 → 7 |
| 122B-A10B | -29 pp | 0.045 → 0.055 | 3 → 4 |

Task-bootstrap 95% confidence intervals for ASCII minus vision accuracy were `[-27, -4]`
pp at 0.8B, `[-42, -9]` pp at 9B, and `[-47, -13]` pp at 122B-A10B.

## Solve-count distributions

### 0.8B

| Solved / 5 | Vision prompts | ASCII prompts |
|---:|---:|---:|
| 0 | 4 | 3 |
| 1 | 4 | 11 |
| 2 | 4 | 1 |
| 3 | 1 | 2 |
| 4 | 3 | 2 |
| 5 | 4 | 1 |

### 9B

| Solved / 5 | Vision prompts | ASCII prompts |
|---:|---:|---:|
| 0 | 3 | 5 |
| 1 | 0 | 2 |
| 2 | 0 | 3 |
| 3 | 2 | 1 |
| 4 | 1 | 1 |
| 5 | 14 | 8 |

### 122B-A10B

| Solved / 5 | Vision prompts | ASCII prompts |
|---:|---:|---:|
| 0 | 2 | 7 |
| 1 | 0 | 1 |
| 2 | 2 | 2 |
| 3 | 1 | 1 |
| 4 | 0 | 0 |
| 5 | 15 | 9 |

## Interpretation

ASCII increased mean within-prompt reward variance and the number of mixed groups at all
three scales, but for different reasons:

- **0.8B:** the policy itself is unstable. Both representations have many mixed groups, and
  severe repetition/output caps confound the interpretation.
- **9B:** the cleanest RL frontier in this sample. ASCII more than doubled mean within-prompt
  variance and mixed groups, while all trajectories completed cleanly.
- **122B-A10B:** mostly saturated. ASCII primarily moves prompts from stable success to stable
  failure, producing only a small increase in mixed groups.

This is evidence that a lossy representation can move a capable model away from a saturated
reward regime. It does not yet show that training on that variance improves the policy; a
training experiment is the next step.

## Provider and artifact caveats

- Prime's listed 122B-A10B deployment accepted text but returned upstream `500` for every
  image request, including a 2×2 white PNG at concurrency one. Both 122B arms therefore use
  OpenRouter's `qwen/qwen3.5-122b-a10b`; providers are not mixed within that comparison.
- Prime's 0.8B endpoint became available after initially returning `404`.
- Dashboard uploads returned `413` because trace task data retains large base64 images. Raw
  traces remain local and are not committed.

## Per-prompt results

### 0.8B

| Task | Vision | ASCII | Delta | Vision capped | ASCII capped |
|---|---:|---:|---:|---:|---:|
| `dev_Accounting_4` | 0/5 | 1/5 | +1 | 5 | 3 |
| `dev_Agriculture_2` | 1/5 | 0/5 | -1 | 0 | 3 |
| `dev_Art_1` | 1/5 | 1/5 | +0 | 0 | 0 |
| `dev_Art_Theory_1` | 4/5 | 2/5 | -2 | 0 | 0 |
| `dev_Basic_Medical_Science_2` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Basic_Medical_Science_4` | 1/5 | 1/5 | +0 | 0 | 0 |
| `dev_Computer_Science_1` | 2/5 | 3/5 | +1 | 0 | 1 |
| `dev_Computer_Science_3` | 0/5 | 1/5 | +1 | 0 | 3 |
| `dev_Computer_Science_4` | 1/5 | 1/5 | +0 | 0 | 0 |
| `dev_Computer_Science_5` | 5/5 | 4/5 | -1 | 0 | 1 |
| `dev_Design_5` | 5/5 | 4/5 | -1 | 0 | 1 |
| `dev_Literature_4` | 0/5 | 0/5 | +0 | 0 | 0 |
| `dev_Materials_3` | 3/5 | 1/5 | -2 | 0 | 0 |
| `dev_Mechanical_Engineering_4` | 0/5 | 1/5 | +1 | 4 | 0 |
| `dev_Music_1` | 2/5 | 1/5 | -1 | 1 | 1 |
| `dev_Pharmacy_2` | 4/5 | 3/5 | -1 | 0 | 0 |
| `dev_Pharmacy_3` | 2/5 | 0/5 | -2 | 0 | 0 |
| `dev_Pharmacy_5` | 5/5 | 1/5 | -4 | 0 | 1 |
| `dev_Physics_1` | 4/5 | 1/5 | -3 | 0 | 0 |
| `dev_Public_Health_1` | 2/5 | 1/5 | -1 | 3 | 3 |

### 9B

| Task | Vision | ASCII | Delta | Vision capped | ASCII capped |
|---|---:|---:|---:|---:|---:|
| `dev_Accounting_4` | 5/5 | 2/5 | -3 | — | — |
| `dev_Agriculture_2` | 0/5 | 0/5 | +0 | — | — |
| `dev_Art_1` | 5/5 | 5/5 | +0 | — | — |
| `dev_Art_Theory_1` | 4/5 | 4/5 | +0 | — | — |
| `dev_Basic_Medical_Science_2` | 5/5 | 5/5 | +0 | — | — |
| `dev_Basic_Medical_Science_4` | 5/5 | 5/5 | +0 | — | — |
| `dev_Computer_Science_1` | 5/5 | 5/5 | +0 | — | — |
| `dev_Computer_Science_3` | 3/5 | 1/5 | -2 | — | — |
| `dev_Computer_Science_4` | 0/5 | 0/5 | +0 | — | — |
| `dev_Computer_Science_5` | 5/5 | 5/5 | +0 | — | — |
| `dev_Design_5` | 5/5 | 5/5 | +0 | — | — |
| `dev_Literature_4` | 0/5 | 1/5 | +1 | — | — |
| `dev_Materials_3` | 5/5 | 3/5 | -2 | — | — |
| `dev_Mechanical_Engineering_4` | 5/5 | 5/5 | +0 | — | — |
| `dev_Music_1` | 3/5 | 2/5 | -1 | — | — |
| `dev_Pharmacy_2` | 5/5 | 5/5 | +0 | — | — |
| `dev_Pharmacy_3` | 5/5 | 2/5 | -3 | — | — |
| `dev_Pharmacy_5` | 5/5 | 0/5 | -5 | — | — |
| `dev_Physics_1` | 5/5 | 0/5 | -5 | — | — |
| `dev_Public_Health_1` | 5/5 | 0/5 | -5 | — | — |

### 122B-A10B

| Task | Vision | ASCII | Delta | Vision capped | ASCII capped |
|---|---:|---:|---:|---:|---:|
| `dev_Accounting_4` | 5/5 | 2/5 | -3 | 0 | 0 |
| `dev_Agriculture_2` | 0/5 | 0/5 | +0 | 0 | 0 |
| `dev_Art_1` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Art_Theory_1` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Basic_Medical_Science_2` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Basic_Medical_Science_4` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Computer_Science_1` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Computer_Science_3` | 5/5 | 0/5 | -5 | 0 | 0 |
| `dev_Computer_Science_4` | 0/5 | 0/5 | +0 | 0 | 0 |
| `dev_Computer_Science_5` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Design_5` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Literature_4` | 3/5 | 3/5 | +0 | 0 | 0 |
| `dev_Materials_3` | 2/5 | 0/5 | -2 | 0 | 0 |
| `dev_Mechanical_Engineering_4` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Music_1` | 2/5 | 0/5 | -2 | 0 | 0 |
| `dev_Pharmacy_2` | 5/5 | 5/5 | +0 | 0 | 0 |
| `dev_Pharmacy_3` | 5/5 | 2/5 | -3 | 0 | 0 |
| `dev_Pharmacy_5` | 5/5 | 0/5 | -5 | 0 | 0 |
| `dev_Physics_1` | 5/5 | 1/5 | -4 | 0 | 0 |
| `dev_Public_Health_1` | 5/5 | 0/5 | -5 | 0 | 0 |

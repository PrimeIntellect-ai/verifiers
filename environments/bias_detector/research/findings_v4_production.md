# Bias Detector Findings v4: Production Deployment Focus

## Critical Discovery: Oracle Dependency in Iterative Training

### The Problem
V31 (iterative oracle correction) appeared to be our best model with MAE=0.392. But this metric includes oracle corrections that require ground truth — which is unavailable in production.

### Analysis Results (from v31_first_pass_analysis.py)

| Metric | MAE | Notes |
|--------|-----|-------|
| V31 Final (with oracle) | 0.489 | What wandb shows |
| V31 First-Pass (no oracle) | **0.512** | Actual production performance |
| V25 Baseline (no oracle) | 0.632 | Non-iterative baseline |
| V37 Best Combo | 0.447 | With oracle; first-pass TBD |

**V31 first-pass IS 19% better than V25** — iterative training does bake in some improvement.

### The Concerning Trend
- Correction rate INCREASES over training: 24.4% → 38.2%
- Model becomes MORE dependent on oracle, not less
- The oracle correction only adds 4.5% improvement (0.512 → 0.489)

### Implication
For production deployment without ground truth, the gap between training MAE and production MAE is significant but not catastrophic. The iterative training does improve internal scoring ability.

## Production Strategy

### Current Best Production Models (Training MAE, actual production TBD)
1. **V31** — MAE=0.392 (training), est. 0.512 (production)
2. **V37** — MAE=0.447 (training), est. similar gap
3. **V25** — MAE=0.581 (production = training, no oracle)

### Experiments Designed to Optimize Production
| Version | Strategy | Key Feature |
|---------|----------|------------|
| V89 | Iterative + AccReward | Max gradient signal, deploy single-pass |
| V90 | Blind Self-Correction | NO oracle needed at inference |
| V91 | Devil's Advocate | Self-adversarial reasoning, no oracle |
| V92 | Blind + Oracle | Train both, deploy blind-only |
| V93 | Devil's + Oracle | Train both, deploy devil's-only |
| **V94** | **First-Pass Bonus** | Reward first-pass accuracy explicitly |
| **V95** | **Oracle Withdrawal** | 50% oracle withheld during training |
| **V96** | **Bonus + Withdrawal** | Combined approach |
| **V97** | **Blind + Devil's** | FULLY production viable, no oracle |
| **V98** | **AccReward + Devil's** | Better gradients, no oracle |
| **V99** | **Ultimate Production** | All production-viable strategies combined |
| V100 | Evidence-First | Extract evidence → score |
| V101 | Pairwise Ranking | Reference article comparison |
| V102 | Consistency Training | Score twice, penalize inconsistency |

### New Tracking Metrics
Added to environment rubric:
- `first_pass_mae` — MAE of INITIAL prediction (before any correction)
- `correction_delta` — How much corrections change the scores

These allow us to directly compare first-pass and final accuracy during training.

## Architecture Changes Made

### Environment (`bias_detector.py`)
1. **`disc_first_scores`** now captured for ALL runs (not just iterative)
2. **`disc_first_pass_bonus`** — advantage bonus for first-pass accuracy
3. **`disc_oracle_withdrawal`** — randomly skip oracle correction to force first-pass learning
4. **`disc_evidence_first`** — extract evidence spans before scoring
5. **`disc_pairwise_rank`** — reference article comparison
6. **`disc_consistency_training`** — score twice, consistency bonus

### Key Insight: First-Pass Bonus Design
```
disc_adv += disc_first_pass_bonus * (first_accuracy - advantage_center)
```
This gives the first discriminator turn its OWN advantage signal, separate from the final correction.
With `disc_first_pass_bonus=1.0`, first-pass and final accuracy are weighted equally.

### Key Insight: Oracle Withdrawal Design
```
if random.random() < disc_oracle_withdrawal:
    max_revisions = 0  # Skip oracle this rollout
```
With `disc_oracle_withdrawal=0.5`, half the rollouts get no oracle feedback.
The model must learn to score well on its own because it can't predict which rollouts get oracle.

## Next Steps
1. Run production eval (vLLM + eval_production.py) when GPU becomes available
2. Analyze V94-V102 results as they complete
3. Design V103+ based on results
4. Build ensemble of best production models

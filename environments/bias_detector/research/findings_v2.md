# Bias Detector Research Findings V2

## Summary of Methods & Results (49 articles, 5 hard cases)

### Best Overall MAE (all articles)
| Rank | Method | MAE | Notes |
|------|--------|-----|-------|
| 1 | Cross-model cal (Base+V22+V23) | 0.79 | Median ensemble across 3 models |
| 2 | V22_s100 + calibrated | 0.84-0.96 | Best single model (varies by run) |
| 3 | Within-model all_methods ensemble | 0.91 | 7 methods combined per model |
| 4 | cal+percrit+antiprior ensemble | 0.82-0.97 | 3 methods, median vote |
| 5 | V20_s100 + calibrated | 0.93 | Good individual model |

### Best Hard5 MAE (articles 3, 7, 10, 15, 18)
| Rank | Method | Hard5 MAE | Notes |
|------|--------|-----------|-------|
| 1 | V23 + two_pass | 0.73 | Self-calibration fix. Overall MAE 1.48! |
| 2 | V22 + cal+two_pass ensemble | 0.93-1.00 | Two-pass saves hard cases |
| 3 | Base + per_crit_concise | 1.00 | Art. 18 & 3: PERFECT scores |
| 4 | Base + antiprior | 1.07 | Anti-prior rules work on base |
| 5 | cal+percrit+antiprior ensemble | 1.20 | Consistent across all models |

## Key Finding: The Accuracy-Robustness Trade-off

Methods that optimize overall accuracy (calibrated, cross-model ensemble) fail on
hard cases because they've learned topic/source priors that happen to be correct
for most articles. Methods that fix hard cases (two_pass, per_criterion) remove
those priors but lose the signal that was helping on easy cases.

## Root Cause: Three Model Failure Modes

### 1. Topic Prior (affects C1, C2)
Model sees "immigration" or "climate" → assumes biased framing.
- Article 3 (BBC immigration): Neutral wire-service style, model predicts -3,-3
- Article 18 (BBC wind/Trump): Neutral reporting, model predicts -3,-3

### 2. Quote/Voice Conflation (affects C2)
Model counts inflammatory quotes from officials as the article's rhetorical heat.
- Article 10 (ABC gender care): Neutral reporting of heated debate, model sees heat

### 3. Source Prior (affects C1)
Model uses publication name as shortcut.
- Article 7 (Vox economists): Actually pro-market frame (C1=+2), model assumes Vox = left
- Article 15 (NYT Philadelphia): Structural frame (C1=-2), model gets it backwards

## What Fixes Each Failure Mode

| Fix | Topic Prior | Quote Conflation | Source Prior |
|-----|------------|-----------------|-------------|
| Per-criterion scoring | Strong | Moderate | Weak |
| Anti-prior rules | Moderate | Strong | Strong |
| Two-pass self-calibration | Strong | Strong | Strong |
| Quote stripping | None | Moderate | None |
| Contrastive examples | Weak | None | Moderate |

## Training Experiments

| Version | Prompt | Data | Best MAE | Best Hard5 |
|---------|--------|------|----------|------------|
| V17 | Standard | Real only | 0.78* | - |
| V20 | Standard | 90% synth | 0.76* | - |
| V22 | Calibrated | Real only | 0.84-0.96 | 1.80 |
| V23 | Calibrated | Corpus+real | 0.94-1.05 | 2.13-2.93 |
| V24 | **Anti-prior** | Corpus+real | TBD | TBD |

*Previous session results, slightly different eval setup

## Next Experiments

### V24 Anti-Prior Training (RUNNING)
Train with anti-prior discriminator prompt to internalize topic≠bias, quote≠voice rules.
Hypothesis: Model trained with anti-prior will show lower Hard5 MAE while maintaining
competitive overall MAE. The anti-prior rules won't be "overhead" — they'll be default behavior.

### Planned: V25 Per-Criterion Training
Train discriminator to predict one criterion at a time (3 turns per article).
This would require env restructuring but directly trains the skill that makes
per_crit_concise work at eval time.

### Planned: Self-Consistency at Training Time
Instead of single prediction, generate 3-5 predictions per article at temp>0
and reward the median. This trains the model to produce stable, well-calibrated
predictions.

## Ensemble Strategy for Production

Best combined approach (theory):
1. Get calibrated scores from best model (V22 or V24)
2. Get per_criterion scores for C1 and C2 only (fixes topic/quote priors)
3. Get two_pass scores as a "sanity check"
4. Median vote across all three
Expected: ~0.85 overall MAE with ~1.0 Hard5 MAE (balanced)

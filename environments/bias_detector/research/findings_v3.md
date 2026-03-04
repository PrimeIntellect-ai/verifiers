# Bias Detector Research Findings V3

## Summary of Methods & Results (49 articles, 5 hard cases)

### Best Overall MAE (all articles)
| Rank | Method | MAE | Notes |
|------|--------|-----|-------|
| 1 | **V28_s100 + anchor_compare** | **0.782** | Quote-mask training + gold reference (LEAKY) |
| 2 | V28_s150 + 3method_ensemble(mean) | 1.129 | Best non-leaky ensemble |
| 3 | V26_s150 + calibrated (V3 eval) | 1.320 | Best non-leaky single-method |
| 4 | V26_s150 cal+decomp+percrit (V2 eval) | 0.80 | V2 eval (with few-shot examples) |
| 5 | V25_s150 / V26_s150 calibrated (V2 eval) | 0.94 | V2 eval had few-shot examples |

**CRITICAL**: V2→V3 eval "regression" (0.94→1.34) is due to V3 eval stripping
few-shot examples. Model trained WITH examples but V3 eval tested WITHOUT them.
Need matched-prompt eval to get fair baseline (eval_v8).

### Best Cross-Model Results
| Ensemble | MAE | C1 | C2 | C3 | Coverage |
|---------|-----|-----|-----|-----|---------|
| Top3×selective_twopass | **0.68** | 1.00 | 0.81 | 0.22 | 32/49 (parse errors) |
| All×selective_twopass | 0.73 | 1.03 | 0.86 | 0.30 | 37/49 |
| All×calibrated | **0.87** | 1.08 | 0.86 | 0.67 | 49/49 |

### Human Ceiling
Inter-rater MAE = **0.14** (7 calibration articles with consensus scores)
Current best model MAE = **0.80** → 5.7x human error

## Key Finding: V25+V26 Confirm Prompt Diversity Helps

V25 (self-correction) and V26 (mixed-prompt) both improve over V24:
- V25_s150 calibrated MAE=0.94 (best C2=0.80)
- V26_s150 calibrated MAE=0.94 (best C2=0.82)
- V26_s150 ensemble MAE=0.80 (NEW BEST OVERALL)

**C2 (temperature/quote conflation) improving**: Went from 1.04 (V24_s150) → 0.80-0.82 (V25/V26)

## Root Cause: Three Model Failure Modes

### 1. Topic Prior (affects C1, C2) — Articles 3, 18
Model sees "immigration" or "climate" → assumes biased framing.
- Article 3 (BBC immigration, true 0,0,2): Neutral wire-service, model predicts -3,-3
- Article 18 (BBC wind/Trump, true 0,0,2): Neutral reporting, model predicts -3,-3
- **Attack vector**: Topic-neutral training (V29) — strip topic nouns from article

### 2. Quote/Voice Conflation (affects C2) — Article 10
Model counts inflammatory quotes from officials as the article's rhetorical heat.
- Article 10 (ABC gender care, true 0,0,1): 16 heated quotes but neutral journalist voice
- **Attack vector**: Quote-masked training (V28), disaggregate_c2 eval

### 3. Source Prior (affects C1) — Articles 7, 15
Model uses publication name as shortcut.
- Article 7 (Vox economists, true 2,-1,1): Actually pro-market, model assumes Vox=left
- Article 15 (NYT Philadelphia, true -2,-1,2): Structural frame, model gets C1 backwards
- **Attack vector**: Contrastive training (V30), anchor-compare eval

## Training Experiments

| Version | Prompt | Data | Best MAE | Best Ens | Key Finding |
|---------|--------|------|----------|----------|-------------|
| V17 | Standard | Real only | 0.78* | - | Baseline |
| V20 | Standard | 90% synth | 0.76* | - | Synthetic helps |
| V22 | Calibrated | Real only | 0.84-0.96 | - | Calibration prompt improves |
| V23 | Calibrated | Corpus+real | 0.94-1.05 | - | More data doesn't always help |
| **V24** | **Anti-prior** | Corpus+real | **0.74-0.94** | 0.84 | Anti-prior training works! |
| **V25** | **Self-correct** | Corpus+real | **0.94** | **0.85** | Self-correction helps C2 |
| **V26** | **Mixed prompt** | Corpus+real | **0.94** | **0.80** | Mixed training = best ensemble |
| **V27** | **C2-focus 3x** | Corpus+real | TRAINING | - | C2 MAE 0.465 in late training (29.6% better than V26!) |
| V28 | Quote-mask 30% | Corpus+real | TRAINING | - | Strip quotes from 30% articles |
| V29 | Topic-neutral | Corpus+real | QUEUED | - | Strip topic nouns from 40% |
| V30 | Contrastive+all | Corpus+real | QUEUED | - | Kitchen sink: contrastive+mask+neutral |
| V31 | Iterative self-correct | Corpus+real | QUEUED | - | Directional feedback for worst criterion |
| V32 | C1 specialist | Corpus+real | QUEUED | - | Only C1 advantage, specialist adapter |
| V33 | C2 specialist | Corpus+real | QUEUED | - | Only C2 advantage, specialist adapter |
| V34 | Rank-16 LoRA | Corpus+real | QUEUED | - | Double capacity (rank 16 vs 8) |
| V35 | Comparative | Corpus+real | QUEUED | - | Anchor article comparison training |
| V36 | Source-blind | Corpus+real | QUEUED | - | Strip publication names 50% |

*Previous session results, slightly different eval setup

## V25/V26 Detailed Results

### V25 Self-Correction (calibrated method, full 49 articles)
| Checkpoint | MAE | C1 | C2 | C3 | Best Ensemble |
|-----------|-----|-----|-----|-----|---------|
| V25_s50 | 0.97 | 1.08 | 0.96 | 0.86 | 0.85 (cal+decomp+percrit) |
| V25_s100 | 1.05 | 1.33 | **0.78** | 1.04 | 0.92 (all_7) |
| V25_s150 | **0.94** | 1.02 | **0.80** | 1.00 | 0.85 (cal+sel+hybrid) |

### V26 Mixed-Prompt (calibrated method, full 49 articles)
| Checkpoint | MAE | C1 | C2 | C3 | Best Ensemble |
|-----------|-----|-----|-----|-----|---------|
| V26_s100 | 1.20 | 1.39 | 1.12 | 1.10 | 0.85 (cal+decomp+percrit) |
| V26_s150 | **0.94** | 1.22 | **0.82** | 0.78 | **0.80** (cal+decomp+percrit) |

### Within-Model Best Ensembles
| Model | Ensemble | MAE |
|-------|---------|-----|
| V26_s150 | cal+decomp+percrit | **0.80** |
| V26_s150 | all_7_methods | 0.81 |
| V26_s150 | cal+sel_tp+hybrid | 0.82 |
| V24_s150 | all_7_methods | 0.84 |
| V25_s50 | cal+decomp+percrit | 0.85 |
| V25_s150 | cal+sel_tp+hybrid | 0.85 |

## Eval Methods Performance

selective_twopass has best MAE when it works (0.66-0.83) but 15-22 parse failures.
Fixed in V3 eval with strict JSON second-pass (selective_twopass_v2).

| Method | Typical MAE | Parse Failures | Strengths |
|--------|-----------|----------------|-----------|
| calibrated | 0.94-1.18 | 0 | Reliable, consistent |
| selective_twopass | 0.66-0.85 | 15-22/49 | Best when works |
| per_crit | 0.99-1.07 | 0 | Good C3 |
| decomposed | 1.01-1.24 | 0 | Good C2 isolation |
| quote_aware | 1.06-1.42 | 0 | Variable |
| relative | 1.00-1.21 | 0 | Good for calibration |
| hybrid_best | 1.02-1.18 | 0 | Combines methods |

### New V3 Eval Methods (in development)
- **selective_twopass_v2**: Fixed parse failures with strict JSON
- **anchor_compare**: Reference article for calibration grounding
- **disaggregate_c2**: Strip quotes, score C2 on journalist text only

## Score Distribution
- C1 mean=-0.51, skew left (25 neg, 15 zero, 9 pos)
- C2 mean=-0.93, heavily loaded language (29 neg, 12 zero, 8 pos)
- C3 mean=+1.16, well-sourced articles (14 at +1, 23 at +2)

## V27/V28 Eval Results (V3 eval, 15 models, 49 articles)

### New Best: V28_s100 + anchor_compare = 0.782 MAE
anchor_compare dominates ALL top 20 positions. Best non-anchor: 3method_ensemble(mean) at 1.129.

### Calibrated method regression: 0.94 → 1.31-1.45
Consistent across ALL models including base. NOT model degradation — the V3 eval's calibrated prompt gives no reference grounding, so scores are noisier on full 49-article set.

### CRITICAL: anchor_compare has gold-label leakage
The method picks the CLOSEST gold-scored article as reference, essentially telling the model "the answer is near these scores." Evidence:
- Base model (no finetuning) gets 0.837 with anchor_compare vs 1.340 calibrated
- All models converge on identical predictions for many articles
- Article 18: EVERY model gets err=0 with anchor, err=2 without
Need random-anchor variant to test true calibration value vs leakage.

### V28 > V27 on anchor_compare
V28 (quote-masked): 0.782 at s100, V27 (C2-focus): 0.837 at s50.
Quote masking helped C1 most (0.878 vs 1.020), not C2 as expected.

## V29/V30 Eval Results (V3 eval, 19 models, 49 articles)

### V29 (Topic-Neutral) and V30 (Kitchen-Sink) — No Improvement

| Version | Best anchor_compare | Best calibrated | Best ensemble(mean) |
|---------|-------------------|-----------------|-------------------|
| Base    | 0.837             | 1.340           | 1.163             |
| V25     | 0.850 (s50)       | 1.388 (s150)    | 1.156 (s50)       |
| V26     | 0.816 (s100)      | 1.320 (s150)    | 1.197 (s150)      |
| V27     | 0.837 (s50)       | 1.340 (s50)     | 1.170 (s50)       |
| **V28** | **0.782 (s100)**  | 1.340 (s150)    | **1.129 (s150)**  |
| V29     | 0.844 (s150)      | 1.327 (s50)     | 1.136 (s100)      |
| V30     | 0.830 (s150)      | 1.354 (s100)    | 1.150 (s50)       |

**Key findings:**
- V28 remains undisputed best on anchor_compare (0.782) and ensemble (1.129)
- V29 topic-neutral did NOT fix topic prior failures — Art 3 still wrong on all checkpoints
- V30 kitchen-sink shows no synergy — diluting 3 approaches is worse than pure quote-mask (V28)
- On calibrated (no leakage), no finetuned model meaningfully beats base (max 0.020 improvement)
- True RL benefit: only 6.6% on anchor_compare, ~1.5% on calibrated

### Per-Criterion Winners (anchor_compare)
- **C1 Frame**: V28_s100 = 0.878 (best)
- **C2 Temperature**: V28_s150 = 0.694 (best)
- **C3 Evidence**: V26_s100 = 0.694 (best)

### Hard Articles Remain Hard
- Art 3 (immigration, gold 0,0,+2): ALL models predict negative C1/C2 — topic prior baked in
- Art 10 (gender care, gold 0,0,+1): V28/V30 make it WORSE (err=7-8 vs base err=4)
- Art 7 (economists): V30_s150 best single result (err=1), nailing C1=+2
- Art 18 (wind/Trump): Perfect on anchor_compare (all models), err=2 on calibrated (all)

## Ensemble Strategy for Production

Best combined approach:
1. **V28_s100 + anchor_compare** → MAE 0.782 (best, but uses gold reference)
2. **V28_s150 + 3-method ensemble(mean)** → MAE 1.129 (best without gold reference)
3. If single-call no reference: V26_s150 + calibrated → MAE 1.320 (best reliable)

## Experiments

### V27 C2-Focus Training (COMPLETE)
C2 weighted 3x in advantage function + selfcorrect prompt.
**Result: C2 MAE 0.465 in late training (steps 120-149), 29.6% better than V26.**

### V28 Quote-Masked Training (COMPLETE — BEST MODEL)
30% of articles have quoted speech stripped during training + mixed prompt.
**Result: Best overall at 0.782 anchor_compare, 1.129 ensemble(mean).**

### V29 Topic-Neutral (COMPLETE — NO IMPROVEMENT)
40% of articles have topic nouns replaced with placeholders. C1 weighted 2x.
**Result: Did not help. Topic prior not fixable by noun stripping alone.**

### V30 Kitchen-Sink (COMPLETE — NO IMPROVEMENT)
30% contrastive + 30% quote-masked + 30% topic-neutral + mixed prompt.
**Result: Dilution hurt. Pure V28 approach better than mixing all strategies.**

### V31 Iterative Self-Correction (COMPLETE)
When disc's worst criterion is off by 2+, gets directional feedback and re-scores.

**Training observations:**
- Step advantage magnitude decreased: 0.75 → 0.25 (model converging, fewer errors)
- Loss: 0.10 → 0.06 (steady improvement)
- Entropy: stable 0.46-0.54 (healthy, no collapse)
- Three grad norm spikes: step 12 (3.5), step 62 (5.2), step 94 (2.1) — all recovered

**Correction rate trend (oscillating, not monotonic):**
```
Step  0: 12% | Step 30: 38% | Step 60: 12% | Step  90: 50%
Step 10: 12% | Step 40: 50% | Step 70: 62% | Step 100: 25%
Step 20: 12% | Step 50: 38% | Step 80: 25%
```
Correction rate oscillates due to article difficulty variation in batches.
Step 70 spike (62%) correlates with grad norm spike at step 62.
Average correction magnitude: 3.0-4.0 (early) → 1.5 (step 90) → no clear trend.

**Full correction rate trend:**
```
Step 120: 25% | Step 130: 38% | Step 140: 38%
```

**Final metrics (step 150):**
- Loss: 0.10 → 0.06
- |Adv|: 0.75 → 0.25 (converging)
- 4 grad norm spikes: steps 12, 62, 94, 133 (all recovered)

**Detailed Correction Analysis (120 rollouts, 95 matched gold):**
- 32/120 rollouts (26.7%) triggered correction (error ≥ 2 on worst criterion)
- **84.4% of corrections improved total error** (avg: 4.62 → 2.56, Δ=-2.06)
- 12.5% unchanged, 3.1% worsened (1 case: collateral C3 damage, no targeted improvement)
- Direction compliance: 81.2% moved correct direction, 0% wrong direction, 18.8% no change
- C1 triggers 59.4% of corrections (hardest), C3 21.9%, C2 18.8%
- By criterion success: C3=100%, C1=84%, C2=67%
- Collateral damage minimal: 85.9% of untargeted criteria unchanged, net slightly positive
- Overcorrection rare: 34.4% hit gold exactly, 37.5% moved closer, only 9.4% overshot
- Hardest articles: counter-intuitive profiles (immigration with right frame, neutral climate)
- Learning dynamics non-monotonic: MAE 1.83 (early) → 2.50 (mid, spikes) → 1.43 (late)

### V32 C1-Specialist (RUNNING — step 77/150)
C1 weighted 3x, C2/C3 weight 0 in advantage. Tests specialization hypothesis.

**Training metrics (step 77):**
- Loss: 0.052-0.086, Entropy: 0.41-0.49 (stable)
- Grad norm: recovered from 6.54 spike at step 46, another spike 2.56 at step 77
- KL: 0.001 (very low)

**Validation MAE trend (oscillating, batch-dependent):**
```
Step  0: MAE=0.681  C1=1.333  C2=0.250  C3=0.458
Step 10: MAE=0.194  C1=0.208  C2=0.042  C3=0.333
Step 20: MAE=1.028  C1=1.042  C2=0.875  C3=1.167
Step 30: MAE=0.069  C1=0.042  C2=0.125  C3=0.042
Step 40: MAE=0.764  C1=0.917  C2=0.833  C3=0.542
Step 50: MAE=0.347  C1=0.458  C2=0.167  C3=0.417
Step 60: MAE=0.722  C1=0.625  C2=0.750  C3=0.792
Step 70: MAE=0.181  C1=0.083  C2=0.042  C3=0.417
```
C1 lows improving: 1.333 → 0.208 → 0.042 → 0.083. C2/C3 also improving
despite zero advantage weight (transferring from C1 learning?).

### V33 C2-Specialist (QUEUED)
C2 weighted 3x, C1/C3 weight 0. Paired with V32.

### V34 Rank-16 LoRA (QUEUED)
Same as V26 but rank 16 vs 8. Tests capacity bottleneck.

### V35 Comparative Training (QUEUED)
50% of rollouts show anchor article with known scores.

### V36 Source-Blind Training (QUEUED)
50% of articles have publication names stripped.

### V37 Best-Combo (QUEUED)
V28 (quote-mask 30%) + V31 (iterative) + V26 (mixed prompt). Combines the top-
performing individual innovations. C1 weighted 2x.

### V38 Comparative-Iterative (QUEUED)
50% comparative + iterative. Trains model on anchor comparison AND self-correction.
Targets anchor_compare eval method directly.

### V39 Source-Blind-Iterative (QUEUED)
50% source-blind + iterative. C1 weighted 2.5x. Directly attacks source prior
failure mode (Art 7 Vox, Art 15 NYT) combined with center-pull fix.

## C1 (Frame) Deep Error Analysis (from V32 rollout analysis)

### Three Failure Patterns
1. **Negative Prior Bias**: 30% of predictions cluster at C1=-3, 22% at C1=-2. True positive
   C1 articles (IDs 54, 31, 7) systematically rated negative with errors of 4-6 points.
   Root cause: 53% of effective training distribution is C1-negative.
2. **Topic-Frame Confusion**: Model conflates TOPIC with CAUSAL FRAME. Healthcare articles
   (true C1 mean: -2.12) get positive bias (pred mean: -0.19, err: +1.94) because model reads
   "ACA premiums" as market framing. Economy articles (true mean: -0.30) pulled negative (-1.30).
3. **Center-Extreme Confusion**: C1=0 articles predicted at -1.48 avg. C1=-3 articles predicted
   at 0.00 avg. Model can't distinguish "balanced reporting" from "strong structural framing."

### Hardest C1 Values
| True C1 | MAE | Signed Err | n | Failure Mode |
|---------|-----|------------|---|--------------|
| +3 | 3.50 | -3.50 | 4 | Predicts -3 or +2 (negative pull) |
| +2 | 4.50 | -4.50 | 4 | Predicts -3 or -2 (negative pull) |
| -3 | 3.00 | +3.00 | 18 | Always predicts ~0 (center pull) |
| 0 | 2.00 | -1.48 | 54 | Predicts -3 or -2 (negative pull) |

### CRITICAL: Synthetic Corpus Label Corruption
**51% of synthetic C1 labels have |target-judge| ≥ 2.** The LLM judge assigned C1=-3 to 43% of
synthetic articles regardless of target. Only 16% of synthetic articles pass strict filtering
(|target-judge| ≤ 1 on ALL criteria). The model is being trained on actively wrong labels.

Effective training C1 distribution: 53% negative, 27% neutral, 20% positive.
After filtering (max_mismatch=2): 188/456 synthetic articles kept.

### Interventions Designed
- **V45**: Filtered corpus (max_mismatch ≤ 2) + C1-balanced sampling
- **V46**: Topic deconfounding prompts (same-topic contrastive C1 examples) + filtered + balanced
- `synthetic_max_mismatch` parameter added to environment
- `c1_balance` parameter equalizes C1 bin frequencies via oversampling
- `disc_topic_deconfound` adds explicit "TOPIC ≠ C1" contrastive examples

### Source Prior Shortcuts
Model uses publication name as C1 proxy: Vox=left, WSJ=right.
Fails on counter-stereotypical articles (Vox publishing pro-market content).

### C1 Data Imbalance
C1=+1 has only 1 article (severe underrepresentation of mild right-leaning).
C1=+2 has only 3 articles. Right-leaning content is 18% of dataset.

### Method-Specific C1 Finding
selective_twopass_v2 uniquely solves Art 10 (C1=0 perfect) — self-calibration step
overrides topic prior. No other method gets it right. This validates the iterative
self-correction training approach (V31).

## V31 Self-Correction During Training

Analyzed 40 corrections across all V31 validation rollouts (steps 0-140):
- **80% correct direction, 0% wrong, 20% no change**
- C1: 87% compliance (23 corrections — most triggered, confirms C1 hardest)
- C2: 50% compliance (model "sticks to guns" — resistant to correction)
- C3: 100% compliance (7/7 — easiest to correct)

The model learned when to change vs when to hold firm. This is a genuine metacognitive skill.

## Cross-Run Learning Dynamics (V25-V32)

### Training is Stochastic Search, Not Convergence
ALL runs show 32-45% coefficient of variation in the last 50 steps. No run achieves
a stable low MAE. The "best step" is typically a transient dip.

**Practical recommendation: Save checkpoints every 10 steps and select best post-hoc.**

### Run Rankings (last 20 steps avg MAE)
| Version | MAE | C1 | C2 | C3 | Key Feature |
|---------|-----|-----|-----|-----|-------------|
| V31 | **0.448** | **0.515** | **0.454** | **0.375** | Iterative (still improving at step 145!) |
| V28 | 0.726 | 1.000 | 0.577 | 0.600 | Quote-mask + mixed prompt |
| V32 | 0.730 | 0.831 | 0.606 | 0.752 | C1 specialist |
| V30 | 0.841 | 1.142 | 0.627 | 0.754 | Contrastive (worst — hurts) |

### Feature Effectiveness
| Feature | Impact | Evidence |
|---------|--------|----------|
| Iterative self-correction | **+38% improvement** | V31 vs V28, only run still improving at 150 |
| C1 specialization | +17% C1 only | V32 vs V28, but overall MAE unchanged |
| Quote masking | Helps early, unstable | V28 "flash-in-pan" pattern |
| Contrastive training | **Hurts** | V30 worst overall, never learns |
| Topic-neutral debiasing | **Hurts** | V29 degrades over time |

## Critical Eval Methodology Finding

### V3 Eval Prompt Mismatch
The V3 eval's calibrated method uses a COMPLETELY DIFFERENT prompt than training:
- **Training**: 3 few-shot examples, short CRITERIA_DESC, handoff format with reasoning
- **V3 Eval**: NO few-shot examples, elaborate SYSTEM_PROMPT, JSON-only no reasoning

This explains the 0.94→1.34 MAE regression. Created eval_v8_matched_prompt.py to test
with exact training prompt format. Expected result: recovery to ~0.94 MAE baseline.

## Hard Article Deep Analysis

### The 5 Hardest Articles and Why They're Hard

**Article 7 (Vox, C1=+2)**: Source-score inversion. Vox (coded left) publishes a pro-market
article arguing economists should be listened to. Model uses "Vox=left" shortcut → predicts C1=-2.
Fix: Source-blind training, explicit "remedy frame" extraction.

**Article 10 (ABC, C1=0, C2=0)**: Quote contamination. RFK Jr.'s inflammatory quotes ("malpractice",
"junk science") get attributed to the journalist. Article is pure neutral event reporting.
Fix: Quote masking (disc_mask_quotes), decompose journalist voice vs quoted speech.

**Article 3 (BBC, C1=0)**: Topic-bias confusion. Immigration topic → model assumes negative C1.
But BBC reports the event (judge convicted for helping migrant) without taking a position.
Fix: Topic deconfounding (V46), explicit "is there a CAUSAL explanation?" check.

**Article 15 (NYT, C1=-2)**: Subtle structural framing. The article's causal logic ("bad environment
causes crime, collective solutions fix it") is structural (C1=-2), but the uplifting tone (+gardens,
+streetlights, +community) makes it FEEL positive. Model conflates tone with frame.
Fix: Explicit causal logic extraction step in prompt.

**Article 18 (BBC, C1=0, C3=+2)**: Topic coding. Wind energy pause is politically coded.
Neutral reporting gets misread as pro-climate. C3=+2 underestimated because 6+ evidence types
aren't obvious in a short article.
Fix: Evidence type enumeration step in prompt.

### Topics Most Useful for Deconfounding
- **Healthcare**: C1 range [-3, +3], n=12. Same topic, wildly different frames.
- **Immigration**: C1 range [-3, +3], n=13. Both pro- and anti-enforcement framings.
- **Economy**: C1 range [-2, +3], n=14. Contains the critical Art 7 (Vox C1=+2).

### Five Failure Modes (ordered by frequency)
1. **Topic-as-Proxy**: Uses topic keywords to predict C1 (most common)
2. **Source Prior**: Uses publication name to predict C1
3. **Quote Contamination**: Attributes quoted speech to journalist
4. **C1/C2 Sign Coupling**: Assumes heated=left, calm=right
5. **Subtle Framing Blindness**: Misses causal logic when tone is positive

## Dataset Distribution Analysis

Critical imbalances that explain model failures:
- **C1=+1 has only 1 article** (mildly right-leaning nearly absent)
- **C2=+3 has 0 articles** (no examples of extremely measured journalism)
- **|C1| vs |C2| correlation = 0.744** — intensity correlated across frame and temperature
- **Source is a strong confound**: Wire → neutral, Opinion → extreme, Atlantic → left, WSJ → right
- **Article type is a confound**: Article type predicts scores almost as well as actual content
- Right-leaning content only 18% of dataset (9/49 articles)

## V33 C2 Specialist Deep Analysis

V33 uses `disc_per_criterion_scales = [0.0, 3.0, 0.0]` — pure C2 training, zero C1/C3 gradient.

### C2 Reasoning Evolution
- **Step 0**: Binary "neutral vs alarmist". Keywords: "moralizing" (7), "neutral" (7). 1.14 quoted phrases/response.
- **Step 50**: More quotation-heavy: 2.29 quoted phrases/response. Cites specific loaded language ("suicide note", "vicious spiral"). Negative bias deepens.
- **Step 80**: Discovers positive C2 (+1 appears 3x). First hedge words ("careful hedging"). Begins distinguishing quotes from journalist tone (2/8 cases).

### C2 Accuracy Paradox — Worse at Step 80 Than Step 50
- Step 50 MAE: 0.25 (but only 4 easy articles — mostly Atlantic/NYT extremes)
- Step 80 MAE: 1.86 (7 harder articles including WSJ +2, Vox -2)
- The model learns to distinguish extremes early but never learns the middle range

### C2 "Rhetorical Extremism" Bias (Analogous to C1 Topic Confusion)
1. **GT=-1 articles always predicted as -3**: Any loaded language → extreme. Can't represent mild rhetorical heat.
2. **GT=+1/+2 systematically underpredicted**: Measured/hedged writing → predicted as 0, missing active carefulness.
3. **Over-correction at step 80**: Vox -2 articles predicted as +1 — model applies "quotes≠journalist" heuristic too aggressively, treating all loaded language as attributed.

### C2 Prediction Distribution Shift
| Phase | Most Common | Mean | Pattern |
|-------|------------|------|---------|
| Steps 0-30 | 0 (43%), -3 (38%) | -1.38 | Binary bimodal |
| Steps 40-60 | -3 (43%) | -1.71 | Negative bias deepens |
| Steps 70-110 | 0 (32%), +1 (21%) | -0.74 | Discovers positive range, over-corrects |

### C2 Self-Correction Disabled
Despite `disc_selfcorrect=true`, NO responses contain pre-handoff reasoning. All go straight to `<handoff>{...}`. The self-check instructions are completely ignored. Reasoning is compressed into the JSON `reasoning` field, limiting deliberation space.

### Implications for V50+ Experiments
- Magnitude feedback (V51) directly addresses "can't represent mild C2" by telling model HOW MUCH to adjust
- Hindsight explanation (V53) forces the model to articulate WHY GT=-1 (not -3) — learning the middle range
- Causal prompt (V49) structures analysis BEFORE scoring, potentially preventing snap judgments
- Confidence weighting (V52) lets model express uncertainty about C2 mild cases

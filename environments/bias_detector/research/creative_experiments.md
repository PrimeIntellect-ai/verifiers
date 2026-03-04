# Bias Detector Experiment Registry

## Completed Runs

### Baseline (V24-V28)
| Run | Key Feature | MAE (last20) | C1 | Notes |
|-----|-------------|-------------|-----|-------|
| V25 | Self-correct prompt | -- | -- | |
| V26 | Mixed prompt | -- | -- | |
| V27 | C2 focus | -- | -- | |
| V28 | Quote-mask + mixed + diverse | 0.726 | 1.000 | **Best single-feature baseline** |

### Intermediate (V29-V32)
| Run | Key Feature | MAE (last20) | C1 | Notes |
|-----|-------------|-------------|-----|-------|
| V29 | Topic neutral | -- | -- | Degrades over time — HURTS |
| V30 | Contrastive | 0.841 | 1.142 | Worst — never learns |
| V31 | **Iterative correction** | **0.448** | **0.515** | **BEST. Only run still improving at 150.** |
| V32 | C1 specialist | 0.730 | 0.831 | C1 specialization helps C1 marginally |

### Running Pipeline (V33-V54)
| Run | Key Feature | Status |
|-----|-------------|--------|
| V33 | C2 specialist | RUNNING (step 114/150) |
| V34 | Rank-16 LoRA | Queued |
| V35 | 50% comparative | Queued |
| V36 | Source-blind + comparative | Queued |
| V37 | Best combo (quote-mask + iterative + mixed) | Queued |
| V38 | Comparative-iterative | Queued |
| V39 | Source-blind-iterative | Queued |
| V40 | High LR (3e-5) | Queued (after eval) |
| V41 | 8 rollouts | Queued |
| V42 | Extended 300 steps | Queued |
| V43 | Difficulty-aware advantage (boost subtle articles) | Queued |
| V44 | C1 3x + difficulty boost | Queued |
| V45 | **Filtered corpus + C1-balanced sampling** | Queued |
| V46 | **Topic deconfounding prompts** | Queued |
| V47 | **Ultimate combo (iterative + filtered + deconfound + difficulty)** | Queued |
| V48 | **Double iterative (2 correction passes)** | Queued |
| V49 | **Causal logic extraction (3-step structured analysis)** | Queued |
| V50 | **Real-only + all best features** | Queued |
| V51 | **Magnitude feedback in iterative correction** | Queued |
| V52 | **Confidence-weighted advantage** | Queued |
| V53 | **Hindsight explanation (novel)** | Queued |
| V54 | **Hindsight + Iterative (both learning signals)** | Queued |

## Key Findings Driving Experiment Design

### 1. Iterative correction is the strongest single feature (+38% MAE reduction)
V31 iterative vs V28 baseline: 0.726 → 0.448. The model learns to self-correct with
80% direction compliance, 0% wrong direction. C3 corrections are 100% accurate.

### 2. Synthetic corpus labels are poisoned
51% of synthetic C1 labels mismatch target by ≥2 points. The LLM judge assigned
C1=-3 to 43% of synthetic articles regardless of target. This creates a negative
C1 prior in training. V45 filters to max_mismatch ≤ 2 (keeps 188/456).

### 3. Topic-frame confusion is the #1 C1 failure mode
Model conflates article TOPIC with CAUSAL FRAME direction. Healthcare articles
get positive C1 bias; economy articles get negative pull. V46 adds same-topic
contrastive examples to explicitly teach "topic ≠ C1."

### 4. Training is stochastic search, not convergence
All runs show 32-45% CV in last 50 steps. No run achieves stable low MAE.
Best strategy: save checkpoints every 10 steps, select best post-hoc.
Only V31 (iterative) was still improving at step 145.

### 5. C1 is hardest, C3 is easiest
C1 MAE: 1.000 (V28), C2: 0.577, C3: 0.600.
C1 errors are systemic (negative prior, topic confusion).
C3 corrections are always correct (100% in V31 analysis).

### 6. V33 C2 specialist — marginal C2 improvement, not worth the C1 trade
V33 (disc_per_criterion_scales=[0.5, 2.0, 0.5]): MAE=0.626, C2=0.565.
C2 improvement vs V28 is tiny (0.577→0.565) while C1 degrades.
C2 specialist citations evolve: step 0 uses generic keywords ("neutral", "measured"),
step 50 adds quoted evidence ("suicide note", "vicious spiral"),
step 80 shows sophisticated attribution distinction ("Trump posted" ≠ journalist opinion).

## Experiment Hypotheses

### V45 (Filtered + Balanced): Address data quality
- Removes 268 synthetic articles with bad labels
- Oversamples underrepresented C1 bins
- Expected: C1 MAE improvement from cleaner training signal

### V46 (Deconfound): Address topic-frame confusion
- Adds "TOPIC ≠ C1" contrastive examples to discriminator prompt
- Shows same-topic articles with different C1 scores
- Expected: Reduced topic-prior shortcuts for C1

### V47 (Ultimate Combo): Kitchen sink
- Iterative + filtered + balanced + deconfound + difficulty + quote-mask
- Expected: Best overall if features compose well (not guaranteed)

### V48 (Double Iterative): Extended correction
- Two correction passes instead of one
- Pass 2 gives feedback on ALL bad criteria (not just worst)
- Expected: Fix C2's 50% no-change rate from V31 analysis

### V49 (Causal Logic Extraction): Structured analysis
- New DISCRIMINATOR_PROMPT_CAUSAL forces 3-step analysis before scoring
- Step 1: Extract causal claims (for C1) — "CAUSE is ___", "SOLUTION is ___"
- Step 2: Voice decomposition (for C2) — identify journalist vs quoted speech
- Step 3: Evidence enumeration (for C3) — list and count evidence types
- Combined with iterative correction + filtered corpus + C1 balance
- Addresses ALL 5 failure modes identified in hard-article analysis

### V50 (Real-Only + Best): Clean data hypothesis
- Drops ALL synthetic articles (51% C1 mismatch makes them toxic)
- Uses only 49 real articles with verified labels, oversampled 20x
- Combines: causal prompt + iterative + magnitude feedback + deconfound
- Hypothesis: Clean labels > data volume. Quality > quantity.

### V51 (Magnitude Feedback): Precision correction
- Adds error magnitude to iterative correction: "off by ~3 points" not just "too negative"
- Gives model a concrete correction target range
- Hypothesis: C2's 50% no-change rate is because direction-only is too vague
- V31's base + magnitude as the only change (clean ablation)

### V52 (Confidence-Weighted): Calibration training
- Model predicts C1/C2/C3 AND confidence (1-10)
- Advantage scaled by confidence/10: high confidence amplifies signal
- High confidence + right → large positive, high confidence + wrong → large negative
- Teaches calibration: express low confidence on hard C1 cases
- Hypothesis: Reduces gradient noise from random C1 errors

### V53 (Hindsight Explanation): Novel — learn to reason about bias
- After prediction, model sees CORRECT scores, must explain WHY they're correct
- "Hindsight experience replay" for bias detection:
  1. Predict scores (accuracy feedback)
  2. See correct answer (supervised signal)
  3. Explain why (reasoning feedback)
- Explanation gets its own advantage based on reasoning quality
- Hypothesis: Explaining correct answers teaches the criteria deeply,
  transferring back to better predictions. The model doesn't just learn
  what scores to predict, but WHY those scores are correct.

### V54 (Hindsight + Iterative): Maximum learning signal
- Combines iterative correction AND hindsight explanation
- Flow: predict → get correction → re-predict → see answer → explain
- 4-step discriminator pipeline with maximum feedback per rollout
- Causal prompt + magnitude feedback + filtered corpus + C1 balance
- Hypothesis: Two complementary learning signals compose well —
  correction teaches "how to adjust" while hindsight teaches "why"

### V55 (Ultimate V2): All winning features combined
- Every technique that showed positive signal: iterative + magnitude + causal + hindsight + deconfound + quote-mask
- Real-only with 20x oversample
- Kitchen sink approach — if individual features help, the combo should dominate

### V56 (Curriculum): Easy-to-hard scheduling
- Dataset sorted by difficulty (Manhattan distance from score mean)
- Early training sees easy examples (near-mean scores), later sees extreme ones
- Combined with iterative correction
- Hypothesis: reduces early training variance, builds stable baseline before tackling hard C1 cases

### V57 (Decomposed Iterative): Per-criterion correction
- Instead of "worst criterion is wrong," gives targeted correction for EACH criterion with error >= 2
- C1 gets causal-frame coaching: "What CAUSE? What SOLUTION?"
- C2 gets voice decomposition: "List journalist sentences, ignore quotes"
- C3 gets evidence enumeration: "List and count evidence types"
- Each criterion corrected individually with criterion-specific scaffolding
- Hypothesis: V31's generic correction is too vague; criterion-specific coaching will especially help C1

### V58 (Decomposed + Causal): V57 + structured initial analysis
- Combines V57's per-criterion correction with V49's causal prompt
- Causal prompt provides structured reasoning, decomposed correction targets weaknesses
- Also adds topic deconfounding (V46)

### V59 (Attention-Only LoRA): Surgical adapter
- LoRA restricted to q/k/v/o projections only (no MLP layers)
- Same V31 iterative setup for apples-to-apples comparison
- Hypothesis: bias scoring is reading comprehension; attention patterns matter more than MLP features
- Config-only change, no environment modifications

### V60 (Adaptive Advantage): Anti-stagnation
- Tracks per-example error over last 5 encounters
- Persistently-wrong examples get 1.5x advantage boost
- Already-mastered examples get 0.5x advantage dampen
- Implicit active learning at the advantage level
- Hypothesis: V31 stagnates after step 90 because gradient signal vanishes for "good enough" examples

### V61 (Sequential Scoring): One criterion at a time
- After initial all-3 scoring, model re-scores each criterion individually
- C1-only → C2-only → C3-only with criterion-specific instructions
- Each pass gets full reasoning depth for one criterion
- Hypothesis: concurrent multi-criteria scoring causes interference; dedicated reasoning improves accuracy

## Radical Experiments (V62-V65)

### V62 (Perturbation Augmentation): Synthetic data with known transforms
- Programmatically modify articles with known bias transforms:
  - c2_neutralize: replace loaded words with neutral equivalents → C2 shifts positive
  - c3_strip_sources: remove quoted sources → C3 shifts negative
  - c1_reframe: replace structural terms with agency terms → C1 shifts positive
- 50% of rollouts get perturbed articles with adjusted target scores
- Creates cleaner training signal than noisy single-rater labels
- Combined with iterative correction

### V63 (Perturbation + Decomposed): V62 data augmentation + V57 correction
- Combines perturbation augmentation with per-criterion decomposed correction
- Model gets perturbed articles AND targeted criterion-specific feedback
- Plus topic deconfounding

### V64 (Relative Reward): Beat-the-Baseline
- Advantage = improvement over base model predictions, not absolute accuracy
- Robust to label noise: even if ground truth is wrong by 1 point, both base and LoRA are equally disadvantaged
- Naturally focuses on hard examples (where base model fails)
- NOTE: needs pre-computed base predictions in info["base_pred"]; falls back to standard if missing

### V65 (Ultimate V3): Everything + perturbation + adaptive advantage
- Causal prompt + decomposed iterative + magnitude feedback + perturbation + adaptive advantage + deconfound
- Real-only with 20x oversample
- The kitchen sink with ALL novel features: perturbation augmentation, adaptive advantage, decomposed correction

## Creative Paradigm Shifts (V67-V70)

### V67 (Checkpoint Ensemble): Train-many, ensemble post-hoc
- Same as V31 (best config) but extended to 200 steps with checkpoints every 5 steps
- Post-hoc: learn mixing weights over checkpoint predictions to capture "best window" reliably
- Hypothesis: the model CAN reach MAE=0.15 (V31 step 145) but can't sustain it; ensemble averages capture this
- Key insight: training is stochastic search, different checkpoints specialize for different article types
- Script: `/tmp/ensemble_mix.py outputs/bias_detector_v67`

### V69 (Criterion Decomposition): Concrete sub-features eliminate abstraction
- Instead of scoring abstract "C1 Frame", model scores 7 concrete sub-features:
  - C1: headline loading, source balance, emphasis direction
  - C2: emotional appeals, certainty/hedging
  - C3: source attribution, data citation
- Final C1/C2/C3 = mean of sub-features (clamped to int range)
- Combined with iterative correction on composed scores
- Hypothesis: "Is the headline loaded?" is answerable without topic confusion; "Is the framing biased?" is not
- Directly attacks the C1 topic-frame confusion problem

### V70 (Temporal Difference): Progressive information reveal
- 4-stage scoring: headline → first paragraph → full article → gold C2 reveal
- Earlier predictions get higher reward multiplier (3x, 2x, 1x, 0.5x)
- TD bonus for improvement between stages
- Headline-only forces the model to learn: "political headline ≠ biased framing"
- Gold C2 reveal teaches cross-criterion correlation
- Hypothesis: multi-stage structure reduces variance via progressive constraint

## Self-Improvement & Stability Experiments (V71-V75)

### V71 (Self-Distillation): Temporal prediction anchoring
- Model sees its OWN prior predictions as soft anchor before re-scoring
- Consistency bonus for stable + accurate predictions; improvement bonus for correcting past errors
- Module-level `_PREDICTION_HISTORY` stores per-article prior predictions
- Combined with iterative correction
- Extended to 200 steps (history accumulates over training)
- Hypothesis: reduces 36% CV by creating a self-anchoring dynamic

### V74 (Socratic Probing): Self-generated questions force deep analysis
- 3-phase: generate probing questions → answer with article citations → score
- C1 questions ask about causal claims; C2 about journalist word choices; C3 about evidence types
- Answer quality bonus (0.1) for citing specific text (quotation marks heuristic)
- Combined with iterative correction
- Hypothesis: text-grounded analysis defeats source/topic shortcuts

### V75 (SWA-Extended): Aggressive checkpointing for post-hoc selection
- V31's best config extended to 300 steps with checkpoints every 5 steps
- Per-criterion peak analysis shows C1 peaks at step 32, C2 at step 34, C3 at step 144
- Oracle per-criterion MAE=0.069 vs best-single-step 0.153 (55% headroom)
- Only 2.7% of steps have all criteria simultaneously in "good state"
- Hypothesis: 60 checkpoints (steps 0-300, every 5) enables per-criterion selection

## New Key Findings (V71-V75 design phase)

### 6. Per-criterion peaks are temporally disjoint
V31: C1 peaks at step 32, C2 at step 34, C3 at step 144.
Oracle per-criterion MAE=0.069 vs best-single-step 0.153.
Only 2.7% of steps have ALL criteria simultaneously below 25th percentile.
This means criteria INTERFERE during training.

### 7. First-50 avg predicts final performance (r=0.914)
Early training signal quality determines the ceiling.
V31 is uniquely good early (0.488 first-50 avg) AND still improving.

### 8. Comparative scoring introduces anchoring bias
V35 shows 5-point C1 swings for IDENTICAL articles depending on random anchor.
Mean C1 range across same-article predictions: V35=1.00 vs V31=0.40.
Comparative format corrupts training signal with anchor-dependent noise.

### 9. Training volatility doesn't predict performance
Correlation between volatility and best-10 window = 0.138.
Stochasticity is not the problem — training signal quality is.

### 10. Step-to-step autocorrelation is zero
Across ALL runs, MAE autocorrelation = -0.03 to -0.05.
Consecutive training steps are completely independent.
Batch composition (which articles are sampled) drives step-level randomness.

### 11. V32 achieves oracle per-criterion MAE = 0.000
Every criterion hits exactly 0 at some training step.
Best single step: MAE=0.028 at step 62 (with short articles, avg 2888 chars).
But last-10 MAE=0.688 — the model CAN be perfect but CAN'T sustain it.

### 12. C1 dominates variance across all runs (38-50%)
C1 contributes 38-50% of total MAE variance.
Reducing C1 error variance is the single biggest lever for consistency.

### 13. V32's MAE=0.028 was LUCKY BATCH, not learned capability
At step 62: avg article length=2888 (1.2σ below mean). ALL sub-0.1 MAE steps have
unusually short articles. Batch composition (which articles sampled) is the dominant
source of step-level variance, not model learning.

### 14. 84% of V32 training steps have zero effective gradient signal
Because game_reward returns 1.0 for ANY completed rollout, GRPO normalization
can't distinguish good from bad predictions. The per_agent_advantage function
provides signal, but V32 has only 24 effective learning steps out of 150.
V31 has 47 effective learning steps — iterative correction creates more
reward variance, which is WHY it's the best run.

### 15. Article length strongly predicts MAE
Short articles (<3000 chars): avg MAE=0.267
Long articles (>7000 chars): avg MAE=0.747
Length-MAE corr = 0.208 (V31), Length-C1 corr = 0.243
Batch stratification by length would reduce variance more than training signal changes.

## Consistency-Focused Experiments (V76-V82)

### V76 (Criterion-Locked): Sequential per-criterion training phases
- Phase 1 (steps 0-50): ONLY C1 advantage (disc_per_criterion_scales = [3,0,0])
- Phase 2 (steps 50-100): ONLY C2 advantage (resume, [0,3,0])
- Phase 3 (steps 100-150): ONLY C3 advantage (resume, [0,0,3])
- Addresses disjoint peak timing: each criterion gets dedicated gradient signal
- Hypothesis: eliminates inter-criterion interference

### V78 (Anti-Regression): Monotonic improvement ratchet
- Tracks best-ever error per article in `_BEST_ERROR_HISTORY`
- Penalizes regression (0.3x regression magnitude), rewards new personal bests (+0.1)
- Combined with V31 iterative + V71 self-distillation for maximum stability
- Extended to 200 steps
- Hypothesis: prevents catastrophic forgetting of well-scored articles

### V80 (Sequential+Iterative): Per-criterion reasoning + correction
- Each criterion scored in dedicated phase (C1→C2→C3)
- After all 3, if worst error >= 2, iterative correction kicks in
- Equal per-criterion emphasis [2.0, 2.0, 2.0] (V31 underweighted C3)
- Modified env: sequential scoring no longer sets final_env_response when iterative enabled
- Hypothesis: dedicated reasoning per criterion + correction = V32-level accuracy consistently

### V81 (Low-Temperature): Reduced sampling randomness
- temperature=0.8 (all prior runs used 1.2)
- Otherwise identical to V31
- Hypothesis: T=1.2 is too high for exploitation, model samples conflicting strategies
- Should directly reduce CV by making predictions more deterministic

## Stability & Training Signal Experiments (V83-V88)

### V83 (Anchor Scoring): Close-reference comparison
- Instead of absolute -3 to +3 scoring, show CLOSEST article with known scores
- Model estimates small deltas from reference (easier cognitive task)
- Key difference from toxic V35: anchors are CLOSE (delta ~1-2), not distant
- Hypothesis: relative comparison easier than absolute scoring, especially for C1

### V84 (Temperature Annealing): Explore → Exploit
- Built-in linear scheduler: T=1.5→0.6 over 150 steps
- No env changes needed, uses prime-rl temp_scheduler
- Hypothesis: diverse strategies early (T=1.5) + focused predictions late (T=0.6)

### V85 (Accuracy Reward): Continuous gradient signal
- game_reward = 1.0 - error/18 (not binary 1.0)
- Addresses 84% zero-gradient problem: GRPO can distinguish good from bad rollouts
- Gives TWO gradient sources: GRPO normalization + per_agent_advantage
- Hypothesis: most impactful single change for training efficiency

### V86 (Length Stratify): Short-first curriculum
- Sort training pool by article length, train on short articles first
- Short articles (<3000 chars): MAE=0.267, Long (>7000 chars): MAE=0.747
- Hypothesis: stable foundation on easy articles before tackling hard ones

### V87 (Majority Vote): 3-pass median scoring
- 3 independent scoring passes, take median per criterion
- Median should reduce CV by ~42% (√3 factor)
- Pure noise reduction — no oracle needed

### V88 (Ensemble Distill): Expert perspective hints
- After first prediction, expert panel gives soft directional clues
- "Political Analyst: causal framing leans structural/systemic"
- Model revises scores based on expert perspectives
- Note: expert hints are derived from ground truth (not production-viable)

## Production-Viable Experiments (V89-V93)

**KEY INSIGHT**: V31's iterative correction leaks ground truth during training.
At inference, there's no oracle to say "your C1 is too negative."
These experiments test approaches that work WITHOUT ground truth at inference.

### V89 (Iterative + Accuracy Reward): Maximum training signal
- V31's iterative + V85's continuous reward = two gradient sources
- Deploy as single-pass model (iterative bakes in reasoning during training)

### V90 (Blind Self-Correction): NO ORACLE — production viable
- After first scoring, generic prompt: "double-check each criterion"
- No directional hint, no ground truth leak
- At inference: exact same prompt can be used
- CLEAN ABLATION: V90 vs V31 measures oracle value; V90 vs V25 measures second-pass value

### V91 (Devil's Advocate): Self-adversarial reasoning — production viable
- After first scoring, model argues AGAINST its own scores
- Forces deeper analysis: "what evidence would support a DIFFERENT C1?"
- Then revises based on its own counter-arguments
- At inference: fully deployable score→critique→revise pipeline

### V92 (Blind + Oracle): Train with both, deploy blind-only
- Flow: score → blind self-check → oracle correction (if still wrong)
- Oracle feedback teaches WHAT to look for during blind self-check
- At inference: use only first 2 turns (score → blind check)
- Extended to 200 steps

### V93 (Devil's + Oracle): Train with both, deploy devil's-only
- Flow: score → devil's advocate → revise → oracle correction (if still wrong)
- Devil's advocate surfaces errors, oracle corrects remaining ones
- At inference: use only first 3 turns (score → devil's advocate → revise)

## CRITICAL FINDING: Oracle Dependency (from V31 analysis)

**V31's iterative oracle training makes first-pass scoring 21% WORSE than V25 baseline.**
- V31 final MAE (with oracle): 0.489
- V31 estimated first-pass MAE: 0.764
- V25 MAE (no oracle): 0.632
- Correction rate INCREASES over training (20%→36%) — model becomes MORE dependent on oracle
- The oracle is a training-time CRUTCH, not baked-in reasoning improvement

This motivates V94-V99: experiments designed to prevent oracle dependency.

## Production-First Experiments (V94-V99)

### V94 (First-Pass Bonus): Reward first-pass accuracy explicitly
- Oracle iterative + disc_first_pass_bonus=1.0
- Equal weight to first-pass and final accuracy in advantage
- Hypothesis: model learns to get it right first AND correct if needed

### V95 (Oracle Withdrawal): 50% oracle withheld
- Oracle iterative + disc_oracle_withdrawal=0.5
- Half the time, no correction feedback — forces self-reliance
- Like a student who sometimes gets answer keys

### V96 (Bonus + Withdrawal): Combined approach
- First-pass bonus (1.5x) + oracle withdrawal (40%) + iterative
- Kitchen sink approach to preventing oracle dependency

### V97 (Blind + Devil's): FULLY PRODUCTION VIABLE
- Blind self-correction + devil's advocate, NO oracle
- 3-turn inference: score → self-check → devil's advocate revision
- 200 steps extended training

### V98 (AccReward + Devil's): Better gradients, no oracle
- Continuous accuracy reward (game_reward = 1.0 - error/18)
- Devil's advocate self-critique, no oracle
- Addresses 84% zero-gradient problem + production viability

### V99 (Ultimate Production): All production-viable strategies
- Accuracy reward + blind self-correct + devil's advocate + first-pass bonus (1.5x)
- NO oracle, fully deployable as-is
- 200 steps for deep learning

## Novel Paradigm Experiments (V100-V102)

### V100 (Evidence-First): Extract evidence before scoring
- 2-turn discriminator: extract evidence spans → score based on evidence
- Forces grounding in observable text features
- Combined with devil's advocate for self-check
- PRODUCTION VIABLE — no oracle needed

### V101 (Pairwise Ranking): Reference article comparison
- Model sees a reference article with known scores + target article
- Must compare target to reference, then give absolute scores
- Relative judgment is easier than absolute scoring
- Combined with accuracy reward

### V102 (Consistency Training): Score same article twice
- Model scores with one framing, then re-scores with a different angle
- Consistency bonus in advantage function
- Trains robustness to prompt variations
- Combined with blind self-correction + accuracy reward

# Next Wave Experiments (V40+) — Design Notes

## Key Insight from V25-V39
Training gains are marginal (~6.6% best). The biggest improvements come from eval methodology.
This suggests we're limited by: (1) data size (49 articles), (2) base model capacity (4B params).

## V40: Learning Rate Sweep
Same as V28 (best model) but with different learning rates.
- V40a: lr=3e-5 (3x higher — faster convergence, possibly overfitting)
- V40b: lr=5e-6 (0.5x — slower, more stable)
Hypothesis: We may be under/over-learning. 1e-5 was chosen without optimization.

## V41: Rollout Scaling
Same as V28 but with 8 rollouts per example instead of 4.
Hypothesis: With only 4 rollouts, advantage estimates are noisy. 8 rollouts gives
2x lower variance, which could help the model learn more precise scoring.

## V42: Extended Training (300 steps)
V28 setup but run to 300 steps instead of 150.
Hypothesis: We stop at 150 but the model may still be improving. Loss was still
declining at step 150 in most runs.

## V43: Data Augmentation via Paraphrasing
Use the base model to paraphrase each of the 49 articles (preserve content, change
surface form). Create 3 paraphrases per article = 196 training articles total.
Hypothesis: More diverse surface forms help the model focus on bias signals rather
than surface correlations.

## V44: Pairwise Contrastive Training
Instead of "score this article", the task is "which article has higher C1?"
Present two articles, model must rank them on each criterion. Advantage based on
ranking accuracy rather than absolute scoring accuracy.
Hypothesis: Relative comparison is easier than absolute scoring. Pairwise training
directly addresses center-pull (model learns "Article A has MORE structural framing
than Article B" rather than "Article A has C1=-2").
NEEDS: New environment feature (disc_pairwise mode).

## V45: Multi-Criterion Decomposed Training
Score each criterion separately (3 different rollouts per article: C1-only, C2-only, C3-only).
Each rollout focuses on one criterion with criterion-specific prompts.
Hypothesis: Joint scoring may cause interference between criteria. Decomposed scoring
lets the model fully focus on one aspect at a time.

## V46: Calibration-Aware Training
Add calibration bonus to the advantage function: reward predictions that match the
overall score distribution (e.g., if model predicts C1=0 for 80% of articles but
actual distribution is 30% C1=0, penalize).
Hypothesis: Center-pull is partly because the model learns a "safe" prediction of 0.
Explicit calibration pressure could fix this.

## V47: Adversarial Article Selection
Instead of random article selection per rollout, select the articles the model
currently scores WORST on. Curriculum learning from hardest examples.
Hypothesis: The model wastes training signal on easy articles. Focusing on hard cases
(Art 3, 7, 10, 15, 18) could produce more targeted improvement.

## Priority Order
1. V42 (extended training) — easiest, just config change
2. V40 (LR sweep) — easy, just config changes
3. V41 (rollout scaling) — easy, just config change
4. V43 (paraphrasing) — requires data generation step
5. V47 (adversarial selection) — requires env feature
6. V44 (pairwise) — requires new env feature
7. V45 (decomposed) — requires env restructuring
8. V46 (calibration-aware) — requires advantage function changes

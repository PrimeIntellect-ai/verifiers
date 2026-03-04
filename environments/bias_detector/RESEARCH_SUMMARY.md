# Bias Detector: RL-Trained News Article Bias Scoring

## Overview

This environment trains a Qwen3-4B-Instruct model via GRPO reinforcement learning to score news articles on three bias criteria (C1 Frame, C2 Temperature, C3 Evidence), each on a -3 to +3 integer scale. The model is trained in a GAN-like setup where a generator produces articles with target bias scores and a discriminator learns to predict those scores.

## Dataset

- **49 labeled real news articles** with human-annotated C1/C2/C3 scores
- **~1000 synthetic corpus articles** generated via LLM with target scores
- Real articles oversampled 10x to dominate training signal
- Articles sourced from NYT, BBC, Vox, WSJ, Atlantic, ABC, etc.

## Architecture

- **Model**: Qwen/Qwen3-4B-Instruct-2507 with LoRA rank 8
- **Training**: GRPO with per-step advantages, batch_size=24, rollouts_per_example=4
- **Sequence length**: 12288 tokens (articles + few-shot examples + reasoning)
- **Environment**: `MultiAgentEnv` with generator (disabled, advantage=0) and discriminator (active)

## Leaderboard (109 experiments, V1-V109)

### Top Models by Training MAE (last-10 step average)

| Rank | Version | MAE | C1 | C2 | C3 | Key Feature |
|------|---------|-----|-----|-----|-----|-------------|
| 1 | **V31** | **0.392** | 0.446 | 0.417 | 0.312 | Iterative oracle correction |
| 2 | **V38** | **0.392** | 0.400 | 0.404 | 0.371 | V31 + source blind + 3 examples |
| 3 | V104 | 0.407 | 0.442 | 0.421 | 0.358 | V31 + 3 examples (no source blind) |
| 4 | V37 | 0.447 | 0.533 | 0.487 | 0.321 | Best combo (quote-mask + iterative) |
| 5 | V25 | 0.581 | 0.850 | 0.463 | 0.429 | Self-correction prompt |

### Best-5 Window (peak performance)

| Version | Best-5 MAE | At Step | Notes |
|---------|-----------|---------|-------|
| **V104** | **0.250** | 33 | Best peak ever |
| V38 | 0.272 | 29 | Second best peak |
| V103 | 0.300 | 106 | V31 + source blind only |
| V31 | 0.342 | 143 | Baseline champion |

## Key Findings

### 1. Iterative Oracle Correction is the Dominant Feature (+50% MAE reduction)
V31 (iterative) vs V28 (baseline): 0.726 → 0.392. When the discriminator's worst criterion is off by 2+ points, it receives directional feedback ("your C2 seems too negative") and re-scores. 84% of corrections improve total error.

### 2. Fewer Examples Improve Performance (5 → 3)
V104 (3 examples) achieved Best-5 of 0.250 vs V31's 0.342 (5 examples). Less context noise = better learning signal.

### 3. Source Blinding Has Mixed Results
V103 (V31 + source blind only): MAE 0.476 (+21% worse). V38 (V31 + source blind + 3 examples): MAE 0.392 (tied). Source blinding alone hurts; combined with fewer examples it's neutral.

### 4. Multi-Turn Self-Correction Strategies HURT Performance
Every attempt to add production-viable self-correction turns degraded MAE:
- **V105** (Devil's Advocate + Oracle Withdrawal + First-Pass Bonus): MA10=0.532 — first-pass bonus creates "never revise" stubbornness
- **V106** (Blind Self-Correction, no oracle): MA10=0.8+ — no learning signal without oracle
- **V108** (Blind Self-Correction + Oracle): MA10=0.547 — extra turn adds noise

**Root cause**: Additional turns dilute the advantage signal per token. The model doesn't learn to self-correct; it either ignores the revision prompt or makes random changes.

### 5. Oracle Dependency is a Real Problem
- Correction rate INCREASES over training (12% → 62%)
- Model becomes lazier at first pass, relying on oracle corrections
- Production one-shot MAE is estimated ~0.512 vs training MAE 0.392
- But iterative training DOES bake in 19% improvement over non-iterative (V25)

### 6. Training is Stochastic Search, Not Convergence
All runs show 32-45% coefficient of variation in the last 50 steps. No run achieves stable low MAE. Step-to-step autocorrelation is ~0 (batch composition drives variance). Best strategy: save frequent checkpoints, select best post-hoc.

### 7. C1 (Frame) is Hardest, C3 (Evidence) is Easiest
- C1 contributes 38-50% of total MAE variance
- Three C1 failure modes: topic-as-proxy, source prior, center-extreme confusion
- C3 corrections are 100% accurate in V31 analysis
- C2 struggles with quote/voice conflation (attributed speech vs journalist tone)

### 8. Synthetic Corpus Labels are Poisoned
51% of synthetic C1 labels mismatch target by ≥2 points. LLM judge assigned C1=-3 to 43% regardless of target. Using real articles with oversample (10x) is critical.

### 9. Comparative/Contrastive Training is Toxic
V35 (comparative): MAE 1.132 — worst overall. Introduces anchoring bias with 5-point C1 swings depending on random anchor article.

### 10. Per-Criterion Peaks are Temporally Disjoint
V31: C1 peaks at step 32, C2 at step 34, C3 at step 144. Oracle per-criterion MAE = 0.069 vs best single step 0.153. Only 2.7% of steps have all criteria simultaneously good.

## Feature Impact Analysis (14 runs with wandb data)

| Feature | Impact | Evidence |
|---------|--------|----------|
| Iterative oracle correction | **+50% improvement** | V31 vs non-iterative baselines |
| Fewer examples (5→3) | **+15% best-5** | V104 B5=0.250 vs V31 B5=0.342 |
| Topic hints | +10-15% | Consistent across runs |
| Mixed prompts | +5-10% | V26 vs V25 |
| Diverse example selection | +5% | Manhattan distance selection |
| C2-heavy criterion scales | +8% | [1.5, 2.0, 0.5] vs equal |
| Source blinding | Neutral/harmful | V103 +21% worse, V38 neutral |
| Quote masking | Neutral | V28 early but unstable |
| Topic debiasing | Harmful | V29 degrades over time |
| Comparative training | **Toxic** | V35 worst run (+0.319 MAE) |
| Devil's advocate | Harmful | V105 stubbornness |
| Blind self-correction | Harmful | V106/V108 worse than baseline |
| First-pass bonus | Harmful | Creates "never revise" incentive |

## Production Deployment Strategy

### Problem
The best models (V31, V38, V104) are trained with oracle corrections requiring ground truth — unavailable in production.

### Current Best Production Approach
Use V104 adapter (3 examples, iterative training) for single-pass inference:
1. Load adapter at step 75-100 (best stable window)
2. Use the same few-shot prompt format as training
3. Single-pass scoring (no iterative correction at inference)
4. Expected production MAE: ~0.5 (vs 0.392 training MAE)

### What Didn't Work for Production
- Multi-turn self-correction (V105-V108): all worse than single-pass
- Oracle withdrawal during training: insufficient signal when combined with other changes
- First-pass bonus: creates stubbornness

### Unexplored Production Directions
- Pure single-turn training (no iterative at all) — V109 config created but not run
- Curriculum: train with oracle first, then fine-tune without oracle
- Evidence-first scoring (V107 config created but not run)
- Longer training (200-300 steps)
- Higher learning rate

## Environment Parameters

### Core Parameters
- `advantage_center`: Baseline subtracted from advantage (0.3)
- `gen_advantage_scale`: Generator advantage weight (0.0 = disabled)
- `disc_advantage_scale`: Discriminator advantage weight (2.0)
- `disc_per_criterion_scales`: Per-criterion weights (best: [1.5, 2.0, 0.5])

### Data Parameters
- `disc_num_examples`: Few-shot examples in prompt (best: 3)
- `disc_diverse_examples`: Manhattan distance selection (true)
- `disc_topic_hint`: Include topic in prompt (true)
- `disc_mixed_prompt`: Rotate prompt variants (true)
- `use_real_articles`: Include real labeled articles (true)
- `use_corpus`: Include synthetic corpus (true)
- `real_oversample`: Real article oversample factor (10)
- `num_seed_rows`: Training pool size (1000)

### Multi-Turn Parameters
- `disc_iterative`: Oracle directional correction (true = best)
- `disc_double_iterative`: Two correction passes
- `disc_oracle_withdrawal`: Probability of skipping oracle (0-1)
- `disc_blind_selfcorrect`: "Double-check" prompt without oracle
- `disc_devils_advocate`: Argue against own scores
- `disc_evidence_first`: Extract evidence before scoring
- `disc_consistency_training`: Score twice, reward agreement
- `disc_first_pass_bonus`: Advantage bonus for initial accuracy
- `disc_source_blind`: Strip publication names (0-1 probability)

## File Structure

```
environments/bias_detector/
├── bias_detector.py          # Main environment (2300+ lines)
├── pyproject.toml            # Package config
├── articles.json             # 49 labeled real articles
├── corpus_v2.json            # Synthetic corpus v2
├── corpus_v3.json            # Synthetic corpus v3
├── synthetic_corpus.json     # Original synthetic corpus
├── generate_corpus.py        # Corpus generation script
├── rescore_and_expand.py     # Corpus expansion script
├── eval_checkpoint.py        # Checkpoint evaluation script
├── research/                 # Research notes and findings
│   ├── findings_v2.md        # V2 findings (eval methods)
│   ├── findings_v3.md        # V3 findings (V25-V39 analysis)
│   ├── findings_v4_production.md  # Production deployment focus
│   ├── creative_experiments.md    # Full experiment registry (V1-V102)
│   ├── INSTRUCTIONS.md       # Scoring criteria instructions
│   └── articles.json         # Research copy of articles
└── RESEARCH_SUMMARY.md       # This file
```

## Reproducibility

### Running a training experiment
```bash
cd /home/ubuntu/prime-rl
export $(grep -v '^#' .env | xargs)
cd /home/ubuntu/verifiers/environments/bias_detector && uv pip install -e .
cd /home/ubuntu/prime-rl
uv run rl @ configs/bias_detector_v31/rl.toml  # or any version
```

### Monitoring
```bash
python3 scripts/training_health.py outputs/bias_detector_v31
python3 scripts/view_rollouts.py outputs/bias_detector_v31 --step 50
```

### Converting adapters for inference
```bash
# After training, convert .bin to .safetensors and fix key names
uv run python -c "
import torch
from safetensors.torch import save_file
from pathlib import Path
d = Path('outputs/bias_detector_v31/weights/step_100/lora_adapters')
for f in sorted(d.glob('*.bin')):
    w = torch.load(f, map_location='cpu', weights_only=True)
    new_w = {k.replace('.lora_A.0.weight', '.lora_A.weight').replace('.lora_B.0.weight', '.lora_B.weight'): v for k, v in w.items()}
    save_file(new_w, str(d / f.name.replace('.bin', '.safetensors')))
    f.unlink()
"
```

# Textify experiments

This directory records experiments run from PR #2034. The implementation is an
experimental vehicle and reference for a future user-space `@vf.intercept` version; these
reports are not claims of benchmark quality.

## Main results

Qwen3.5-9B was evaluated on all 847 usable MMMU validation prompts with 10 rollouts
per prompt and arm (`8,470` clean rollouts each). ASCII used width 160; Braille used
width 80, which samples the same 160 horizontal source pixels because each Braille cell
spans two pixels.

| Arm | Accuracy | Mixed prompts | Mean within-prompt sample variance | Cost |
|---|---:|---:|---:|---:|
| Vision | 72.05% | 256/847 | 0.0545 | $8.58 |
| ASCII, fixed | 52.59% | 452/847 | 0.1029 | $11.87 |
| ASCII, Otsu | 52.55% | 454/847 | 0.1025 | $12.18 |
| Braille, fixed | 52.05% | 462/847 | 0.1060 | $13.14 |
| Braille, Otsu | 51.94% | 474/847 | 0.1062 | $13.31 |

All `42,350` official rollouts completed cleanly. Otsu did not improve aggregate
accuracy or materially improve variance over fixed rendering, so it was removed from
the prototype after this experiment.

## Reports

- [MMMU model scale](mmmu-model-scale.md) — native vision versus ASCII at Qwen3.5 0.8B,
  9B, and 122B-A10B, with five rollouts per prompt.
- [Otsu thresholding](otsu.md) — rendering analysis plus the full reward-level validation
  that led to removing the adaptive option.

## Reproducibility notes

The raw traces were produced locally and are too large to commit because task data retains
base64 images. Each report records its configuration, providers, data selection, and known
failure modes. Provider/infrastructure failures were retried and excluded from model rewards.

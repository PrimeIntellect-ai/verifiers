# Textify experiments

This directory records experiments run from PR #2034. The implementation is an
experimental vehicle and reference for a future user-space `@vf.intercept` version; these
reports are not claims of benchmark quality.

## Reports

- [MMMU model scale](mmmu-model-scale.md) — native vision versus ASCII at Qwen3.5 0.8B,
  9B, and 122B-A10B, with five rollouts per prompt.
- [Otsu thresholding](otsu.md) — rendering-level comparison of fixed `0.5` and adaptive
  Otsu thresholds on the same MMMU images.

## Main finding so far

ASCII lowers mean reward but can increase within-prompt reward variance. On this small
sample, Qwen3.5-9B is the cleanest candidate for RL experiments: ASCII increased mixed
reward groups from 3/20 to 7/20 and mean within-prompt sample variance from 0.040 to 0.090,
without the severe output repetition observed at 0.8B.

## Reproducibility notes

The raw traces were produced locally and are too large to commit because task data retains
base64 images. Each report records its configuration, providers, data selection, and known
failure modes. Provider/infrastructure failures were retried and excluded from model rewards.

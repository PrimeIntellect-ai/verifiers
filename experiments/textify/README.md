# Textify experiments

This directory records experiments run from PR #2034. The implementation is an
experimental vehicle and reference for a future user-space `@vf.intercept` version; these
reports are not claims of benchmark quality.

## Reports

- [MMMU model scale](mmmu-model-scale.md) — native vision versus ASCII at Qwen3.5 0.8B,
  9B, and 122B-A10B, with five rollouts per prompt.
- [Otsu thresholding](otsu.md) — rendering analysis plus the full reward-level validation
  that led to removing the adaptive option.

## Main findings

ASCII lowers mean reward but increases within-prompt reward variance. The full Qwen3.5-9B
validation run confirmed this across 847 prompts: vision scored 72.05% with 256 mixed groups,
while the four text arms clustered at 51.94–52.59% with 452–474 mixed groups. Adaptive Otsu
thresholding did not improve accuracy or materially improve variance over fixed rendering, so
the option was removed from the prototype.

## Reproducibility notes

The raw traces were produced locally and are too large to commit because task data retains
base64 images. Each report records its configuration, providers, data selection, and known
failure modes. Provider/infrastructure failures were retried and excluded from model rewards.

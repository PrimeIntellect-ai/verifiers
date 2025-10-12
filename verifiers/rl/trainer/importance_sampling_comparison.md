# Importance Sampling Ratio Comparison

## Overview
This note compares the importance sampling (IS) implementations in the local `RLTrainer` and PrimeIntellect's `prime-rl` trainer. The two systems log very different ratio curves in practice—`prime-rl` reports ratios that stay near one, while our trainer often starts below one and drifts downward. The sections below highlight why those behaviors diverge and what would have to change for the implementations to match.

## Current `verifiers` implementation
* The PPO-style ratio inside `compute_loss` now contrasts the model's fresh log probabilities with the rollout-time `sampling_logprobs`, mirroring `prime-rl`'s definition while keeping our symmetric `[1 - ε, 1 + ε]` clamp. That means the unclipped term reflects how much the updated policy deviates from the rollout policy on a per-token basis.【F:verifiers/rl/trainer/trainer.py†L335-L374】
* No extra off-policy multiplier is applied—the clipped PPO ratio alone scales the advantages. The logged diagnostic therefore reports the same ratio that drives the loss.【F:verifiers/rl/trainer/trainer.py†L371-L406】
* The async generator stores those rollout log probabilities (zeros for prompts plus the sampled completion logprobs) when it assembles per-process microbatches, so `sampling_logprobs` encode how vLLM scored the tokens at collection time.【F:verifiers/rl/trainer/generator.py†L340-L383】

## `prime-rl` implementation
* In contrast, `prime-rl` feeds the trainer both the freshly computed log probabilities and the rollout-time log probabilities (`old_logprobs`) and defines the PPO ratio directly as `logprobs - old_logprobs`. The ratio is optionally reduced to a single scalar per sequence, with an exponential cap before clipping.【F:verifiers/rl/trainer/importance_sampling_comparison.md†L32-L38】
* Clipping enforces only an upper bound (`clip_ratio`, default 8.0) on the exponentiated ratio, so values can drop well below one when the policy underfits the rollout distribution.【F:verifiers/rl/trainer/importance_sampling_comparison.md†L40-L41】
* During training, `prime-rl` may wake a secondary "logprob" model that replays the microbatch using the checkpoint that produced the rollout. This recomputation keeps `old_logprobs` aligned with the data even when inference runs asynchronously, which pushes the measured ratios toward one.【F:verifiers/rl/trainer/importance_sampling_comparison.md†L43-L51】

```python
# prime-rl/src/prime_rl/trainer/rl/loss.py (excerpt)
for logprobs, old_logprobs, advantages, loss_mask in zip(...):
    log_importance_ratio = logprobs - old_logprobs
    if loss_config.ratio_type == "sequence":
        seq_log_importance_ratio = (log_importance_ratio[loss_mask]).sum()
        ...
    importance_ratio = torch.exp(log_importance_ratio)
    clipped_importance_ratio = torch.clamp(importance_ratio, max=loss_config.clip_ratio)
    loss = -clipped_importance_ratio * advantages
```

```python
# prime-rl/src/prime_rl/trainer/rl/train.py (excerpt)
if logprob_model is not None:
    ...
    recomputed_logprobs = selective_log_softmax(...)
    micro_batch["logprobs"] = recomputed_logprobs.cpu()
...
old_logprobs = micro_batch["logprobs"].to("cuda")
loss, _ = compute_loss(
    logprobs=logprobs.squeeze().split(response_lengths),
    old_logprobs=old_logprobs.squeeze().split(response_lengths),
    ...
)
```

## Why the logged ratios differ
1. **Asynchronous staleness.** We rely on the generator's cached `sampling_logprobs`, so if inference runs ahead of optimization the stored rollout policy can drift from the current model. `prime-rl` mitigates this by optionally recomputing log probabilities with the checkpoint that created the batch, which keeps its ratios closer to one.【F:verifiers/rl/trainer/importance_sampling_comparison.md†L43-L51】
2. **Clipping strategy.** Our symmetric clamp pushes ratios back toward one on both sides of the window, while `prime-rl` only caps large ratios. When advantages are positive, double-sided clipping biases our logged averages upward relative to `prime-rl`'s mostly one-sided behavior.
3. **Sequence reduction.** We still support per-token or per-sequence averaging for the log-ratio prior to exponentiation. Differences in this setting (or in masking) can alter the reported mean ratio even when the underlying inputs match exactly.【F:verifiers/rl/trainer/trainer.py†L347-L363】

## Matching `RLTrainer` to `prime-rl`
To mirror the `prime-rl` behavior completely we would still need to adjust a few pieces:
1. **Adopt one-sided clipping.** Switching to `torch.clamp(..., max=clip_ratio)` would let low ratios flow freely like `prime-rl`, at the cost of losing the symmetric guardrails we currently prefer.
2. **Introduce an optional recomputation pass.** Matching `prime-rl`'s auxiliary logprob model (or enforcing synchronous rollouts) would keep the logged ratios pinned near one even when generation runs ahead of training.【F:verifiers/rl/trainer/importance_sampling_comparison.md†L43-L51】
3. **Replicate sequence-level defaults.** Aligning configuration defaults for token vs. sequence averaging and mask handling would reduce remaining differences in how ratios aggregate across tokens.【F:verifiers/rl/trainer/trainer.py†L347-L363】

With those changes we would fully reproduce `prime-rl`'s reported ratios, though we currently prioritize the symmetric clip and simpler inference pipeline for stability in our async training setups.

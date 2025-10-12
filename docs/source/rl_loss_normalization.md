# Reinforcement Learning Loss Normalization

This note documents how the RL trainer keeps gradient magnitudes invariant to
how rollouts are partitioned across processes or microbatches.  The end goal is
that a distributed update matches the update we would compute if every rollout
were processed in a single microbatch on a single device.

## Notation

We work with the following notation for one optimizer step:

- $\mathcal{P}$ – set of processes participating in data parallel training.
- $\mathcal{B}_p$ – microbatches assigned to process $p \in \mathcal{P}$.
- $\mathcal{S}_{b}$ – sequences contained in microbatch $b$.
- $\mathcal{T}_{s}$ – completion tokens for sequence $s$ (prompt tokens are
  masked out before loss aggregation).
- $A_s$ – group-normalized advantage for sequence $s$.
- $r_{s,t}$ – truncated importance sampling (TIS) weight applied to token $t$ in
  sequence $s$.
- $g_{s,t}(\theta)$ – policy gradient contribution for token $t$ in sequence
  $s$ under parameters $\theta$ before any normalization is applied.

Two aggregate counts appear throughout the loss definitions:

- $N_{\text{seq}} = \sum_{p \in \mathcal{P}} \sum_{b \in \mathcal{B}_p}
  |\mathcal{S}_b|$ – the total number of sequences in the global batch.
- $N_{\text{tok}} = \sum_{p \in \mathcal{P}} \sum_{b \in \mathcal{B}_p}
  \sum_{s \in \mathcal{S}_b} |\mathcal{T}_s|$ – the total number of
  completion tokens that remain after masking and truncation.

For DR-GRPO we additionally cap every sequence at a fixed horizon
$H = \texttt{max\_seq\_len}$, so the global denominator is
$N_{\text{seq}} \cdot H$.

## Shared objective and truncated importance sampling

Every loss variant in the trainer implements the same clipped-TIS REINFORCE
objective.  For each token we form a truncated importance weight

$$
\tau_{s,t} = \min\left(\exp(\log \pi_\theta - \log \pi_{\text{ref}}),
                       1 + \varepsilon\right),
$$

optionally averaged over the sequence when `importance_sampling_level="sequence"`.
The trainer then applies the environment-supplied sampling correction

$$
\rho_{s,t} = \min\left(\exp(\log \pi_{\text{ref}} - \log \pi_{\text{sampler}}),
                       c_{\text{vllm}}\right),
$$

and the per-token gradient contribution becomes

$$
 g_{s,t}(\theta) = - \min(\tau_{s,t}, 1 + \varepsilon)
                    \, \rho_{s,t} \, A_s \, \nabla_\theta \log \pi_\theta.
$$

The only remaining choice is how to normalize the sum of these contributions.

## Loss-specific normalization

### GRPO (sequence-level normalization)

GRPO keeps the per-sequence mean of masked token contributions and then averages
across sequences:

$$
\mathcal{L}_{\text{GRPO}}
  = \frac{1}{N_{\text{seq}}}
    \sum_{p,b} \sum_{s \in \mathcal{S}_b}
      \frac{1}{|\mathcal{T}_s|}
      \sum_{t \in \mathcal{T}_s} g_{s,t}(\theta).
$$

Dividing by $N_{\text{seq}}$ ensures that adding more processes or slicing the
batch into additional microbatches does not change the effective learning rate.

### DR-GRPO (fixed-horizon normalization)

DR-GRPO follows GRPO’s token averaging but always divides by
$N_{\text{seq}} \cdot H$ with $H$ equal to the configured
`max_seq_len`.  This mirrors the delayed-reward objective in Prime RL where the
loss is normalized by the number of sequences times the fixed horizon, ensuring
that partial rollouts and padded tokens contribute consistently.

> **Note**
> The earlier BNPO option normalized by the live token count, which introduced
> cross-process variance once asynchronous generation created uneven
> microbatching.  Prime RL defaults to GRPO-style sequence averaging or the
> fixed-horizon DR-GRPO objective, so the trainer now exposes the same two
> choices.

## Implementation in the trainer

The training loop mirrors the mathematics above:

1. The async generator packages the batch with precomputed per-process item
   totals and the global denominator so that every rank receives the same
   metadata alongside the token tensors.【F:verifiers/rl/trainer/generator.py†L24-L54】【F:verifiers/rl/trainer/generator.py†L332-L383】
2. `training_step` materializes each microbatch once, padding and truncating on
   the fly before dispatching the tensors to `compute_loss`.  Each call ships the
   per-rank average `global_item_count / num_processes`, so gradient scaling
   happens before `Accelerate` reduces the partitions.【F:verifiers/rl/trainer/trainer.py†L210-L286】
3. `compute_loss` receives the shared denominator through `loss_denominator`
   and applies the appropriate normalization rule for the selected loss type,
   reproducing the behaviour of the reference implementation even when different
   ranks see different numbers of microbatches.【F:verifiers/rl/trainer/trainer.py†L392-L483】

Because the denominator is derived from the global counts and rescaled by the
world size, increasing the number of GPUs, changing the microbatch size, or
rearranging rollouts (e.g. future packing of non-sequential trajectories) leaves
the effective update unchanged as long as the global batch size and optimization
hyperparameters are fixed.

### Interaction with the Hugging Face Trainer and ZeRO-3

`training_step` performs the normalization before invoking
`self.accelerator.backward(loss)`.  Both PyTorch DDP and DeepSpeed’s ZeRO-3
integration in the Hugging Face Trainer average gradients across ranks, so the
trainer divides the summed loss by the per-rank average item count.  This keeps
the final update identical to the single-device reference while still letting
`Accelerate` and ZeRO-3 partition and reduce tensors transparently.  When
ZeRO-3 is active, the trainer temporarily gathers partitioned parameters before
syncing with vLLM, matching Hugging Face’s expectation that every rank returns a
loss already reduced over its local work.【F:verifiers/rl/trainer/trainer.py†L210-L334】【F:verifiers/rl/trainer/trainer.py†L354-L423】

## Relationship to Prime RL

Prime RL’s trainer accumulates a `loss_scale` per process equal to the number of
unmasked tokens when operating in token ratio mode, or to the number of
sequences in the local batch when using sequence ratio mode, and divides the
summed loss by that value.  Our trainer follows the same semantics by computing
those totals as part of batch preparation and shipping the aggregated
`global_item_count` with the rollout data; we then divide by the per-rank
average to counteract the distributed gradient averaging so every rank applies
the same normalizer even when asynchronous generation yields uneven microbatch
counts.【F:verifiers/rl/trainer/generator.py†L24-L116】【F:verifiers/rl/trainer/trainer.py†L210-L286】

# Training

This guide covers RL training with the included trainer and vLLM inference, orchestrated via a single TOML config.

## Training options

You can train with the included `RLTrainer` (via `vf-rl`) or with external projects like `prime-rl`.

- Use the included trainer when you want a simple, hackable training loop and LoRA-first defaults.
- Use `prime-rl` when you want FSDP-first orchestration and large-scale features.

### Summary of similarities and differences

- Similarities
  - OpenAI-compatible inference (vLLM) and async rollouts
  - One-step off-policy overlap by default (generate at step n-1 while training at step n)

- Differences
- RLTrainer: Accelerate/DeepSpeed-based; optional LoRA/PEFT; easy to script and extend in Python
  - PRIME-RL: FSDP-first; `rl` entrypoint; strong checkpointing; extensive CLI/TOML configuration

## Train with vf-rl (included trainer)

The included trainer runs alongside a vLLM server, managed automatically by `vf-rl` inside a tmux session. Configure everything in a single TOML.

## Quick Start

Install RL extras and set up default configs:

```bash
uv add 'verifiers[rl]'
uv run vf-setup
```

Launch training from a TOML (tmux with vLLM + trainer panes):

```bash
uv run vf-rl @ configs/rl/config.toml
```

## TOML Configuration

Minimal TOML example:

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"

[env]
id = "kalomaze/alphabet-sort" # auto-installed from hub if given as user/env-id, or from local project if given as env-id

[inference]
gpus = 1

[inference.args]
enforce_eager = true

[trainer]
gpus = 1

[trainer.args]
run_name = "alphabet-sort"
use_lora = true
learning_rate = 1e-5
micro_batch_size = 4
rollouts_per_example = 16
batch_size = 512
max_steps = 100
max_tokens = 512
max_seq_len = 2048
```

## Key Hyperparameters

### Batch Configuration

Key fields in `[trainer.args]`:
- `rollouts_per_example`: completions per prompt (group size)
- `micro_batch_size`: rollouts per GPU per step
- `batch_size`: rollouts per global batch (must be divisible by `micro_batch_size * world_size`)

**How to think about batch settings:**
- `num_generations`: Larger groups (16-32) increase reward diversity but use more memory
- `per_device_train_batch_size`: Limited by GPU memory after model weights
- `gradient_accumulation_steps`: Use to achieve larger effective batch sizes

### Generation Parameters

Specify in `[trainer.args]`:
- `max_tokens` (per-turn), `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`
- `max_prompt_len`, `max_seq_len`

**Generation strategy:**
- High temperature (0.8-1.0) increases diversity within groups
- Consider your model's context window when setting lengths
- Longer completions allow more complex reasoning but increase memory usage

### Training Schedule

Core fields in `[trainer.args]`:
- `learning_rate`, `lr_scheduler_type`, `warmup_steps`, `max_steps`
- `max_grad_norm`, `bf16`, `gradient_checkpointing`

### Async Generation

`RLTrainer` is asynchronous (one step off-policy) by default. Generation is controlled via `[trainer.args]` and the environment:
- `generation_timeout`, `max_concurrent`

## Evaluation During Training

Set `eval_strategy`/`eval_steps` in `[trainer.args]` and provide an eval split via your environment configuration if supported.

## Parameter-Efficient Training

LoRA is enabled by default; configure via `[trainer.args]` fields like `use_lora`, `lora_rank`, `lora_alpha`, `lora_dropout`, and optionally `lora_target_modules`.

## RL Rules of Thumb

RL is notoriously sensitive to implementation details. Here's practical guidance:

### Before Training

1. **Evaluate baseline performance**: If your model gets 0% reward after 10+ attempts, the task is too hard
2. **Check task difficulty**: If baseline is already 80%+, consider harder examples
3. **Ensure reward diversity**: You want varied scores within each generation group

### Stability vs Performance Trade-offs

**For more aggressive training** (higher risk of collapse):
- Set `beta = 0` (no KL penalty)
- Increase learning rate (2e-6 to 5e-6)
- Increase `num_iterations` (2-4)

**For more stable training** (slower progress):
- Increase `num_generations` (32-64)
- Increase batch size via `gradient_accumulation_steps`
- Decrease `max_grad_norm` (0.001-0.005)
- Use larger models (14B+)
- Keep `num_iterations = 1` (stay on-policy)

### Best Practices

**Likely beneficial:**
- Learning rate warmup (10-20 steps minimum)
- Periodic reference model updates for 500+ step runs
- One-step off-policy training (`num_batches_ahead = 1`)

**Context-dependent:**
- High `beta` values (0.1+) - more conservative
- Overlong filtering - depends on task
- Tool response masking - useful for multi-turn

**Key insight**: The best way to improve training is ensuring appropriate task difficulty for your model - not too easy, not too hard.

## Troubleshooting

### Common Issues

**Non-Increasing Chat Templates:** The Qwen3 and DeepSeek-R1 model series both remove `<think>` sections from messages when processing inputs, which violates the increasing context requirement for multi-turn training. We provide versions of many of these models with modified chat templates [here](https://huggingface.co/collections/willcb/qwen3-68434f4883925bfdb4570ee5).

**OOM during generation:**
- Reduce `num_generations` or `per_device_train_batch_size`
- Use LoRA instead of full finetuning
- Check vLLM server has sufficient memory

**Training instability:**
- Reduce learning rate
- Decrease `max_grad_norm`
- Increase `beta` for stronger KL regularization

**Poor reward diversity:**
- Increase temperature
- Check if task difficulty matches model capability
- Ensure your rubric differentiates quality levels

### Infrastructure
- Ensure `huggingface` and `wandb` logins are configured
- Set `OPENAI_API_KEY` (can be dummy for vLLM)
- Increase ulimit for high concurrency: `ulimit -n 4096`
- For NCCL issues: try `NCCL_P2P_DISABLE=1`

## Advanced Configuration

### Custom Sampling

```python
# Fine-grained generation control
args.repetition_penalty = 1.1   # Reduce repetition
args.top_k = 50                # Limit vocabulary
args.min_p = 0.05              # Min probability threshold
```

### Resource Optimization

```python
# Memory-constrained settings
args.gradient_checkpointing = True
args.ds3_gather_for_generation = False  # For very large models
args.generation_batch_size = 16  # Control generation batch size
```

### Monitoring

```python
# Logging configuration
args.logging_steps = 1
args.log_completions = True
args.report_to = "wandb"  # or "none" to disable
args.num_completions_to_print = 5  # Sample size to log
```

## Train with PRIME-RL

If you prefer an FSDP-first setup with higher throughput, you can train the same `verifiers` Environments using `prime-rl`.

- Install `prime-rl` (see its README for CUDA requirements):

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

- Create or install a Verifiers Environment module (inside your `prime-rl` checkout if developing there):

```bash
# create a new Environment template
uv run vf-init vf-custom-environment

# OR install an existing Environment from this repo
uv run vf-install vf-math-python --from-repo
```

- Configure the orchestrator to use your Environment. In your orchestrator TOML (e.g. `configs/my_exp/orch.toml`):

```toml
[environment]
id = "vf-math-python"  # or your custom environment ID

[environment.args]
# Example args forwarded to the Environment
split = "train"
rollouts_per_example = 8
max_concurrent = 512
```

- Launch a single-node run (adjust GPU split to your hardware):

```bash
uv run rl \
  --trainer @ configs/my_exp/train.toml \
  --orchestrator @ configs/my_exp/orch.toml \
  --inference @ configs/my_exp/infer.toml \
  --trainer-gpus 2 --inference-gpus 6
```

Tips:
- Use `bash scripts/tmux.sh` in `prime-rl` to open a panes layout for trainer/orchestrator/inference logs.
- Log to W&B by adding `--wandb.project <proj> --wandb.name <run>` on `uv run rl` (shared to trainer + orchestrator).
- For checkpointing/resume, see the `prime-rl` README (supports step-tagged checkpoints across trainer/orchestrator).


## Next Steps

- Explore [Environments](environments.md) to create custom tasks
- Review [Components](components.md) for advanced patterns
- See the [examples directory](https://github.com/PrimeIntellect-ai/verifiers/tree/main/examples) on GitHub for complete training scripts
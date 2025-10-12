### Verifiers Trainers

This directory contains the built-in GRPO trainer and configuration utilities for training with Verifiers environments.

- **Modules**: `GRPOTrainer`, `GRPOConfig`, `grpo_defaults`, `lora_defaults`
- **Exports**: available directly from `verifiers` (lazy-imported)

### Installation

For GPU training with the bundled trainer:

```bash
uv add 'verifiers[all]' && uv pip install flash-attn --no-build-isolation
```

To use latest `main`:

```bash
uv add verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git
```

### Quick Start (Python)

```python
import verifiers as vf

# 1) Load an installed Environment
env = vf.load_environment("vf-math-python")

# 2) Load a HF model + tokenizer
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

# 3) Configure GRPO
args = vf.grpo_defaults(run_name="my-experiment")

# 4) Train
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)
trainer.train()
```

See `examples/grpo/` for end-to-end scripts.

### Infrastructure

- **vLLM server** (inference):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model Qwen/Qwen2.5-7B-Instruct \
  --data-parallel-size 6 --enforce-eager --disable-log-requests
```

- **Training launcher** (Accelerate/DeepSpeed):

```bash
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config-file configs/zero3.yaml \
  --num-processes 2 your_training_script.py
```

### Key Arguments (GRPO)

- **Batching**: `per_device_train_batch_size`, `num_generations`, `gradient_accumulation_steps`
- **Lengths**: `max_prompt_length`, `max_completion_length`, `max_seq_len`, `max_tokens`
- **Optimization**: `learning_rate`, `lr_scheduler_type`, `warmup_steps`, `max_steps`, `max_grad_norm`, `num_iterations`
- **KL/Ref**: `beta`, `sync_ref_model`, `ref_model_sync_steps`, `ref_model_mixup_alpha`, `loss_type`
- **Async**: `num_batches_ahead`, `async_generation_timeout`, `max_concurrent`

Start from `vf.grpo_defaults(...)` and override as needed.

### LoRA / PEFT

```python
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
    peft_config=vf.lora_defaults(r=8, alpha=16),
)
```

### PRIME-RL Option

For FSDP-first, higher-throughput training (and richer orchestration), you can train the same Environments with `prime-rl`. See the `prime-rl` README for installation and `uv run rl ...` usage.

### Troubleshooting

- Configure `wandb` and `huggingface-cli` logins (or set `report_to=None`).
- Ensure `OPENAI_API_KEY` is set (a dummy key is fine for vLLM).
- For high concurrency, raise open files limit: `ulimit -n 4096`.
- For NCCL/vLLM weight-sync issues, try toggling `NCCL_P2P_DISABLE=1`.

### References

- Top-level README: repository setup, environment install, eval CLI (`vf-eval`).
- Docs: `docs/source/training.md` for deeper guidance on hyperparameters, stability trade-offs, and infra tips.



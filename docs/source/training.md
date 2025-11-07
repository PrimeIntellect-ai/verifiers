# Training

This guide covers RL training with `prime-rl` as well as the included `vf.RLTrainer`, both of which can be orchestrated via a single TOML config.

If your primary goal is to train a model on a Verifiers Environment, we recommend using `prime-rl`, which distills the best practices from our research team's experience and the broader community into a stable, easy-to-use recipe. If you want to hack on new training algorithms and are less concerned with maximum performance or advanced features, you can use the included `RLTrainer` (via `vf-rl`), whose core files are under 1000 lines of code and include only the most essential logic for fairly-performant async off-policy training (with the same core algorithm as `prime-rl`).


## `prime-rl`

We recommend using the [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) trainer, and provide a basic setup guide below. See the [prime-rl documentation](https://github.com/PrimeIntellect-ai/prime-rl) for more information.

To get started, do: 

```bash
uv run vf-setup
```

This will clone and install the `prime-rl` trainer and its dependencies, and set up a default configuration for training with the included `wiki-search` Environment.

Then, you can start training with:

```bash
uv run prime-rl @ configs/prime-rl/wiki-search.toml
```

This will launch a tmux session with separate panes for the trainer, orchestrator, and inference server.

### Configuration

`prime-rl` can be used with a single TOML file via its native `rl.py` script, which is used by the `uv run prime-rl` command from verifiers. 

Example configuration file for the `primeintellect/wiki-search` Environment with `Qwen/Qwen3-4B-Instruct-2507`:

```toml
inference_gpu_ids = [0]
trainer_gpu_ids = [1]

max_steps = 500

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[wandb]
project = "wiki-search"
name = "wiki-search-4b"

[trainer.optim]
lr = 1e-5
weight_decay = 0.0

[trainer.model.experimental.lora]
rank = 8
alpha = 32
dropout = 0.0
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

[orchestrator]
batch_size = 512
rollouts_per_example = 16
seq_len = 4096
mask_truncated_completions = false
zero_truncated_completions = true

[orchestrator.sampling]
max_tokens = 512

[orchestrator.buffer]
type = "online-difficulty"
oversampling_factor = 2.0

[[orchestrator.env]]
id = "primeintellect/wiki-search"

[inference.model]
enable_auto_tool_choice = true
tool_call_parser = "hermes"
```

For multi-node training, you will need to use separate configs for the trainer, orchestrator, and inference server, and launch processes separately. See the [prime-rl documentation](https://github.com/PrimeIntellect-ai/prime-rl) for more information.

## vf.RLTrainer

The included `RLTrainer` is a minimal, hackable training loop based on `transformers.Trainer` that supports both full-parameter finetuning and LoRA training. `RLTrainer` can be viewed as a "baby" `prime-rl` that adopts a similar default training recipe (async CISPO with one-step off-policy overlap), intended for single-node test runs with dense models. The primary files (`trainer.py` and `orchestrator.py`, located in `verifiers/rl/trainer/`) are under 1000 lines of code, and are designed to be a convenient starting point for writing your own training loop.

The feature set is intentionally kept minimal and focused. Users seeking maximum performance, MoE support, multi-node training, multidimensional parallelism, and other advanced features should use the `prime-rl` trainer. 

To use `vf.RLTrainer` in your own project, install with RL extras:
```bash
uv add 'verifiers[rl]'
```

Then, create a training configuration file, e.g. `configs/vf-rl/wiki-search.toml`, and do:

```bash
uv run vf-rl @ configs/vf-rl/wiki-search.toml
```

Example configuration files can be created in your project by running `uv run vf-setup`.

### Configuration

`vf-rl` can be used with a single TOML file, largely mirroring the configuration options for `prime-rl` but with some key differences in organization and feature sets.

Example configuration file for the `primeintellect/wiki-search` Environment with `Qwen/Qwen3-4B-Instruct-2507`:

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"

[env]
id = "primeintellect/wiki-search"

[env.args]
max_turns = 10

[inference]
gpus = 1

[inference.args]
enable_auto_tool_choice = true
tool_call_parser = "hermes"

[trainer]
gpus = 1

[trainer.args]
run_name = "wiki-search"
micro_batch_size = 4
rollouts_per_example = 16
batch_size = 1024
max_steps = 500
max_tokens = 512
max_seq_len = 4096
```

## Key Hyperparameters

### Batch Configuration

Key fields in `[trainer.args]`:
- `rollouts_per_example`: completions per prompt (group size)
- `micro_batch_size`: rollouts per GPU per step
- `batch_size`: rollouts per global batch (must be divisible by `micro_batch_size * world_size`)

**How to think about batch settings:**
- `rollouts_per_example`: Larger groups (16-32) increase reward diversity but increase training time and memory usage
- `micro_batch_size`: Limited by GPU memory after model weights
- `batch_size`: Total rollouts per global batch (must be divisible by `micro_batch_size` and `rollouts_per_example`)

### Generation Parameters

Both `prime-rl` and `vf-rl` support configurable generation parameters, including:
- `max_tokens`: maximum number of tokens to generate per turn
- `temperature`: temperature for sampling
- `top_p`: top-p sampling
- `top_k`: top-k sampling
- `min_p`: minimum probability for sampling
- `repetition_penalty`: repetition penalty for sampling

In `prime-rl`, these parameters are configured in the `[orchestrator.sampling]` section, and in `vf-rl`, they are configured in the `[trainer.args]` section.

### Training Schedule

Core fields in `[trainer.args]`:
- `learning_rate`, `lr_scheduler_type`, `warmup_steps`, `max_steps`
- `max_grad_norm`, `bf16`, `gradient_checkpointing`


## LoRA Training

LoRA training is supported in both `prime-rl` and `vf-rl`. In `prime-rl`, it can be configured via the `[trainer.model.experimental.lora]` section. In `vf-rl` it is enabled by default, and can be configured via the `[trainer.args]` section.


## RL Rules of Thumb

RL is notoriously sensitive to implementation details. Here's practical guidance:

### Before Training

1. **Evaluate baseline performance**: If your model gets 0% reward after 10+ attempts, the task is too hard
2. **Check task difficulty**: If baseline is already 80%+, consider harder examples
3. **Ensure reward diversity**: You want varied scores within each generation group

### Stability vs Performance Trade-offs

**For more aggressive training** (higher risk of collapse):
- Increase learning rate (3e-5 to 1e-4 for LoRA, 3e-6 to 1e-5 for full finetuning)
- Decrease `rollouts_per_example` and `batch_size` for faster generation

**For more stable training** (slower progress):
- Increase `rollouts_per_example` (16-32)
- Increase `batch_size` (512-1024)
- Use larger models (14B+)

However, the best way to improve training is ensuring appropriate task difficulty for your model - not too easy, not too hard.

## Troubleshooting

### Common Issues

**Non-Increasing Chat Templates:** The Qwen3 and DeepSeek-R1 model series both remove `<think>` sections from messages when processing inputs, which violates the increasing context requirement for multi-turn training. We provide versions of many of these models with modified chat templates [here](https://huggingface.co/collections/willcb/qwen3-68434f4883925bfdb4570ee5).

**OOM during generation:**
- Reduce `rollouts_per_example` or `micro_batch_size`
- Use LoRA instead of full finetuning
- Check vLLM server has sufficient memory

**Training instability:**
- Decrease learning rate
- Increase `rollouts_per_example`
- Increase `batch_size`

**Slow training:**
- Increase learning rate
- Leverage continuous rewards
- Use online difficulty filtering
- Calibrate difficulty appropriately via smarter models, easier tasks


## Next Steps

- Explore [Environments](environments.md) to create custom tasks
- Review [Components](components.md) for advanced patterns
- See the [examples directory](https://github.com/PrimeIntellect-ai/verifiers/tree/main/examples) on GitHub for complete training scripts
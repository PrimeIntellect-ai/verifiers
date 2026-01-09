# Training

This section covers how to use Verifiers environments for RL training with our Hosted Training platform, our open-source `prime-rl` trainer, or other supported libraries.

## Table of Contents

- [Hosted Training](#hosted-training)
    - [Configuration](#configuration)
- [Training with `prime-rl`](#training-with-prime-rl)
    - [Setup and Configuration](#setup-and-configuration)
- [Training with `vf.RLTrainer`](#training-with-vfrltrainer)
    - [Setup and Configuration](#setup-and-configuration)
    - [Generation Parameters](#generation-parameters)
    - [Training Schedule](#training-schedule)
- [RL Rules of Thumb](#rl-rules-of-thumb)
    - [Before Training](#before-training)
    - [Performance Trade-offs](#performance-trade-offs)
    - [Common Issues](#common-issues)
- [Other Trainers](#other-trainers)
    - [SkyRL](#skyrl)
    - [Tinker](#tinker)
    - [Integrating with Other Trainers](#integrating-with-other-trainers)

## Hosted Training

Hosted Training, available within our Lab platform, enables you to automatically train models via `prime-rl` without needing to manage your own infrastructure. Hosted Training supports LoRA for RL training, and can be used with any environment built with Verifiers. 

### Configuration

Use the `vf-setup` script to download example configuration files for Hosted Training into your workspace:

```bash
uv run vf-setup
```

This will download example TOML configs for Hosted Training into `configs/lab/`, along with `endpoints.py`:

```
configs/
├── endpoints.py
└── lab/
    ├── alphabet-sort.toml
    ├── gsm8k.toml
    ├── math-python.toml
    ├── reverse-text.toml
    ├── wiki-search.toml
    └── wordle.toml
```

Example configuration file for the `primeintellect/alphabet-sort` environment with `Qwen/Qwen3-30B-A3B-Instruct-2507`:

```toml
model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
max_steps = 500
batch_size = 256
rollouts_per_example = 8

[sampling]
max_tokens = 512

[[env]]
id = "primeintellect/alphabet-sort"
args = { min_turns = 3, max_turns = 5, power_per_turn = false }

[wandb]
project = "alphabet-sort"
name = "qwen3-30b-i-alphabet-sort"
```

We currently support the following models for Hosted Training:
- `Qwen/Qwen3-4B-Instruct-2507` 
- `Qwen/Qwen3-4B-Thinking-2507`
- `Qwen/Qwen3-30B-Instruct-2507`
- `Qwen/Qwen3-30B-Thinking-2507`
- `Qwen/Qwen3-235B-Instruct-2507`
- `Qwen/Qwen3-235B-Thinking-2507`
- `PrimeIntellect/INTELLECT-3`

Hosted Training is currently in Private Beta. For access, please fill out [this form](https://form.typeform.com/to/iYn9UliG).

## Training with `prime-rl`

Our [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) trainer is a production-ready async RL training framework that supports large-scale multi-node training, agentic rollouts with Verifiers environments, Mixture-of-Experts (MoE) models, LoRA adapters, and other training algorithms such as SFT and online distillation. We recommend using `prime-rl` for training with Verifiers environments on self-managed GPU infrastructure. The default configuration distills the best practices from our research team's experience and the broader community into a stable, easy-to-use recipe, including advanced features such as online difficulty filtering, continuous batching, in-flight weight updates, importance sampling and logprob clipping for stability, and more. 

### Setup and Configuration

To set up your workspace for training with `prime-rl`, run:
```bash
uv run vf-setup --prime-rl
```

This will clone and install the `prime-rl` trainer and its dependencies, and set up a default TOML config for training with the included `wiki-search` Environment on 8 GPUs.

Then, you can start training with:
```bash
uv run prime-rl @ configs/prime-rl/wiki-search.toml
```

This will launch a tmux session with separate panes for the trainer, orchestrator, and inference server. For further configuration options, see the [prime-rl documentation](https://docs.primeintellect.ai/prime-rl). 

## Training with `vf.RLTrainer`

If you want to hack on new training algorithms and are less concerned with maximum performance or advanced features, you can use the included `RLTrainer` (via `vf-rl`), whose core files are under 1000 lines of code and include only the most essential logic for fairly-performant async off-policy training (with a similar core algorithm as `prime-rl`).

The included `RLTrainer` is a minimal, hackable training loop based on `transformers.Trainer` that supports both full-parameter finetuning and LoRA training. `RLTrainer` can be viewed as a "baby" `prime-rl` that adopts a similar default training recipe (async CISPO with one-step off-policy overlap), intended for single-node test runs with dense models. The primary files (`trainer.py` and `orchestrator.py`, located in `verifiers/rl/trainer/`) are under 1000 lines of code, and are designed to be a convenient starting point for writing your own training loop.

The feature set is intentionally kept minimal and focused. Users seeking maximum performance, MoE support, multi-node training, multidimensional parallelism, and other advanced features should use the `prime-rl` trainer. 

### Setup and Configuration

To use `vf.RLTrainer` in your own project, install with RL extras:
```bash
uv add 'verifiers[rl]'
```

Then, use the `vf-setup` script to download example configuration files for `vf.RLTrainer` into your workspace:

```bash
uv run vf-setup --vf-rl
```
This will download example TOML configs for `vf.RLTrainer` into `configs/vf-rl/`, along with `endpoints.py`:

```
configs/
├── endpoints.py
└── vf-rl/
    ├── alphabet-sort.toml
    ├── gsm8k.toml
    ├── math-python.toml
    ├── reverse-text.toml
    ├── wiki-search.toml
    └── wordle.toml
```

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

To start a training run with `vf.RLTrainer`, do:

```bash
uv run vf-rl @ configs/vf-rl/wiki-search.toml
```

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

### Model loading

By default, `vf.RLTrainer` will use Liger Kernel for optimized training. To disable Liger Kernel, set `use_liger = false` in `[trainer.args]`.

## RL Rules of Thumb

RL training can be sensitive to implementation details and hyperparameters. Some simple practical guidance:

### Before Training

1. **Evaluate baseline performance**: If your model gets 0% reward after 10+ attempts, the task is too hard
2. **Check task difficulty**: If baseline is already 80%+, consider harder examples
3. **Ensure reward diversity**: You want varied scores within each generation group

### Performance Trade-offs

**For more aggressive training** (higher risk of collapse):
- Increase learning rate (1e-5 to 1e-4 for LoRA, 1e-6 to 1e-5 for full finetuning)
- Decrease `rollouts_per_example` and `batch_size` for faster generation

**For more stable training** (slower progress):
- Increase `rollouts_per_example` (16-32)
- Increase `batch_size` (512-1024)
- Use larger models (14B+)

The best way to improve training is to ensure appropriate task difficulty for your model. When using Hosted Training or `prime-rl`, you can enable online difficulty filtering to ensure that rollout groups used for training always contain a diversity of rewards.

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

## Other Trainers

`verifiers` is intended to be largely trainer-agnostic. It is supported by [SkyRL](https://github.com/novasky-ai/skyrl) and [Tinker](https://github.com/thinking-machines-lab/tinker), and is straightforward to support for any trainer which can expose an OpenAI-compatible inference client for rollouts.

### SkyRL

[Verifiers + SkyRL](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/integrations/verifiers)

### Tinker

[Verifiers + Tinker](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/verifiers_rl)

### Integrating with Other Trainers

TODO: Add instructions for integrating with other trainers.
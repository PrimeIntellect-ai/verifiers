# Training

This section covers how to use Verifiers environments for RL training with our Hosted Training platform, our open-source `prime-rl` trainer, or other supported libraries.

## Table of Contents

- [Hosted Training](#hosted-training)
    - [Configuration](#configuration)
- [Training with `prime-rl`](#training-with-prime-rl)
    - [Setup and Configuration](#setup-and-configuration)
- [Prompt Optimization with `prime gepa run`](#prompt-optimization-with-prime-gepa-run)
    - [Usage](#usage)
    - [Output](#output)
- [RL Rules of Thumb](#rl-rules-of-thumb)
    - [Before Training](#before-training)
    - [Performance Trade-offs](#performance-trade-offs)
    - [Common Issues](#common-issues)
- [Other Trainers](#other-trainers)
    - [Tinker](#tinker)
    - [SkyRL](#skyrl)
    - [rLLM](#rllm)
    - [Integrating with Other Trainers](#integrating-with-other-trainers)

## Hosted Training

Hosted Training, available within our Lab platform, enables you to automatically train models via `prime-rl` without needing to manage your own infrastructure. Hosted Training supports LoRA for RL training, and can be used with any environment built with Verifiers. 

### Configuration

Use the `prime lab setup` script to download example configuration files for Hosted Training into your workspace:

```bash
prime lab setup
```

This will download example TOML configs for Hosted Training into `configs/rl/`, example eval configs into `configs/eval/`, along with `configs/endpoints.toml` and GEPA starter configs in `configs/gepa/`:

```
configs/
├── endpoints.toml
├── eval/
│   ├── minimal.toml
│   └── multi-env.toml
├── rl/
│   ├── alphabet-sort.toml
│   ├── gsm8k.toml
│   ├── math-python.toml
│   ├── reverse-text.toml
│   ├── wiki-search.toml
│   └── wordle.toml
└── gepa/
    ├── base.toml
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
prime lab setup --prime-rl
```

This will clone and install the `prime-rl` trainer and its dependencies, and set up a default TOML config for training with the included `wiki-search` Environment on 8 GPUs.

Then, you can start training with:
```bash
uv run prime-rl @ configs/prime-rl/wiki-search.toml
```

This will launch a tmux session with separate panes for the trainer, orchestrator, and inference server. For further configuration options, see the [prime-rl documentation](https://docs.primeintellect.ai/prime-rl). 

## Prompt Optimization with `prime gepa run`

`prime gepa run` is the CLI entrypoint for automatic system prompt optimization using [GEPA](https://github.com/gepa-ai/gepa) (Genetic-Pareto prompt optimization). It iteratively refines your environment's system prompt using a teacher LLM to reflect on evaluation results, without requiring gradient-based training. Current support is for system prompt optimization only.

### Usage

Basic usage mirrors `prime eval run`:
```bash
prime gepa run wiki-search --model google/gemini-3-flash-preview
```

This will optimize the system prompt for the `wiki-search` environment using the specified model for both evaluation rollouts and reflection. Results are saved to `environments/wiki-search/outputs/gepa/`.

Key options:
- `--model` / `-m`: Model for evaluation rollouts
- `--reflection-model` / `-M`: Teacher model for prompt reflection (defaults to `--model`)
- `--max-calls` / `-B`: Evaluation budget (default: 500)
- `--num-train` / `-n`: Training examples (default: 100)
- `--num-val` / `-N`: Validation examples (default: 50)
- `--minibatch-size`: Number of examples evaluated together per reflection step (default: 3)
- `--perfect-score`: Maximum score for a rollout in your environment (if applicable); minibatches achieving this score are skipped during reflection (useful if your environment has a known max score)
- `--state-columns`: Additional state columns to copy into the reflection dataset. By default, `query`, `completion`, `expected_answer`, `reward`, and `error` are included. Use this to add environment-specific state fields (e.g., `--state-columns tool_calls reasoning_trace`)

### Output

After optimization, you'll find:
- `best_prompt.txt` - The optimized system prompt
- `pareto_frontier.jsonl` - Best prompts per validation example
- `metadata.json` - Run configuration and summary

Use `vf-eval` to verify performance before and after optimization.

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

`verifiers` is intended to be largely trainer-agnostic and is straightforward to support for any trainer which can expose an OpenAI-compatible inference client for rollouts.

### `vf.RLTrainer` (Legacy)

The legacy `vf.RLTrainer` still exists for educational and experimental purposes via the optional `verifiers-rl` package and the `vf-rl` entrypoint, but it is not actively maintained. It is a compact single-node async RL trainer with a narrower feature set than production trainers. Its core implementation (`trainer.py` and `orchestrator.py` under `packages/verifiers-rl/verifiers_rl/rl/trainer/`) remains intentionally lightweight for algorithm experimentation. For production training and current guidance, use [`prime-rl`](#training-with-prime-rl).

### Tinker

[Tinker](https://thinkingmachines.ai/tinker/) supports Verifiers environments via the `tinker-cookbook` recipes.

- [Verifiers + Tinker Recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/verifiers_rl)

### SkyRL

[SkyRL](https://github.com/NovaSky-AI/SkyRL) supports Verifiers environments via its `skyrl-train` integration.

- [Verifiers + SkyRL Integration](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/integrations/verifiers)

### rLLM

[rLLM](https://github.com/rllm-project/rllm) supports Verifiers environments with both [verl](https://github.com/volcengine/verl) (local GPU) and [Tinker](https://thinkingmachines.ai/tinker/) (remote GPU) backends.

- [Verifiers + rLLM Documentation](https://rllm-project.readthedocs.io/en/latest/examples/verifiers/)

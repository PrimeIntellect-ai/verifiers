# RLTrainer Tutorial: End-to-End Training Guide

This tutorial walks you through training a language model using `vf.RLTrainer` from start to finish. You'll learn how to set up your environment, configure training parameters, run a training job, and analyze results.

## What You'll Learn

By the end of this tutorial, you will:
- Set up a complete training workspace
- Configure and run your first RL training job
- Monitor training progress and understand the output
- Analyze training results and metrics
- Troubleshoot common issues

## Who This Tutorial Is For

This tutorial is designed for developers who:
- Have basic Python programming experience
- Understand fundamental machine learning concepts
- Want to train language models with reinforcement learning
- Are new to `vf.RLTrainer` or RL training in general

## Before You Begin

### Prerequisites

You need:
- Python 3.10 or later
- `uv` package manager installed
- At least one GPU with 16GB+ VRAM (for the examples in this tutorial)
- Basic familiarity with command-line tools

### Installation

First, install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the Prime CLI tool:

```bash
uv tool install prime
prime login
```

## Setting Up Your Workspace

### Create a New Project

Set up a new workspace for RL training:

```bash
prime lab setup --vf-rl
```

This command:
1. Creates a Python project (if needed)
2. Installs `verifiers` with RL extras
3. Sets up the recommended workspace structure
4. Downloads example configuration files

You should see output similar to:

```
✓ Created Python project
✓ Installed verifiers[rl]
✓ Created workspace structure
✓ Downloaded example configs
```

### Understanding the Workspace Structure

Your workspace now contains:

```
configs/
├── endpoints.py        # API endpoint configuration
└── vf-rl/             # Example training configs
    ├── alphabet-sort.toml
    ├── gsm8k.toml
    ├── math-python.toml
    ├── reverse-text.toml
    ├── wiki-search.toml
    └── wordle.toml
environments/
└── AGENTS.md          # Documentation for AI agents
```

The `configs/vf-rl/` folder contains example TOML configuration files for different environments. Each file defines the model, environment, and training parameters.

### Configure API Endpoints

Open `configs/endpoints.py` to configure your inference endpoints. By default, it uses Prime Inference:

```python
# configs/endpoints.py
import os

ENDPOINTS = {
    "default": {
        "base_url": "https://api.primeintellect.ai/v1",
        "api_key": os.getenv("PRIME_API_KEY"),
    }
}
```

You can add custom endpoints for local inference or other providers.

## Choosing Your First Environment

### Understanding Environments

Environments in Verifiers define:
- A dataset of task inputs
- A harness for the model (tools, context management, etc.)
- A reward function to score performance

For this tutorial, we'll use the `reverse-text` environment because it:
- Has a simple, well-defined task
- Provides clear success metrics
- Trains quickly on a single GPU
- Works well with small models

### The Reverse Text Task

The `reverse-text` environment asks the model to reverse a given text character-by-character. For example:

**Input:** "hello world"
**Expected output:** "dlrow olleh"

The model is scored using LCS (Longest Common Subsequence) similarity between its answer and the correct reversal.

## Configuring Your Training Job

### Understanding the Configuration File

Open `configs/vf-rl/reverse-text.toml` to see the default configuration:

```toml
model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

[env]
id = "primeintellect/reverse-text"

[inference]
gpus = 1

[inference.args]
enforce_eager = true

[trainer]
gpus = 1

[trainer.args]
run_name = "reverse-text"
micro_batch_size = 16
rollouts_per_example = 16
batch_size = 128
max_steps = 100
max_tokens = 128
max_seq_len = 512
```

### Key Configuration Parameters

Let's understand what each parameter does:

**Model Configuration:**
- `model`: The base model to train. We're using a small 0.6B parameter model that's been pre-trained on the reverse text task.

**Environment Configuration:**
- `env.id`: The environment to use for training. This references the `primeintellect/reverse-text` environment from the Environments Hub.

**Inference Configuration:**
- `inference.gpus`: Number of GPUs to use for generating rollouts (completions).
- `inference.args.enforce_eager`: Forces eager execution mode in vLLM for compatibility.

**Trainer Configuration:**
- `trainer.gpus`: Number of GPUs to use for training.
- `run_name`: Name for this training run (used in logs and checkpoints).
- `micro_batch_size`: Number of rollouts processed per GPU per training step.
- `rollouts_per_example`: Number of completions to generate for each prompt (group size).
- `batch_size`: Total number of rollouts per global training batch.
- `max_steps`: Total number of training steps to perform.
- `max_tokens`: Maximum tokens to generate per completion.
- `max_seq_len`: Maximum sequence length for training.

### How Batch Settings Work Together

The batch settings control how training data is generated and processed:

1. **`rollouts_per_example`** (16): For each prompt, generate 16 different completions. This creates diversity in the training data and helps the model learn from comparing good and bad attempts.

2. **`micro_batch_size`** (16): Process 16 rollouts at a time on each GPU. This is limited by GPU memory.

3. **`batch_size`** (128): Use 128 total rollouts per training step. With 1 GPU and `micro_batch_size=16`, this means 8 micro-batches per step.

The relationship must satisfy:
```
batch_size % (micro_batch_size × num_gpus) = 0
```

### Customizing the Configuration

For this tutorial, we'll use the default configuration. However, you can customize it by creating a copy:

```bash
cp configs/vf-rl/reverse-text.toml configs/vf-rl/my-reverse-text.toml
```

Then edit `my-reverse-text.toml` to adjust parameters like:
- `max_steps`: Increase to 200 for longer training
- `learning_rate`: Add `learning_rate = 1e-5` under `[trainer.args]` to adjust learning speed
- `temperature`: Add `temperature = 0.8` under `[trainer.args]` to control sampling randomness

## Running Your First Training Job

### Starting the Training

Start training with:

```bash
uv run vf-rl @ configs/vf-rl/reverse-text.toml
```

This command:
1. Loads the configuration from the TOML file
2. Starts a vLLM inference server for generating rollouts
3. Initializes the RLTrainer
4. Begins the training loop

### Understanding the Output

When training starts, you'll see several stages of output:

**1. Initialization:**
```
Loading model PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT...
Starting vLLM server on port 8000...
Waiting for vLLM server to be ready...
vLLM server is ready
Loading environment primeintellect/reverse-text...
```

**2. Training Steps:**

Each training step shows:
```
{'loss': 0.234, 'learning_rate': 1e-05, 'epoch': 0.01}
Step 1/100: reward=0.45, entropy=2.34, importance_ratio=1.02
```

**3. Sample Completions:**

Periodically, you'll see example prompts and completions:
```
=== Step 5 Sample ===
Prompt: Reverse the text: "hello"
Completion: <reversed_text>olleh</reversed_text>
Reward: 1.0
```

### Monitoring Progress

Key metrics to watch:

- **`reward`**: Average reward across rollouts. Higher is better. For reverse-text, 1.0 means perfect reversal.
- **`loss`**: Training loss. Should generally decrease over time.
- **`entropy`**: Diversity of model outputs. Too low means the model is becoming deterministic.
- **`importance_ratio`**: Ratio between current and reference policy. Should stay close to 1.0.

### Expected Training Time

On a single GPU (e.g., RTX 4090), training for 100 steps typically takes:
- **Setup**: 2-3 minutes (loading model, starting vLLM)
- **Training**: 15-20 minutes (depends on GPU and batch size)
- **Total**: ~20-25 minutes

## Analyzing Results

### Finding Your Outputs

After training completes, you'll find outputs in:

```
outputs/reverse-text/
├── checkpoint-50/          # Checkpoint at step 50
├── checkpoint-100/         # Final checkpoint
└── runs/                   # Training logs
```

### Understanding Checkpoints

Each checkpoint contains:
- `adapter_model.safetensors`: LoRA adapter weights (if using LoRA)
- `adapter_config.json`: LoRA configuration
- `trainer_state.json`: Training state (step, metrics, etc.)

By default, `vf.RLTrainer` uses LoRA (Low-Rank Adaptation) to train only a small set of parameters, making training faster and more memory-efficient.

### Interpreting Training Metrics

Let's look at what good training looks like:

**Reward progression:**
```
Step 1:   reward=0.35
Step 25:  reward=0.52
Step 50:  reward=0.68
Step 75:  reward=0.79
Step 100: reward=0.85
```

This shows steady improvement. The model is learning to reverse text more accurately.

**Loss progression:**
```
Step 1:   loss=0.45
Step 25:  loss=0.32
Step 50:  loss=0.21
Step 75:  loss=0.15
Step 100: loss=0.12
```

Decreasing loss indicates the model is learning the task.

**Entropy:**
```
Step 1:   entropy=3.2
Step 50:  entropy=2.8
Step 100: entropy=2.5
```

Slightly decreasing entropy is normal as the model becomes more confident. Very low entropy (< 1.0) might indicate the model is becoming too deterministic.

### Visualizing Results with Weights & Biases

If you configured Weights & Biases (wandb), you can view detailed metrics:

1. Training curves (reward, loss, entropy over time)
2. Sample completions at each step
3. Distribution of rewards across rollouts
4. Learning rate schedule

To enable wandb logging, add to your TOML config:

```toml
[trainer.args]
report_to = "wandb"

[trainer.args.wandb]
project = "reverse-text-tutorial"
name = "my-first-run"
```

## Advanced Configuration

### Adjusting Batch Sizes

If you encounter out-of-memory errors, reduce batch sizes:

```toml
[trainer.args]
micro_batch_size = 8      # Reduce from 16
batch_size = 64           # Reduce from 128
rollouts_per_example = 8  # Reduce from 16
```

Smaller batches mean:
- Less memory usage
- Faster iteration
- But potentially less stable training

### Configuring Generation Parameters

Control how the model generates completions:

```toml
[trainer.args]
temperature = 0.8        # Lower = more deterministic (default: 1.0)
top_p = 0.9             # Nucleus sampling threshold (default: 1.0)
max_tokens = 256        # Max tokens per completion (default: 512)
```

**Temperature:**
- Lower (0.5-0.8): More focused, deterministic outputs
- Higher (1.0-1.5): More diverse, creative outputs

**Top-p (nucleus sampling):**
- Lower (0.8-0.9): Sample from top probability tokens only
- Higher (0.95-1.0): Consider more token options

### Using Full Finetuning Instead of LoRA

To train all model parameters instead of using LoRA:

```toml
[trainer.args]
use_lora = false
learning_rate = 1e-6    # Use lower LR for full finetuning
```

Note: Full finetuning requires significantly more GPU memory and is slower than LoRA.

### Adjusting Learning Rate and Schedule

```toml
[trainer.args]
learning_rate = 1e-5
lr_scheduler_type = "cosine"  # Options: constant, linear, cosine
warmup_steps = 10             # Gradual LR increase at start
```

**Learning rate guidelines:**
- LoRA: 1e-5 to 1e-4
- Full finetuning: 1e-6 to 1e-5

Start conservative and increase if training is too slow.

## Troubleshooting Common Issues

### Out of Memory (OOM) Errors

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Reduce `micro_batch_size`:
   ```toml
   [trainer.args]
   micro_batch_size = 4  # Reduce from 16
   ```

2. Reduce `max_seq_len`:
   ```toml
   [trainer.args]
   max_seq_len = 1024  # Reduce from 2048
   ```

3. Enable gradient checkpointing:
   ```toml
   [trainer.args]
   gradient_checkpointing = true
   ```

4. Use LoRA instead of full finetuning:
   ```toml
   [trainer.args]
   use_lora = true
   ```

### Training Instability

**Symptom:**
Reward or loss fluctuates wildly, or training diverges.

**Solutions:**

1. Decrease learning rate:
   ```toml
   [trainer.args]
   learning_rate = 5e-6  # Reduce from 1e-5
   ```

2. Increase `rollouts_per_example` for more stable gradients:
   ```toml
   [trainer.args]
   rollouts_per_example = 32  # Increase from 16
   ```

3. Increase `batch_size`:
   ```toml
   [trainer.args]
   batch_size = 256  # Increase from 128
   ```

### Slow Training

**Symptom:**
Training takes much longer than expected.

**Solutions:**

1. Reduce `rollouts_per_example`:
   ```toml
   [trainer.args]
   rollouts_per_example = 8  # Reduce from 16
   ```

2. Reduce `batch_size`:
   ```toml
   [trainer.args]
   batch_size = 64  # Reduce from 128
   ```

3. Reduce `max_tokens`:
   ```toml
   [trainer.args]
   max_tokens = 128  # Reduce from 512
   ```

### vLLM Server Connection Issues

**Symptom:**
```
ConnectionError: Could not connect to vLLM server
```

**Solutions:**

1. Check if port 8000 is already in use:
   ```bash
   lsof -i :8000
   ```

2. Use a different port:
   ```toml
   [trainer.args]
   vllm_server_port = 8001
   ```

3. Increase connection timeout:
   ```toml
   [trainer.args]
   vllm_server_timeout = 600.0  # Increase from 300.0
   ```

### Model Not Improving

**Symptom:**
Reward stays flat or very low throughout training.

**Possible causes and solutions:**

1. **Task is too difficult**: Check baseline performance first:
   ```bash
   prime eval run reverse-text -m PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT -n 20
   ```
   If baseline reward is near 0%, the task may be too hard for this model.

2. **Learning rate too low**: Increase learning rate:
   ```toml
   [trainer.args]
   learning_rate = 2e-5  # Increase from 1e-5
   ```

3. **Not enough diversity**: Increase temperature:
   ```toml
   [trainer.args]
   temperature = 1.2  # Increase from 1.0
   ```

## Next Steps

### Try More Complex Environments

Now that you've completed your first training run, try these environments:

1. **alphabet-sort**: Sort letters alphabetically
   ```bash
   uv run vf-rl @ configs/vf-rl/alphabet-sort.toml
   ```

2. **gsm8k**: Grade school math problems
   ```bash
   uv run vf-rl @ configs/vf-rl/gsm8k.toml
   ```

3. **wiki-search**: Multi-turn environment with tool use
   ```bash
   uv run vf-rl @ configs/vf-rl/wiki-search.toml
   ```

### Transition to Production Training

For production-scale training, use `prime-rl` instead of `vf.RLTrainer`:

```bash
prime lab setup --prime-rl
uv run prime-rl @ configs/prime-rl/wiki-search.toml
```

`prime-rl` offers:
- Multi-node distributed training
- Mixture-of-Experts (MoE) model support
- Advanced features (online difficulty filtering, continuous batching, etc.)
- Production-ready stability and performance

See the [prime-rl documentation](https://docs.primeintellect.ai/prime-rl) for details.

### Create Your Own Environment

Build a custom environment for your specific task:

```bash
prime env init my-custom-env
```

See the [Environments documentation](docs/environments.md) for a complete guide.

### Explore Hosted Training

Use Prime Intellect's Hosted Training platform to train without managing infrastructure:

1. Configure your environment in `configs/lab/`
2. Submit a training job through the [Prime Intellect dashboard](https://app.primeintellect.ai/dashboard/training)
3. Monitor progress and download checkpoints

See the [Hosted Training documentation](docs/training.md#hosted-training) for details.

## Further Reading

- [Training Reference](docs/training.md): Complete reference for all training options
- [Environments Guide](docs/environments.md): Learn how to create custom environments
- [Evaluation Guide](docs/evaluation.md): Evaluate your trained models
- [API Reference](docs/reference.md): Detailed API documentation
- [prime-rl Documentation](https://docs.primeintellect.ai/prime-rl): Production training framework

## Summary

In this tutorial, you:
- Set up a complete RL training workspace
- Configured and ran your first training job with `vf.RLTrainer`
- Learned how to monitor training progress and interpret metrics
- Analyzed training results and checkpoints
- Explored advanced configuration options
- Troubleshot common issues

You're now ready to train language models with reinforcement learning using Verifiers!
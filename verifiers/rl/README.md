## `RLTrainer`

`RLTrainer` is the included RL trainer for `verifiers` environments, built on top of `transformers`, `accelerate` and `vllm`, and which supports both full-parameter finetuning and LoRA training. It is primarily intended for small-scale test runs on a single node with dense models. Users seeking maximum performance, MoE support, multi-node training, multidimensional parallelism, and other advanced features should use the external `prime-rl` trainer; `RLTrainer` can be viewed as a "baby" `prime-rl` that adopts a similar default training recipe (async RL with group-based advantages and ), and is a good starting point for beginners.

### Installation

Install with RL extras:

```bash
uv add 'verifiers[rl]'
```

Install from GitHub main with RL extras:

```bash
uv add 'verifiers[rl] @ git+https://github.com/PrimeIntellect-ai/verifiers.git@main'
```

If you already have the project set up and want to include the `rl` extra:

```bash
uv sync --extra rl
```

### TOML configuration files

`vf-rl` consumes a single TOML file that defines the model, environment, vLLM (inference) process, and trainer.

- Required keys:
  - `model` (string)
  - `[env].id` (string; environment slug)
  - `[inference].gpus` (int; number of GPUs for vLLM)
  - `[trainer].gpus` (int; number of GPUs for training)
- Optional `*.args` tables forward keyword arguments to their respective CLIs:
  - `[inference.args]` → forwarded to `vf-vllm` (keys converted to `--kebab-case` flags)
  - `[trainer.args]` → mapped to `RLConfig` (see Configuration below)

Minimal example:

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"

[env]
id = "kalomaze/alphabet-sort"

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

See more examples under `configs/rl/` (e.g., `reverse-text.toml`, `alphabet-sort.toml`).

### Running with `vf-rl`

`vf-rl` creates a tmux session with two panes: top runs `vf-vllm` (inference server), bottom runs `vf-train` (trainer). GPU assignment is contiguous: inference uses the first `inference.gpus` devices, trainer uses the next `trainer.gpus` devices.

Usage:

```bash
uv run vf-rl @ configs/rl/config.toml -s session-name
```

- `-s/--session`: tmux session name (default: `vf-rl`)
- Requires `tmux` in `PATH`

### Configuration

We have removed a number of features from the previous `GRPOTrainer`, in favor of a more streamlined, opinionated, and hackable training recipe. The primary parameters most users will want to configure are:
- LoRA configuration arguments:
    - `use_lora`: whether to use LoRA training (default is `True`)
    - `lora_rank`: the rank of the LoRA modules (default is `16`)
    - `lora_alpha`: the alpha of the LoRA modules (default is `16`)
    - `lora_dropout`: the dropout of the LoRA modules (default is `0.0`)
    - `lora_target_modules`: the target modules for the LoRA modules (default is `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]`)
    - `lora_modules_to_save`: modules for full-parameter finetuning (instead of LoRA modules; default is `None`)
    - `lora_use_rslora`: whether to use RSLoRA (default is `False`)
- Training configuration arguments:
  - `learning_rate`: the learning rate for the training (default is `1e-5`)
  - `micro_batch_size`: rollouts per GPU per gradient accumulation step (default is `8`)
  - `batch_size`: rollouts per global batch (default is `512`)
  - `rollouts_per_example`: rollouts per example/prompt (default is `16`)
  - `max_seq_len`: the maximum sequence length for the training (default is `2048`)
  - `max_steps`: the maximum number of steps for the training (default is `500`)
- Sampling configuration arguments:
  - `max_tokens`: the maximum number of tokens per request (default is `None`)
  - `temperature`: the temperature for the sampling (default is `0.7`)
  - `top_p`: the top-p value for the sampling (default is `1.0`)
  - `top_k`: the top-k value for the sampling (default is `None`)
  - `min_p`: the minimum probability value for the sampling (default is `0.0`)
  - `repetition_penalty`: the repetition penalty for the sampling (default is `1.0`)
  - `presence_penalty`: the presence penalty for the sampling (default is `0.0`)
  - `frequency_penalty`: the frequency penalty for the sampling (default is `0.0`)

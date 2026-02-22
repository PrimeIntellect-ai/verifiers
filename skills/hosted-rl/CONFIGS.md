# Configs Reference

## Hosted Training TOML

```toml
model = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # run `prime rl models` for current list
max_steps = 100
batch_size = 256
rollouts_per_example = 8

[sampling]
max_tokens = 512

[[env]]
id = "primeintellect/alphabet-sort"
args = { min_turns = 3, max_turns = 5 }

# Optional: multiple envs for multi-env training
# [[env]]
# id = "primeintellect/another-env"
```

### Optional Fields

```toml
learning_rate = 1e-4
lora_alpha = 16
oversampling_factor = 2.0
max_async_level = 2
trajectory_strategy = "interleaved"  # or "branching"
env_file = ["secrets.env"]           # load secrets from file(s)

[wandb]
project = "my-project"
name = "my-run"
entity = "my-team"

[eval]
interval = 100
eval_base_model = true
num_examples = -1
rollouts_per_example = 1

[[eval.env]]
id = "primeintellect/eval-env"
args = { split = "test" }
num_examples = 30
rollouts_per_example = 4

[val]
num_examples = 64
rollouts_per_example = 1
interval = 5

[buffer]
online_difficulty_filtering = false
easy_threshold = 0.8
hard_threshold = 0.2
easy_fraction = 0.0
hard_fraction = 0.0
env_ratios = [0.5, 0.5]
seed = 42
```

## Eval TOML

```toml
model = "openai/gpt-4.1-mini"    # global defaults
num_examples = 50

[[eval]]
env_id = "gsm8k"
num_examples = 100
rollouts_per_example = 5
[eval.env_args]
difficulty = "hard"

[[eval]]
env_id = "math-python"           # uses global defaults
```

## Run Size Guidelines

### Small (Validation)
- Model: smallest available (e.g., 4B)
- `max_steps=50`, `batch_size=128`, `rollouts_per_example=8`
- Verify: rewards non-zero and diverse, env runs without errors

### Medium (Experimentation)
- Model: mid-size (e.g., 30B)
- `max_steps=200–500`, `batch_size=256`, `rollouts_per_example=8–16`
- Enable W&B, add `[eval]` section

### Large (Production)
- Model: largest available
- `max_steps=500+`, `batch_size=512`, `rollouts_per_example=16`
- Enable difficulty filtering, `[eval]` + `[val]` sections

## RL Tuning Guidelines

- Evaluate baseline first: 0% after 10+ attempts → too hard. 80%+ → too easy.
- More stable: increase `rollouts_per_example` (16-32), `batch_size` (512-1024), larger models
- More aggressive: increase LR, decrease batch size (faster but riskier)
